import json
import logging
import os
import re
import time
from pathlib import Path

from .prompts import CATEGORIES, PRIORITIES, SYSTEM_PROMPT, build_user_message

BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "models/supportpilot-lora")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.30"))   # retrieval cut-off: tune on your own eval set
PASSAGES_IN_PROMPT = 1
LOGGER = logging.getLogger("supportpilot.inference")

FALLBACK = {
    "category": "other",
    "priority": "medium",
    "needs_human": True,
    "draft_reply": "Thanks for reaching out. A member of our support team will review your message and reply soon.",
    "reason": "The model output could not be validated, so this ticket was escalated.",
}


def pick_device() -> str:
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_assistant(use_adapter: bool = True):
    """Load the base model once (and the LoRA adapter if present). Returns (tokenizer, model, device)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device()
    dtype = torch.float16 if device in ("mps", "cuda") else torch.float32
    adapter_dir = Path(ADAPTER_PATH).resolve()
    LOGGER.warning(
        "assistant.load.begin base_model=%s adapter_enabled=%s adapter_path=%s device=%s dtype=%s",
        BASE_MODEL, use_adapter, adapter_dir, device, dtype,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    except Exception as exc:
        LOGGER.error(
            "assistant.load.failed stage=base_model exception_type=%s exception_message=%r",
            type(exc).__name__, str(exc),
        )
        raise RuntimeError(
            f"Could not load base model '{BASE_MODEL}'. Check your internet connection, "
            "Hugging Face access, and available disk space, then retry."
        ) from exc
    if use_adapter:
        if not (adapter_dir / "adapter_config.json").is_file() or not (
            adapter_dir / "adapter_model.safetensors"
        ).is_file():
            raise FileNotFoundError(
                f"LoRA adapter is incomplete at '{adapter_dir}'. Expected adapter_config.json "
                "and adapter_model.safetensors. Disable the adapter toggle to run the base model."
            )
        config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
        adapter_base = config.get("base_model_name_or_path")
        if adapter_base and adapter_base != BASE_MODEL:
            raise ValueError(
                f"Adapter expects base model '{adapter_base}', but BASE_MODEL is '{BASE_MODEL}'."
            )
        from peft import PeftModel
        try:
            model = PeftModel.from_pretrained(model, str(adapter_dir))
        except Exception as exc:
            LOGGER.error(
                "assistant.load.failed stage=adapter exception_type=%s exception_message=%r",
                type(exc).__name__, str(exc),
            )
            raise RuntimeError(
                f"Could not load LoRA adapter from '{adapter_dir}'. Verify it matches "
                f"'{BASE_MODEL}' and that its files are intact."
            ) from exc
    model.to(device).eval()
    LOGGER.warning(
        "assistant.load.complete model_class=%s base_model=%s adapter_enabled=%s adapter_path=%s",
        type(model).__name__, BASE_MODEL, use_adapter, adapter_dir if use_adapter else "disabled",
    )
    return tokenizer, model, device


def _completion_for_log(completion: str, customer_message: str) -> str:
    """Log generated text while scrubbing any verbatim copy of the input message."""
    customer_message = customer_message.strip()
    if customer_message:
        completion = re.sub(re.escape(customer_message), "[customer message redacted]", completion,
                            flags=re.IGNORECASE)
    return completion


def parse_output(text: str):
    """Extract and validate the JSON object. Returns (result, is_valid).
    Anything malformed becomes a safe 'escalate to a human' fallback."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return dict(FALLBACK), False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return dict(FALLBACK), False
    if not isinstance(data, dict):
        return dict(FALLBACK), False

    category, priority = data.get("category"), data.get("priority")
    needs_human, draft, reason = data.get("needs_human"), data.get("draft_reply"), data.get("reason")
    ok = (
        category in CATEGORIES and priority in PRIORITIES and isinstance(needs_human, bool)
        and isinstance(draft, str) and draft.strip() and isinstance(reason, str)
    )
    if not ok:
        return dict(FALLBACK), False
    return {
        "category": category,
        "priority": priority,
        "needs_human": needs_human,
        "draft_reply": draft.strip()[:1200],
        "reason": reason.strip()[:400],
    }, True


def generate_from_messages(messages, tokenizer, model, device, max_new_tokens: int = 256):
    """Greedy generation from chat messages. Returns (raw_text, seconds)."""
    import torch
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    customer_message = next(
        (message["content"] for message in reversed(messages) if message.get("role") == "user"),
        "",
    )
    LOGGER.warning(
        "generation.invoked=true model_class=%s base_model=%s adapter_path=%s device=%s prompt_token_count=%d max_new_tokens=%d",
        type(model).__name__, BASE_MODEL,
        Path(ADAPTER_PATH).resolve() if hasattr(model, "peft_config") else "disabled",
        device, inputs["input_ids"].shape[1], max_new_tokens,
    )
    start = time.perf_counter()
    try:
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                repetition_penalty=1.0, pad_token_id=tokenizer.eos_token_id,
            )
    except Exception as exc:
        LOGGER.error(
            "generation.failed model_class=%s base_model=%s adapter_path=%s exception_type=%s exception_message=%r",
            type(model).__name__, BASE_MODEL,
            Path(ADAPTER_PATH).resolve() if hasattr(model, "peft_config") else "disabled",
            type(exc).__name__, _completion_for_log(str(exc), customer_message),
        )
        raise
    seconds = time.perf_counter() - start
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
    LOGGER.warning(
        "generation.complete generated_token_count=%d completion_before_parse=%r",
        new_tokens.shape[-1], _completion_for_log(completion, customer_message),
    )
    return completion, seconds


def generate_reply(question, passages, tokenizer, model, device):
    """Retrieval results in -> validated ticket dict out."""
    usable = [p for p in passages if p["score"] >= MIN_SCORE][:PASSAGES_IN_PROMPT]
    docs_text = "\n".join(p["text"] for p in usable)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, docs_text)},
    ]
    raw, seconds = generate_from_messages(messages, tokenizer, model, device)
    result, valid = parse_output(raw)
    LOGGER.warning("generation.parse valid_json=%s evidence_used=%d", valid, len(usable))
    if not usable:                       # no evidence: never use a model-generated policy claim
        result.update(dict(FALLBACK))
        result["reason"] = "No documentation met the similarity threshold; human review is required."

    billing_issue = result["category"] == "billing" or re.search(
        r"\b(bill(?:ing)?|charg(?:e|ed|es)|payment|credit card|refund|invoice|subscription)\b",
        question,
        re.I,
    )
    security_sensitive = re.search(
        r"\b(security|hack(?:ed)?|compromised|unauthori[sz]ed|someone else|unknown login|"
        r"suspicious login|account takeover|security breach|stolen account)\b",
        question,
        re.I,
    )
    if billing_issue:
        result["needs_human"] = True
        if "Billing issues require human review." not in result["reason"]:
            result["reason"] = (result["reason"] + " Billing issues require human review.")[:400]
    if security_sensitive:
        result["needs_human"] = True
        result["priority"] = "high"
        if "Security-sensitive requests require immediate human review." not in result["reason"]:
            result["reason"] = (result["reason"] + " Security-sensitive requests require immediate human review.")[:400]
    result.update(valid_json=valid, latency_s=round(seconds, 2), raw=raw, evidence_used=len(usable))
    return result
