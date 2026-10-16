"""Usage:
  python -m scripts.run_eval --name baseline --baseline      # original Qwen, no adapter
  python -m scripts.run_eval --name finetuned                # Qwen + your LoRA adapter
"""
import argparse
import json
from pathlib import Path

from src.assistant import generate_from_messages, load_assistant
from src.evaluation import evaluate, load_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--baseline", action="store_true", help="run WITHOUT the LoRA adapter")
    ap.add_argument("--eval", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = load_jsonl(args.eval)[: args.limit]
    print(f"Evaluating '{args.name}' on {len(rows)} examples (adapter={'no' if args.baseline else 'yes'})")
    tok, model, device = load_assistant(use_adapter=not args.baseline)
    records, summary = evaluate(rows, lambda m: generate_from_messages(m, tok, model, device))

    out = Path("results"); out.mkdir(exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    with open(out / f"{args.name}_preds.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
