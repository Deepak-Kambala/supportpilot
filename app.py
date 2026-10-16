import json
import hashlib
import logging
import os
import time
from pathlib import Path

import streamlit as st

from src.assistant import (
    MIN_SCORE, PASSAGES_IN_PROMPT, _completion_for_log, generate_reply, load_assistant,
)
from src.knowledge_base import build_index, read_document, retrieve

st.set_page_config(page_title="SupportPilot", page_icon="🛟", layout="wide")
FEEDBACK_FILE = Path("data/feedback.jsonl")
DEMO_DIR = Path("data/demo_docs")
LOGGER = logging.getLogger("supportpilot.app")

if not st.session_state.get("source_logged"):
    LOGGER.warning("app.start source=%s pid=%s", Path(__file__).resolve(), os.getpid())
    st.session_state["source_logged"] = True


@st.cache_resource(show_spinner="Loading model (the first run takes a minute)...")
def get_model(use_adapter: bool):
    return load_assistant(use_adapter=use_adapter)


def collect_documents(uploads, use_demo):
    docs = []
    if uploads:
        for f in uploads:
            try:
                docs.append({"filename": f.name, "text": read_document(f.name, f.getvalue())})
            except Exception as e:
                st.sidebar.error(f"{f.name}: {e}")
    elif use_demo and DEMO_DIR.exists():
        for p in sorted(DEMO_DIR.glob("*")):
            docs.append({"filename": p.name, "text": read_document(p.name, p.read_bytes())})
    return docs


def save_feedback(rating, last, edited_draft):
    FEEDBACK_FILE.parent.mkdir(exist_ok=True)
    row = {"ts": time.time(), "rating": rating, "question": last["question"],
           "model_result": {k: last["result"][k] for k in ("category", "priority", "needs_human", "draft_reply", "reason")},
           "edited_draft": edited_draft}
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


st.title("🛟 SupportPilot")
st.caption("AI-assisted support inbox. Drafts and recommends only - it never sends replies or changes accounts.")

with st.sidebar:
    st.header("1. Knowledge base")
    uploads = st.file_uploader("Upload PDF, TXT or Markdown", type=["pdf", "txt", "md"], accept_multiple_files=True)
    use_demo = st.checkbox("Use bundled demo docs (fictional)", value=True, disabled=bool(uploads))
    st.header("2. Model")
    use_adapter = st.toggle("Use fine-tuned LoRA adapter", value=True)
    st.caption(f"Evidence below similarity {MIN_SCORE} is ignored and the ticket is escalated.")

docs = collect_documents(uploads, use_demo)
if not docs:
    st.info("Upload a help document (or tick the demo docs) to begin.")
    st.stop()

kb_key = tuple((d["filename"], hashlib.sha256(d["text"].encode("utf-8")).hexdigest()) for d in docs)
if st.session_state.get("kb_key") != kb_key:
    with st.spinner("Indexing documents..."):
        try:
            st.session_state["index"], st.session_state["chunks"] = build_index(docs)
            st.session_state["kb_key"] = kb_key
        except Exception as e:
            st.error(
                "Could not build the knowledge base. Check that the embedding dependencies "
                "are installed and that the embedding model can be downloaded. "
                f"Details: {e}"
            )
            st.stop()
index, chunks = st.session_state["index"], st.session_state["chunks"]
st.sidebar.success(f"{len(docs)} document(s), {len(chunks)} chunks indexed")

with st.form("ticket_form"):
    question = st.text_area("Customer message", placeholder="Example: How do I invite a teammate?", height=120)
    submitted = st.form_submit_button("Generate support draft")

if submitted and question.strip():
    try:
        tokenizer, model, device = get_model(use_adapter)
    except Exception as e:
        st.error(f"Could not load the model: {e}")
        st.stop()
    try:
        with st.spinner("Analyzing the request..."):
            passages = retrieve(question, index, chunks)
            result = generate_reply(question, passages, tokenizer, model, device)
    except Exception as e:
        LOGGER.error(
            "ticket.inference.failed exception_type=%s exception_message=%r",
            type(e).__name__, _completion_for_log(str(e), question),
        )
        st.error(f"Inference failed ({type(e).__name__}); no draft was generated. Please check the model setup and restart SupportPilot.")
        st.stop()
    st.session_state["ticket_no"] = st.session_state.get("ticket_no", 0) + 1
    st.session_state["last"] = {"question": question, "passages": passages, "result": result}

last = st.session_state.get("last")
if last:
    result, passages = last["result"], last["passages"]
    left, right = st.columns([3, 2])
    with left:
        st.subheader("Suggested response")
        c1, c2, c3 = st.columns(3)
        c1.metric("Category", result["category"])
        c2.metric("Priority", result["priority"])
        c3.metric("Latency", f"{result['latency_s']} s")
        if result["needs_human"]:
            st.warning("Needs review: human review required before anything is sent")
        else:
            st.info("Draft ready: review the draft before sending")
        if not result["valid_json"]:
            st.error("The model output was malformed, so this ticket was escalated automatically.")
        draft = st.text_area("Edit the reply", value=result["draft_reply"], height=160,
                             key=f"draft_{st.session_state['ticket_no']}")
        st.write("**Reason:**", result["reason"])
        b1, b2, b3 = st.columns(3)
        for col, label in zip((b1, b2, b3), ("Useful", "Incorrect", "Needs escalation")):
            if col.button(label, key=f"{label}_{st.session_state['ticket_no']}"):
                save_feedback(label, last, draft)
                st.toast(f"Feedback saved: {label}")
    with right:
        st.subheader("Supporting evidence")
        if not passages:
            st.warning("No relevant documents were retrieved.")
        for i, p in enumerate(passages):
            used = p["score"] >= MIN_SCORE and i < PASSAGES_IN_PROMPT
            with st.expander(f"{p['source']}  |  similarity {p['score']:.2f}  |  {'used' if used else 'ignored'}"):
                st.write(p["text"])
    with st.expander("Raw model output (debug)"):
        st.code(result["raw"])
    st.caption("Demo only: this app does not send replies or change accounts.")
