"""python -m scripts.make_dataset
Builds data/train.jsonl, data/eval.jsonl and data/demo_docs/acme_faq.md from scripts/scenarios.py.
The split is BY SCENARIO (not by row) so paraphrases never leak from train into eval."""
import json
import random
from pathlib import Path

from src.prompts import NO_DOCS, SYSTEM_PROMPT, build_user_message
from .scenarios import SCENARIOS

random.seed(7)
NO_EVIDENCE_REPLY = ("I don't have verified information to answer this yet. "
                     "I'll flag it for the support team so they can follow up.")
NO_EVIDENCE_REASON = "The available documentation does not answer the question."


def example(question, docs_text, answer):
    return {"messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(question, docs_text)},
        {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
    ]}


def answer_of(s):
    return {"category": s["category"], "priority": s["priority"], "needs_human": s["needs_human"],
            "draft_reply": s["reply"], "reason": s["reason"]}


def distractors(current_id, k):
    pool = [s["doc"] for s in SCENARIOS if s["doc"] and s["id"] != current_id]
    return random.sample(pool, k)


def join(docs):
    docs = docs[:]
    random.shuffle(docs)
    return "\n".join(docs)


def build(scenarios):
    rows = []
    for s in scenarios:
        for q in s["questions"]:
            if s["doc"]:
                rows.append(example(q, s["doc"], answer_of(s)))                                   # clean evidence
                rows.append(example(q, join([s["doc"]] + distractors(s["id"], random.choice([1, 2]))), answer_of(s)))  # noisy retrieval
                miss = {**answer_of(s), "needs_human": True, "draft_reply": NO_EVIDENCE_REPLY, "reason": NO_EVIDENCE_REASON}
                rows.append(example(q, join(distractors(s["id"], random.choice([1, 2]))), miss))  # answer NOT in retrieved docs
            else:
                rows.append(example(q, NO_DOCS, answer_of(s)))
                rows.append(example(q, join(distractors(s["id"], random.choice([1, 2]))), answer_of(s)))
    random.shuffle(rows)
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ids = sorted(s["id"] for s in SCENARIOS)
    eval_ids = {sid for i, sid in enumerate(ids) if i % 4 == 3}      # every 4th scenario is held out
    train = build([s for s in SCENARIOS if s["id"] not in eval_ids])
    ev = build([s for s in SCENARIOS if s["id"] in eval_ids])
    Path("data").mkdir(exist_ok=True)
    write_jsonl("data/train.jsonl", train)
    write_jsonl("data/eval.jsonl", ev)

    faq = ["# Acme Cloud Help Center (fictional demo document)\n"]
    for s in SCENARIOS:
        if s["doc"]:
            faq.append(f"## {s['questions'][0]}\n{s['doc']}\n")
    Path("data/demo_docs").mkdir(parents=True, exist_ok=True)
    Path("data/demo_docs/acme_faq.md").write_text("\n".join(faq), encoding="utf-8")

    print(f"scenarios: {len(SCENARIOS)}  held-out for eval: {sorted(eval_ids)}")
    print(f"train examples: {len(train)}   eval examples: {len(ev)}")


if __name__ == "__main__":
    main()
