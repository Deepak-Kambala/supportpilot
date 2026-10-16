import json
import statistics

from .assistant import parse_output

def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def macro_f1(gold, pred, labels):
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, pred))
        fp = sum(g != label and p == label for g, p in zip(gold, pred))
        fn = sum(g == label and p != label for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


def summarize(records):
    n = len(records)
    gold_cat = [r["gold"]["category"] for r in records]
    pred_cat = [r["pred"]["category"] for r in records]
    esc_gold = [r["gold"]["needs_human"] for r in records]
    esc_pred = [r["pred"]["needs_human"] for r in records]
    tp = sum(g and p for g, p in zip(esc_gold, esc_pred))
    fp = sum((not g) and p for g, p in zip(esc_gold, esc_pred))
    fn = sum(g and (not p) for g, p in zip(esc_gold, esc_pred))
    return {
        "n": n,
        "valid_json_rate": sum(r["valid"] for r in records) / n,
        "category_accuracy": sum(g == p for g, p in zip(gold_cat, pred_cat)) / n,
        "category_macro_f1": macro_f1(gold_cat, pred_cat, sorted(set(gold_cat) | set(pred_cat))),
        "priority_accuracy": sum(r["gold"]["priority"] == r["pred"]["priority"] for r in records) / n,
        "escalation_recall": tp / (tp + fn) if tp + fn else None,
        "escalation_precision": tp / (tp + fp) if tp + fp else None,
        "median_latency_s": statistics.median(r["latency"] for r in records),
    }


def evaluate(rows, generate_fn, progress=True):
    """rows: chat-format examples (system, user, assistant). generate_fn(messages) -> (raw_text, seconds)."""
    records = []
    for i, row in enumerate(rows, 1):
        msgs = row["messages"]
        gold = json.loads(msgs[2]["content"])
        raw, seconds = generate_fn(msgs[:2])
        pred, valid = parse_output(raw)
        records.append({"question": msgs[1]["content"], "gold": gold, "pred": pred,
                        "valid": valid, "latency": seconds, "raw": raw})
        if progress and i % 10 == 0:
            print(f"  {i}/{len(rows)}")
    return records, summarize(records)
