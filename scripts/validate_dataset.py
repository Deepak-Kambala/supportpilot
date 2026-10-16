"""python -m scripts.validate_dataset data/train.jsonl data/eval.jsonl
Checks every line is valid chat JSON with a valid answer, and that no user message appears in both files."""
import json
import sys

from src.assistant import parse_output


def load(path):
    rows = []
    for n, line in enumerate(open(path, encoding="utf-8"), 1):
        row = json.loads(line)
        roles = [m["role"] for m in row["messages"]]
        assert roles == ["system", "user", "assistant"], f"{path}:{n} bad roles {roles}"
        _, ok = parse_output(row["messages"][2]["content"])
        assert ok, f"{path}:{n} assistant answer is not valid"
        rows.append(row)
    return rows


train, ev = load(sys.argv[1]), load(sys.argv[2])
q = lambda rows: {r["messages"][1]["content"].split("\n")[0] for r in rows}
overlap = q(train) & q(ev)
print(f"train={len(train)} eval={len(ev)} overlapping customer messages={len(overlap)}")
assert not overlap, f"LEAK: {sorted(overlap)[:3]}"
print("OK")
