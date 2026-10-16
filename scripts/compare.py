"""python -m scripts.compare  -> side-by-side table of every results/*.json"""
import json
from pathlib import Path

files = sorted(Path("results").glob("*.json"))
data = {f.stem: json.loads(f.read_text()) for f in files}
if not data:
    raise SystemExit("No results yet. Run scripts.run_eval first.")
keys = list(next(iter(data.values())).keys())
print(f"{'metric':24}" + "".join(f"{name:>14}" for name in data))
for k in keys:
    row = f"{k:24}"
    for name in data:
        v = data[name][k]
        row += f"{'n/a':>14}" if v is None else f"{v:>14.3f}" if isinstance(v, float) else f"{v:>14}"
    print(row)
