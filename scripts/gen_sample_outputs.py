"""Generate service outputs for every public sample input (required deliverable).

Run: python -m scripts.gen_sample_outputs
Writes sample_outputs/sample_outputs.json with {input, our_output, expected_output}.
"""
from __future__ import annotations

import json
import pathlib

from app import pipeline
from app.schemas import TicketRequest

ROOT = pathlib.Path(__file__).resolve().parents[1]
cases = json.loads((ROOT / "sample_cases.json").read_text(encoding="utf-8"))["cases"]

out = []
for c in cases:
    result = pipeline.analyze(TicketRequest(**c["input"]))
    out.append({"id": c["id"], "input": c["input"], "our_output": result,
                "expected_output": c["expected_output"]})

dst = ROOT / "sample_outputs"
dst.mkdir(exist_ok=True)
(dst / "sample_outputs.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {len(out)} outputs to {dst / 'sample_outputs.json'}")
