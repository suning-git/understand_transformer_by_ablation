"""run.py — train all three GQA arms through the blessed text orchestrator
and collect val-loss trajectories.

    python gqa_ablation/run.py      # trains all three arms
    python gqa_ablation/plot.py     # -> gqa_ablation.png
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

import spec

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent if (HERE.parent.parent / "core").is_dir() else HERE
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

# Portable launch context: this experiment's dir on PYTHONPATH plus the
# nanoinfra checkout if present.
_NANO = None
p = HERE
while p != p.parent:
    if (p / "core").is_dir() and (p / "modalities").is_dir():
        _NANO = p
        break
    p = p.parent
_PP = os.pathsep.join([str(HERE)] + ([str(_NANO)] if _NANO else [])
                      + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
BASE_ENV = {**os.environ, "PYTHONPATH": _PP}
BASE_ENV.setdefault("NANOINFRA_BASE_DIR", str(_NANO / "outputs") if _NANO else "./outputs")
BASE_CWD = str(_NANO) if _NANO else str(HERE)

EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=20):
    """~n log-spaced integer steps in [first, max_steps] (deduped, sorted)."""
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, arm_overrides, max_steps, steps):
    ov = spec.train_overrides(arm_overrides, max_steps, steps)
    print(f"[run ] {label}: n_kv_head={arm_overrides.get('model.n_kv_head','?')} "
          f"d{spec.DEPTH} -> {max_steps} steps ...", flush=True)
    out = subprocess.run([sys.executable, "-u", "-m", spec.ORCHESTRATOR, *ov],
                         cwd=BASE_CWD, env=BASE_ENV,
                         capture_output=True, text=True)
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(
            f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, "
          f"val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}", flush=True)
    return {"arm": label, "n_kv_head": int(arm_overrides["model.n_kv_head"]),
            "trajectory": traj}


def main():
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, ov, max_steps, steps) for label, ov in spec.ARMS]
    out = {
        "depth": spec.DEPTH,
        "n_head": 4,
        "max_steps": max_steps,
        "max_tokens": spec.MAX_TOKENS,
        "lr_max": spec.LR_MAX,
        "arms": arms,
    }
    (RESULTS / "curves.json").write_text(json.dumps(out, indent=2))
    print(f"WROTE {RESULTS / 'curves.json'} ({len(arms)} arms)")


if __name__ == "__main__":
    main()
