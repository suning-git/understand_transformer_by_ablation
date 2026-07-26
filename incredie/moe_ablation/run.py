"""run.py — train each arm through nanoinfra's orchestrator.

    python run.py            # trains every arm in spec.ARMS
    python plot.py           # -> the figure with bpb
"""
import json, os, re, subprocess, sys, time
from pathlib import Path
import numpy as np
import spec

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
EVAL_RE = re.compile(r"Step\s+(\d+)\s+\|\s+val/text_ce:\s+([\d.]+)")


def eval_schedule(max_steps, n=spec.N_EVALS, first=5):
    s = np.unique(np.round(np.logspace(np.log10(first), np.log10(max_steps), n)))
    return [int(x) for x in s]


def run_arm(label, trunk_class, max_steps, steps, n_gpus=2):
    ov = spec.train_overrides(trunk_class, max_steps, steps)
    torchrun = os.path.join(os.path.dirname(sys.executable), "torchrun")
    cmd = [torchrun, f"--nproc_per_node={n_gpus}", "-m", spec.ORCHESTRATOR] + ov
    print(f"[run ] {label}: d{spec.DEPTH}, {max_steps} steps, {n_gpus} GPU ...", flush=True)
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    text = out.stdout + "\n" + out.stderr
    traj = [{"step": int(s), "val": float(v)} for s, v in EVAL_RE.findall(text)]
    if out.returncode != 0 or len(traj) < 3:
        raise SystemExit(f"arm {label} FAILED (rc={out.returncode}, {len(traj)} evals, {elapsed/60:.1f}min):\n{text[-3000:]}")
    print(f"[done] {label}: {len(traj)} evals, val {traj[0]['val']:.3f} -> {traj[-1]['val']:.3f}  ({elapsed/60:.1f} min)", flush=True)
    return {"arm": label, "trajectory": traj, "elapsed_min": round(elapsed/60, 1)}


def main():
    max_steps = int(spec.MAX_TOKENS // spec.TBS)
    steps = eval_schedule(max_steps)
    arms = [run_arm(label, tc, max_steps, steps) for label, tc in spec.ARMS]
    (RESULTS / "curves.json").write_text(
        json.dumps({"depth": spec.DEPTH, "max_tokens": spec.MAX_TOKENS, "arms": arms}, indent=2))
    print(f"WROTE {RESULTS / 'curves.json'}")


if __name__ == "__main__":
    main()
