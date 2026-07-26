"""spec.py — DeepSeek-inspired MoE + SWA optimizations on the modern GPT trunk.

Two positive ablations, measuring the benefit of each DeepSeek V4 innovation:
  modern      — baseline (RoPE · RMSNorm · no bias · ReLU^2 · QK-norm)
  modern_moe  — replace dense MLP with 16-expert DeepSeekMoE (top-2 routing, shared expert)
  modern_swa  — sliding window attention (window=96) + per-head sink token

Same data, budget, and recipe — only the trunk differs.
Each arm: d=6, 300M tokens, 2x RTX 5090 DDP.
"""
DEPTH = 6
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 512, 16, 16384
MAX_TOKENS = 300_000_000
WARMUP_STEPS = 1000
WARMDOWN_RATIO = 0.1
FINAL_LR_FRAC = 0.05
N_EVALS = 30
EVAL_TOKENS = 131072

ORCHESTRATOR = "modalities.text.train_text"

ARMS = [
    ("modern",     None),                                    # baseline
    ("modern_moe", "modern_moe.ModernMoETrunk"),             # + MoE (16 experts)
    ("modern_swa", "modern_swa.ModernSWATrunk"),             # + Sliding Window + Sink
]


def train_overrides(trunk_class, max_steps, eval_at):
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": WARMDOWN_RATIO,
        "optimizer.scheduler.final_lr_frac": FINAL_LR_FRAC,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 100,
    }
    if trunk_class:
        ov["model.trunk_class"] = trunk_class
    return [f"{k}={v}" for k, v in ov.items()]
