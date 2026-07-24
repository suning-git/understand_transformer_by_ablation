"""
spec.py — GQA (Grouped Query Attention) ablation recipe.

Vary the number of key/value heads (n_kv_head) while keeping the query-head count
fixed. All three arms use the SAME reference GPT architecture; the ONLY
difference is `model.n_kv_head`.

  * MHA  (baseline)  — n_kv_head = 4  (no GQA, ratio 1:1)
  * GQA-2            — n_kv_head = 2  (ratio 2:1)
  * MQA              — n_kv_head = 1  (ratio 4:1)

Model: d8 (depth=8, dim=512, n_head=4) — small enough to train quickly while
having enough heads (4) for three clean GQA ratios. n_head must be divisible by
n_kv_head; 4 supports {1, 2, 4} natively.

Hypothesis: reducing KV heads incurs a small but measurable increase in val CE,
trading model quality for inference-time KV-cache efficiency. The GQA-2 ratio
(2:1, used by Llama 3 70B) should sit close to the MHA baseline, while MQA
pays a visible but often acceptable penalty — forming a quality-efficiency Pareto
frontier.
"""
DEPTH = 8
LR_MAX = "3e-4"
SEED = 42

SEQ_LEN, DBS, TBS = 1024, 32, 32768
MAX_TOKENS = 500_000_000
WARMUP_STEPS = 200
N_EVALS = 40
EVAL_TOKENS = 524288

ORCHESTRATOR = "modalities.text.train_text"

# (arm label, extra config overrides — n_kv_head ONLY)
ARMS = [
    ("mha",    {"model.n_kv_head": "4"}),   # baseline: all 4 KV heads
    ("gqa-2",  {"model.n_kv_head": "2"}),   # GQA 2:1
    ("mqa",    {"model.n_kv_head": "1"}),   # MQA (single KV head)
]


def train_overrides(arm_overrides, max_steps, eval_at):
    """Hydra CLI overrides pinning this project's recipe, plus the per-arm
    n_kv_head. Everything else is held constant."""
    ov = {
        "model.depth": DEPTH,
        "optimizer.lr_max": LR_MAX,
        "seed": SEED,
        "sequence_len": SEQ_LEN,
        "device_batch_size": DBS,
        "total_batch_size": TBS,
        "max_steps": max_steps,
        "use_compile": "false",
        "optimizer.scheduler.warmup_steps": WARMUP_STEPS,
        "optimizer.scheduler.warmdown_ratio": 0.0,
        "optimizer.scheduler.final_lr_frac": 1.0,
        "checkpoint.enabled": "false",
        "evaluation.text.eval_at": "[" + ",".join(map(str, eval_at)) + "]",
        "evaluation.text.eval_tokens": EVAL_TOKENS,
        "logging.log_every": 200,
    }
    ov.update(arm_overrides)
    return [f"{k}={v}" for k, v in ov.items()]
