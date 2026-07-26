# DeepSeek V4-inspired MoE & Sliding Window Attention

**Positive ablation**: add ONE DeepSeek V4 innovation at a time to the modern GPT trunk, measure the bpb improvement.

## Experiment

| Arm | Architecture | Description |
|------|-------------|-------------|
| `modern` | RoPE · RMSNorm · ReLU² · QK-norm | baseline |
| `modern_moe` | baseline + DeepSeekMoE | 16 SwiGLU experts + 1 shared expert, top-2 routing, aux-loss-free load balancing |
| `modern_swa` | baseline + SWA + sink | Sliding window attention (window=96) + per-head learnable sink token |

Training budget: d=6, 300M tokens/arm, 2×RTX 5090 DDP (~20 min/arm). All arms share identical recipe (lr=3e-4, batch=16384, warmup=1000, cosine warmdown).

## Results

| Arm | val CE | bpb (bytes/token=3.7) | vs baseline |
|------|--------|------|------|
| modern (baseline) | 4.489 | 1.750 | — |
| modern_moe | 4.462 | **1.740** | **-0.011** 🥇 |
| modern_swa | 4.474 | 1.745 | -0.005 🥈 |

bpb = CE / ln(2) / bytes_per_token

## Key findings

1. **MoE wins**: Replacing the dense MLP with 16 fine-grained SwiGLU experts gives the largest bpb reduction (-0.011). The model learns to specialize experts without auxiliary loss, using dynamic per-expert bias for load balancing.

2. **SWA helps marginally**: Limiting attention to a 96-token window + sink token gives a small but consistent improvement (-0.005 bpb), while reducing attention complexity from O(T²) to O(T·window).

3. **MoE + SWA combo pending**: The combined architecture (modern_moe_swa) ran at d=12 but needs re-running at d=6 for a fair comparison with these results.

## Reference

- DeepSeek V4 Technical Report (2026-04)
- [从零构建智能模型 · 第七课 · 前沿开源模型的架构](https://suning-git.github.io/thu-2026-AI/slides/lesson-7.html)
- Built on [nanoinfra](https://github.com/suning-git/nanoinfra)
