"""plot.py — GQA ablation: three val-loss curves + quality-efficiency tradeoff
→ gqa_ablation.png
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_head = data.get("n_head", 4)
    arms_data = data["arms"]

    # ---- style ----
    style = {
        "mha":   ("#1f77b4", f"MHA (n_kv={n_head}, ratio 1:1)"),
        "gqa-2": ("#ff7f0e", f"GQA-2 (n_kv={n_head // 2}, ratio 2:1)"),
        "mqa":   ("#d62728", f"MQA (n_kv=1, ratio {n_head}:1)"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.2, 5.8))

    # ---- left: training curves ----
    for arm in arms_data:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax1.plot([p["step"] for p in tr], [p["val"] for p in tr], "-o",
                 color=color, lw=1.7, ms=3.5, label=label)

    ax1.set_xscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("validation cross-entropy")
    ax1.set_title(f"GQA ratio ablation — training curves  (d{data['depth']}, "
                  f"{data['max_tokens'] / 1e6:.0f}M tokens)")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(fontsize=9)

    # ---- right: quality-efficiency tradeoff ----
    # KV cache size ∝ n_kv_head
    final_ce = []
    kv_fraction = []
    labels = []
    colors = []
    for arm in arms_data:
        tr = arm["trajectory"]
        n_kv = arm.get("n_kv_head", n_head)
        final_ce.append(tr[-1]["val"])
        kv_fraction.append(n_kv / n_head * 100)
        col, lbl = style.get(arm["arm"], ("gray", arm["arm"]))
        labels.append(lbl)
        colors.append(col)

    # Find baseline (MHA) for delta
    mha_ce = next(ce for ce, nkv in zip(final_ce,
                 [a.get("n_kv_head", n_head) for a in arms_data]) if nkv == n_head)

    for i, (ce, kvf, lbl, col) in enumerate(zip(final_ce, kv_fraction, labels, colors)):
        delta = ce - mha_ce
        ax2.scatter(kvf, ce, c=col, s=140, zorder=5, edgecolors="white", linewidths=1.2)
        ax2.annotate(f"{lbl}\nΔ={delta:+.3f} CE\nKV={kvf:.0f}%",
                     (kvf, ce), textcoords="offset points",
                     xytext=(12 if i < 2 else -70, -12), fontsize=8.5,
                     color=col, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=col, lw=0.8))

    ax2.set_xlabel("KV cache size (% of MHA)")
    ax2.set_ylabel("final validation cross-entropy")
    ax2.set_title("GQA quality-efficiency tradeoff\n"
                  "(lower-left is better — less memory AND lower loss)")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.invert_xaxis()  # more efficient (left) is better

    fig.tight_layout()
    out = HERE / "gqa_ablation.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
