"""plot.py — val-loss curves with bpb conversion for Hackathon submission.

Reads results/curves.json and writes frontier_arch_final.png.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ce_to_bpb(ce, bytes_per_token=3.7):
    """Convert cross-entropy (nats/token) to bits-per-byte.
    bpb = CE_nats / ln(2) / bytes_per_token
    3.7 is typical for BPE tokenizers on English text.
    """
    import math
    return ce / math.log(2) / bytes_per_token


def main():
    data = json.loads((HERE / "results" / "curves.json").read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import math

    style = {
        "modern":         ("#1f77b4", "modern baseline"),
        "modern_kv":      ("#2ca02c", "modern + K=V"),
        "modern_swiglu":  ("#ff7f0e", "modern + SwiGLU"),
        "modern_moe":     ("#d62728", "modern + MoE (16 experts)"),
        "modern_swa":     ("#9467bd", "modern + SWA (window=96 + sink)"),
        "modern_moe_swa": ("#8c564b", "modern + MoE + SWA"),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # Left: CE (nats)
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax1.plot([p["step"] for p in tr], [p["val"] for p in tr], "-o",
                 color=color, lw=1.9, ms=3, label=label)

    ax1.set_xscale("log")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("val CE (nats/token)")
    ax1.set_title(f"Architecture comparison (d={data.get('depth','?')}, {data.get('max_tokens',0)/1e6:.0f}M tokens)")
    ax1.grid(True, which="both", ls=":", alpha=0.4)
    ax1.legend(fontsize=8)

    # Right: bpb (for Hackathon submission)
    bytes_per_token = 3.7  # BPE average for FineWeb
    for arm in data["arms"]:
        tr = arm["trajectory"]
        color, label = style.get(arm["arm"], ("gray", arm["arm"]))
        ax2.plot([p["step"] for p in tr], [ce_to_bpb(p["val"], bytes_per_token) for p in tr],
                 "-o", color=color, lw=1.9, ms=3, label=label)

    ax2.set_xscale("log")
    ax2.set_xlabel("training step")
    ax2.set_ylabel("bpb (bits per byte)")
    ax2.set_title(f"bpb = CE / ln(2) / {bytes_per_token} bytes/token")
    ax2.grid(True, which="both", ls=":", alpha=0.4)
    ax2.legend(fontsize=8)

    fig.suptitle("DeepSeek V4 optimizations — Hackathon submission",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    out = HERE / "frontier_arch_final.png"
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    # Print bpb summary
    print("\n=== bpb summary ===")
    for arm in data["arms"]:
        tr = arm["trajectory"]
        final_ce = tr[-1]["val"]
        final_bpb = ce_to_bpb(final_ce, bytes_per_token)
        print(f"  {arm['arm']:20s}  CE={final_ce:.4f}  bpb={final_bpb:.4f}")


if __name__ == "__main__":
    main()
