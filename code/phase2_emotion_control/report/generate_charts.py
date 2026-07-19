"""
generate_charts.py — Generate all charts for the project report
===============================================================
Output:
  - report/figures/pearson_comparison.png
  - report/figures/fad_comparison.png
  - report/figures/clap_comparison.png
  - report/figures/va_error.png
  - report/figures/va_target_pred_v3.png
  - report/figures/dsp_overview.png
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

REPORT_DIR = Path(__file__).parent
FIGS_DIR = REPORT_DIR / "figures"
FIGS_DIR.mkdir(exist_ok=True)

# ===================================================================
# Aggregate evaluation data for all versions
# ===================================================================
VERSIONS = {
    "v1\n(add-768d)": {
        "FAD": None,  # v1 no FAD eval
        "CLAP": 0.0650,
        "V_corr": 0.273,
        "A_corr": -0.400,
        "VAE": 0.243,
        "AAE": 0.278,
    },
    "v2\n(prefix)": {
        "FAD": 1.601,
        "CLAP": -0.0396,
        "V_corr": -0.108,
        "A_corr": 0.576,
        "VAE": 0.256,
        "AAE": 0.244,
    },
    "v2_fixed\n(prefix+)": {
        "FAD": 1.621,
        "CLAP": -0.0395,
        "V_corr": 0.653,
        "A_corr": 0.073,
        "VAE": 0.248,
        "AAE": 0.246,
    },
    "v3\n(balanced)": {
        "FAD": 1.715,
        "CLAP": -0.0536,
        "V_corr": 0.446,
        "A_corr": 0.554,
        "VAE": 0.246,
        "AAE": 0.227,
    },
    "v5\n(identity)": {
        "FAD": 1.713,
        "CLAP": -0.0518,
        "V_corr": -0.174,
        "A_corr": 0.494,
        "VAE": 0.251,
        "AAE": 0.224,
    },
    "v6\n(scale)": {
        "FAD": 1.644,
        "CLAP": 0.0017,
        "V_corr": 0.376,
        "A_corr": 0.427,
        "VAE": 0.250,
        "AAE": 0.236,
    },
    "v7\n(decoder)": {
        "FAD": 1.595,
        "CLAP": 0.0069,
        "V_corr": 0.105,
        "A_corr": 0.565,
        "VAE": 0.247,
        "AAE": 0.238,
    },
    "v7_10tar\n(decoder+)": {
        "FAD": 1.520,
        "CLAP": 0.0087,
        "V_corr": 0.373,
        "A_corr": 0.154,
        "VAE": 0.232,
        "AAE": 0.241,
    },
}

# ===================================================================
# Fig 1: Pearson correlation comparison
# ===================================================================
def plot_pearson_comparison():
    labels = list(VERSIONS.keys())
    v_corrs = [VERSIONS[v]["V_corr"] for v in labels]
    a_corrs = [VERSIONS[v]["A_corr"] for v in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, v_corrs, width, label="Valence Pearson", color="#4C72B0")
    bars2 = ax.bar(x + width/2, a_corrs, width, label="Arousal Pearson", color="#DD8452")

    # Annotate values on bars
    for bar in bars1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                    ha="center", va=va, fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3 if h >= 0 else -3), textcoords="offset points",
                    ha="center", va=va, fontsize=7)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Pearson Correlation")
    ax.set_title("Emotion Fidelity Across Versions (Pearson r)")
    ax.legend(fontsize=9)
    ax.set_ylim(-0.6, 0.9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "pearson_comparison.png", dpi=150)
    plt.close(fig)
    print("  [OK] pearson_comparison.png")


# ===================================================================
# Fig 2: FAD comparison
# ===================================================================
def plot_fad_comparison():
    labels = [k for k in VERSIONS if VERSIONS[k]["FAD"] is not None]
    fads = [VERSIONS[k]["FAD"] for k in labels]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#4C72B0" if f <= 1.6 else "#DD8452" for f in fads]
    bars = ax.bar(range(len(labels)), fads, color=colors, width=0.5)
    for bar, f in zip(bars, fads):
        ax.annotate(f"{f:.3f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    ax.axhline(y=1.17, color="red", linestyle="--", linewidth=1, label="MusicGen baseline (FAD~1.17)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("FAD (lower is better)")
    ax.set_title("Audio Quality Across Versions (Fréchet Audio Distance)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "fad_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✅ fad_comparison.png")


# ===================================================================
# Fig 3: CLAP Score comparison
# ===================================================================
def plot_clap_comparison():
    labels = list(VERSIONS.keys())
    scores = [VERSIONS[v]["CLAP"] for v in labels]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2CA02C" if s > -0.02 else "#DD8452" for s in scores]
    bars = ax.bar(range(len(labels)), scores, color=colors, width=0.5)
    for bar, s in zip(bars, scores):
        ax.annotate(f"{s:.4f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3 if s >= -0.05 else -12), textcoords="offset points",
                    ha="center", fontsize=8)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("CLAP Score")
    ax.set_title("Text-Audio Alignment (CLAP Score)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "clap_comparison.png", dpi=150)
    plt.close(fig)
    print("  ✅ clap_comparison.png")


# ===================================================================
# Fig 4: VAE / AAE comparison
# ===================================================================
def plot_va_error():
    labels = list(VERSIONS.keys())
    vaes = [VERSIONS[v]["VAE"] for v in labels]
    aaes = [VERSIONS[v]["AAE"] for v in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 4))
    bars1 = ax.bar(x - width/2, vaes, width, label="VAE (Valence error)", color="#4C72B0")
    bars2 = ax.bar(x + width/2, aaes, width, label="AAE (Arousal error)", color="#DD8452")

    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Absolute Error")
    ax.set_title("VAE / AAE Across Versions (lower is better)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "va_error.png", dpi=150)
    plt.close(fig)
    print("  ✅ va_error.png")


# ===================================================================
# Fig 5: v3 target vs prediction scatter plot (best balanced)
# ===================================================================
def plot_v3_scatter():
    """Run CLAP evaluation on v3 samples to get per-sample predictions."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from evaluate import EmotionFidelityEvaluator

    metadata_path = Path(__file__).parent.parent / "eval_results_v3/metadata.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    evaluator = EmotionFidelityEvaluator(device="cuda")

    v_targets, v_preds = [], []
    a_targets, a_preds = [], []

    for item in metadata:
        gen_path = Path(__file__).parent.parent / item["gen_path"]
        if not gen_path.exists():
            print(f"  [WARN] File not found: {gen_path}")
            continue
        pred_v, pred_a = evaluator.predict_va(str(gen_path))
        v_targets.append(item["valence"])
        a_targets.append(item["arousal"])
        v_preds.append(pred_v)
        a_preds.append(pred_a)
        print(f"  {item['desc']:20s} target(V={item['valence']:.2f}, A={item['arousal']:.2f}) "
              f"→ pred(V={pred_v:.2f}, A={pred_a:.2f})")

    v_targets = np.array(v_targets)
    v_preds = np.array(v_preds)
    a_targets = np.array(a_targets)
    a_preds = np.array(a_preds)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Valence
    ax1.scatter(v_targets, v_preds, c="#4C72B0", s=60, alpha=0.7)
    lims = [0.1, 0.9]
    ax1.plot(lims, lims, "r--", linewidth=1, alpha=0.5, label="perfect")
    z = np.polyfit(v_targets, v_preds, 1)
    p = np.poly1d(z)
    ax1.plot(lims, p(lims), "b-", linewidth=1, alpha=0.5, label=f"fit (r={np.corrcoef(v_targets, v_preds)[0,1]:.2f})")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("Target Valence")
    ax1.set_ylabel("Predicted Valence")
    ax1.set_title("Valence (v3)")
    ax1.legend(fontsize=8)
    ax1.set_aspect("equal")

    # Arousal
    ax2.scatter(a_targets, a_preds, c="#DD8452", s=60, alpha=0.7)
    ax2.plot(lims, lims, "r--", linewidth=1, alpha=0.5, label="perfect")
    z = np.polyfit(a_targets, a_preds, 1)
    p = np.poly1d(z)
    ax2.plot(lims, p(lims), "b-", linewidth=1, alpha=0.5, label=f"fit (r={np.corrcoef(a_targets, a_preds)[0,1]:.2f})")
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_xlabel("Target Arousal")
    ax2.set_ylabel("Predicted Arousal")
    ax2.set_title("Arousal (v3)")
    ax2.legend(fontsize=8)
    ax2.set_aspect("equal")

    fig.suptitle("v3 Model: Target Emotion vs CLAP-Predicted Emotion", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "va_target_pred_v3.png", dpi=150)
    plt.close(fig)
    print("  ✅ va_target_pred_v3.png")


# ===================================================================
# Fig 6: Two-stage DSP pipeline overview
# ===================================================================
def plot_dsp_overview():
    """Draw DSP pipeline diagram."""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis("off")

    # Define boxes
    boxes = [
        (0.00, 0.0, 0.18, 0.6, "V-A Coords\n(valence, arousal)", "#E8F5E9"),
        (0.22, 0.0, 0.18, 0.6, "Emotion Prompt\n→ MusicGen\n(clean gen)", "#BBDEFB"),
        (0.44, 0.0, 0.18, 0.6, "DSP Params\npitch/tempo/EQ/vol", "#FFF3E0"),
        (0.66, 0.0, 0.18, 0.6, "DSP Process\n→ emotional audio", "#F3E5F5"),
        (0.88, 0.0, 0.18, 0.6, "Output BGM\n(lossless)", "#E8F5E9"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="gray",
                             linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=9)

    # Arrows
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        ax.annotate("", xy=(x2, 0.3), xytext=(x1, 0.3),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.5))

    ax.set_title("Two-Stage Generation Pipeline (DSP)", fontsize=12, pad=10)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "dsp_overview.png", dpi=150)
    plt.close(fig)
    print("  ✅ dsp_overview.png")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Generating report charts...")
    print("=" * 50)

    plot_pearson_comparison()
    plot_fad_comparison()
    plot_clap_comparison()
    plot_va_error()
    plot_dsp_overview()

    print("\nGenerating v3 scatter plot (requires CLAP inference)...")
    plot_v3_scatter()

    print(f"\nAll charts saved to: {FIGS_DIR}/")
    print("Files:")
    for f in sorted(FIGS_DIR.glob("*.png")):
        print(f"  - {f.name}")
