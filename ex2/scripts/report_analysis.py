"""
Analyze evaluate.py outputs and produce report figures.

Usage (after evaluate.py on trained + identity baselines):
    python scripts/report_analysis.py \
        --results eval_outputs/results.json \
        --baseline eval_outputs_baseline/results.json \
        --gen_dir eval_outputs/generated_wavs \
        --test_manifest data/manifests/train_val_test.jsonl/test.jsonl \
        --listening eval_outputs/listening_test_sample.json \
        --controllability eval_outputs/va_controllability.json \
        --va_regressor checkpoints/va_regressor.pt \
        --out_dir report_figures

Requires: matplotlib (pip install matplotlib)
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Allow `python scripts/report_analysis.py` from repo root.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import yaml

from utils.va_regressor import ClapEmbedder, load_regressor

# Report-friendly figure labels (avoid vague "Ours" in captions/legends)
LABEL_TRAINED = "Trained adapter (v3)"
LABEL_IDENTITY = "Identity adapter (untrained)"


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_manifest(path, max_examples=None):
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if max_examples is not None:
        records = records[:max_examples]
    return records


def predict_va_for_wavs(wav_paths, va_regressor_path, clap_checkpoint, device):
    embedder = ClapEmbedder(checkpoint=clap_checkpoint, device=device)
    regressor = load_regressor(va_regressor_path, device=device)
    embeds = embedder.embed_audio_files(wav_paths)
    with torch.no_grad():
        pred = regressor(torch.from_numpy(embeds.astype(np.float32)).to(device))
    return pred.cpu().numpy()


def infer_prompt_category(prompt: str) -> str:
    p = prompt.lower()
    if "melancholic" in p or "sad" in p:
        return "sad"
    if "tense" in p or "aggressive" in p or "intense" in p:
        return "tense"
    if "calm" in p or "peaceful" in p or "relaxed" in p:
        return "calm"
    if "upbeat" in p or "joyful" in p or "energetic" in p or "bright" in p:
        return "happy"
    return "other"


def plot_metrics_comparison(ours, baseline, out_path):
    labels = ["CLAP↑", "V-MAE↓", "A-MAE↓", "V-CCC↑", "A-CCC↑"]
    ours_vals = [
        ours.get("mean_clap_score", 0),
        ours.get("valence_mae", 0),
        ours.get("arousal_mae", 0),
        ours.get("valence_ccc", 0),
        ours.get("arousal_ccc", 0),
    ]
    if ours.get("fad") is not None:
        labels.append("FAD↓")
        ours_vals.append(ours["fad"])
    base_vals = None
    if baseline:
        base_vals = [
            baseline.get("mean_clap_score", 0),
            baseline.get("valence_mae", 0),
            baseline.get("arousal_mae", 0),
            baseline.get("valence_ccc", 0),
            baseline.get("arousal_ccc", 0),
        ]
        if baseline.get("fad") is not None:
            base_vals.append(baseline["fad"])

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    if base_vals is not None:
        ax.bar(x - width / 2, ours_vals, width, label=LABEL_TRAINED, color="#4C78A8")
        ax.bar(x + width / 2, base_vals, width, label=LABEL_IDENTITY, color="#B279A2")
    else:
        ax.bar(x, ours_vals, width, label=LABEL_TRAINED, color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Objective metrics: trained adapter vs identity baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_va_scatter(target_va, pred_va, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, dim, name in zip(axes, [0, 1], ["Valence", "Arousal"]):
        t = target_va[:, dim]
        p = pred_va[:, dim]
        ax.scatter(t, p, alpha=0.65, edgecolors="none")
        lims = [min(t.min(), p.min()) - 0.1, max(t.max(), p.max()) + 0.1]
        ax.plot(lims, lims, "k--", linewidth=1, label="ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Predicted {name}")
        ccc = concordance_correlation_coefficient(t, p)
        mae = np.mean(np.abs(t - p))
        ax.set_title(f"{name}: CCC={ccc:.3f}, MAE={mae:.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle("Emotion fidelity on generated audio (test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_circumplex(target_va, pred_va, prompts, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    cats = [infer_prompt_category(p) for p in prompts]
    palette = {
        "happy": "#F58518",
        "sad": "#4C78A8",
        "tense": "#E45756",
        "calm": "#72B7B2",
        "other": "#999999",
    }
    for ax, data, title in [
        (axes[0], target_va, "Target V-A (annotations)"),
        (axes[1], pred_va, "Predicted V-A (CLAP regressor)"),
    ]:
        for cat in sorted(set(cats)):
            idx = [i for i, c in enumerate(cats) if c == cat]
            ax.scatter(
                data[idx, 0], data[idx, 1],
                s=28, alpha=0.75, label=cat, c=palette.get(cat, "#999999"),
            )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.25)
    fig.suptitle("Russell circumplex view (colored by prompt category)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_training_summary_table(out_path):
    rows = [
        ["Version", "Config", "Val loss", "Subjective quality", "Emotion control"],
        ["v1 FiLM", "use_film=true, lr=1e-4, 100ep", "7.22", "Poor vs baseline", "Weak"],
        ["v2 prefix", "prefix-only, lr=3e-5, 15ep", "6.54", "~2s then noise", "N/A"],
        ["v3 final", "prefix_scale+reg, lr=1e-5, 8ep", "6.26", "Acceptable, mild noise", "Clear happy/sad"],
        ["Baseline", "identity adapter / stock MusicGen", "—", "Good", "No V-A effect"],
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    ax.set_title("Training iterations summary", pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_per_sample_csv(target_va, pred_va, prompts, out_path):
    lines = ["idx,prompt,category,target_valence,target_arousal,pred_valence,pred_arousal"]
    for i, (prompt, (tv, ta), (pv, pa)) in enumerate(zip(prompts, target_va, pred_va)):
        cat = infer_prompt_category(prompt)
        prompt_esc = prompt.replace('"', "'")
        lines.append(
            f'{i},{prompt_esc},{cat},{tv:.4f},{ta:.4f},{pv:.4f},{pa:.4f}'
        )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _rows_by_tag(rows, tag_order=("low", "mid", "high")):
    """Aggregate rows per prompt: {prompt_idx: {tag: row}}."""
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_idx"], {})[r["va_tag"]] = r
    return by_prompt, tag_order


def plot_controllability_bars(ctrl, out_path):
    """Pearson + spread: ours vs identity (fixed prompt, V-A only)."""
    ours, base = ctrl["ours"], ctrl["baseline"]
    labels = ["Pearson V", "Pearson A", "V spread", "A spread"]
    ours_vals = [
        ours.get("pearson_target_vs_pred_valence", 0),
        ours.get("pearson_target_vs_pred_arousal", 0),
        ours.get("mean_valence_spread_low_vs_high", 0),
        ours.get("mean_arousal_spread_low_vs_high", 0),
    ]
    base_vals = [
        base.get("pearson_target_vs_pred_valence", 0),
        base.get("pearson_target_vs_pred_arousal", 0),
        base.get("mean_valence_spread_low_vs_high", 0),
        base.get("mean_arousal_spread_low_vs_high", 0),
    ]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, ours_vals, width, label=LABEL_TRAINED, color="#4C78A8")
    ax.bar(x + width / 2, base_vals, width, label=LABEL_IDENTITY, color="#B279A2")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    prompt = ctrl.get("fixed_prompt") or "(test captions)"
    ax.set_title(f"V-A controllability (fixed prompt)\n{prompt[:60]}{'…' if len(prompt) > 60 else ''}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_controllability_target_pred(ctrl, out_path):
    """Target vs predicted V and A along low→mid→high (lines per model)."""
    tag_order = ["low", "mid", "high"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    styles = [
        ("ours_rows", LABEL_TRAINED, "#4C78A8", "o"),
        ("baseline_rows", LABEL_IDENTITY, "#B279A2", "s"),
    ]

    for ax, dim, key_tgt, key_pred, name in [
        (axes[0], 0, "target_valence", "pred_valence", "Valence"),
        (axes[1], 1, "target_arousal", "pred_arousal", "Arousal"),
    ]:
        for rows_key, label, color, marker in styles:
            rows = ctrl.get(rows_key, [])
            by_prompt, _ = _rows_by_tag(rows, tag_order)
            # mean curve across prompts (for n_prompts=1, single curve)
            t_curve, p_curve = [], []
            for tag in tag_order:
                t_vals, p_vals = [], []
                for tags in by_prompt.values():
                    if tag in tags:
                        t_vals.append(tags[tag][key_tgt])
                        p_vals.append(tags[tag][key_pred])
                if t_vals:
                    t_curve.append(np.mean(t_vals))
                    p_curve.append(np.mean(p_vals))
            ax.plot(
                t_curve, p_curve, f"-{marker}", color=color, label=label,
                markersize=8, linewidth=1.5, alpha=0.9,
            )
            for tv, pv in zip(t_curve, p_curve):
                ax.scatter([tv], [pv], c=color, s=40, zorder=3)

        lims = [-1.05, 1.05]
        ax.plot(lims, lims, "k--", linewidth=1, alpha=0.5, label="ideal")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Target {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(name)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Controllability: V-A sweep (low → mid → high), fixed text")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_controllability_circumplex(ctrl, out_path):
    """Russell plot: target vs predicted points for ours & identity sweeps."""
    tag_order = ["low", "mid", "high"]
    tag_colors = {"low": "#4C78A8", "mid": "#999999", "high": "#E45756"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

    for ax, rows_key, title in [
        (axes[0], "ours_rows", f"{LABEL_TRAINED}: predicted V-A"),
        (axes[1], "baseline_rows", f"{LABEL_IDENTITY}: predicted V-A"),
    ]:
        rows = ctrl.get(rows_key, [])
        for r in rows:
            c = tag_colors.get(r["va_tag"], "#999999")
            ax.scatter(
                r["pred_valence"], r["pred_arousal"],
                c=c, s=80, alpha=0.85, edgecolors="white", linewidths=0.5,
            )
            ax.scatter(
                r["target_valence"], r["target_arousal"],
                c=c, s=40, marker="x", alpha=0.9,
            )
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("Valence")
        ax.set_ylabel("Arousal")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        legend_elems = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#888", markersize=8, label="Predicted"),
            Line2D([0], [0], marker="x", color="#888", linestyle="None", markersize=8, label="Target"),
        ]
        ax.legend(handles=legend_elems, fontsize=8, loc="upper right")

    prompt = ctrl.get("fixed_prompt") or "variable prompts"
    fig.suptitle(f"Controllability circumplex — {prompt[:50]}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--results", default="eval_outputs/results.json")
    ap.add_argument("--baseline", default=None, help="e.g. eval_outputs_baseline/results.json")
    ap.add_argument("--gen_dir", default="eval_outputs/generated_wavs")
    ap.add_argument("--test_manifest", default="data/manifests/train_val_test.jsonl/test.jsonl")
    ap.add_argument("--listening", default="eval_outputs/listening_test_sample.json")
    ap.add_argument(
        "--controllability",
        default="eval_outputs/va_controllability.json",
        help="Output of va_controllability_eval.py; skip plots if missing",
    )
    ap.add_argument("--va_regressor", default="checkpoints/va_regressor.pt")
    ap.add_argument("--max_examples", type=int, default=50)
    ap.add_argument("--out_dir", default="report_figures")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    ours = load_json(args.results)
    baseline = load_json(args.baseline) if args.baseline and os.path.exists(args.baseline) else None

    records = load_manifest(args.test_manifest, args.max_examples)
    wav_paths = [os.path.join(args.gen_dir, f"gen_{i:04d}.wav") for i in range(len(records))]
    missing = [p for p in wav_paths if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} generated wav(s), e.g. {missing[0]}. "
            f"Run evaluate.py first or fix --gen_dir."
        )

    target_va = np.array([[r["valence"], r["arousal"]] for r in records], dtype=np.float32)
    prompts = [r["caption"] for r in records]
    pred_va = predict_va_for_wavs(
        wav_paths, args.va_regressor, cfg["eval"]["clap_checkpoint"], args.device
    )

    summary = {
        "ours_results_json": ours,
        "baseline_results_json": baseline,
        "recomputed_from_wavs": {
            "n": int(len(records)),
            "valence_mae": float(np.mean(np.abs(pred_va[:, 0] - target_va[:, 0]))),
            "arousal_mae": float(np.mean(np.abs(pred_va[:, 1] - target_va[:, 1]))),
            "valence_ccc": float(concordance_correlation_coefficient(target_va[:, 0], pred_va[:, 0])),
            "arousal_ccc": float(concordance_correlation_coefficient(target_va[:, 1], pred_va[:, 1])),
        },
    }
    with open(os.path.join(args.out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if args.controllability and os.path.exists(args.controllability):
        ctrl = load_json(args.controllability)
        summary["controllability"] = {
            "path": args.controllability,
            "ours": ctrl.get("ours"),
            "baseline": ctrl.get("baseline"),
            "fixed_prompt": ctrl.get("fixed_prompt"),
            "guidance_scale": ctrl.get("guidance_scale"),
        }
        with open(os.path.join(args.out_dir, "analysis_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        plot_controllability_bars(
            ctrl, os.path.join(args.out_dir, "fig07_controllability_bars.png")
        )
        plot_controllability_target_pred(
            ctrl, os.path.join(args.out_dir, "fig08_controllability_target_pred.png")
        )
        plot_controllability_circumplex(
            ctrl, os.path.join(args.out_dir, "fig09_controllability_circumplex.png")
        )
        print(f"Controllability figures from {args.controllability}")
    elif args.controllability:
        print(f"(skip controllability plots: {args.controllability} not found)")

    write_per_sample_csv(
        target_va, pred_va, prompts,
        os.path.join(args.out_dir, "per_sample_va.csv"),
    )

    plot_metrics_comparison(ours, baseline, os.path.join(args.out_dir, "fig01_metrics_comparison.png"))
    plot_va_scatter(target_va, pred_va, os.path.join(args.out_dir, "fig02_va_scatter.png"))
    plot_circumplex(target_va, pred_va, prompts, os.path.join(args.out_dir, "fig03_circumplex.png"))
    plot_training_summary_table(os.path.join(args.out_dir, "fig04_training_summary.png"))

    if os.path.exists(args.listening):
        listening = load_json(args.listening)
        l_wavs = [item["wav"] for item in listening]
        l_targets = np.array(
            [[item["target_valence"], item["target_arousal"]] for item in listening],
            dtype=np.float32,
        )
        l_prompts = [item["prompt"] for item in listening]
        l_pred = predict_va_for_wavs(
            l_wavs, args.va_regressor, cfg["eval"]["clap_checkpoint"], args.device
        )
        plot_va_scatter(
            l_targets, l_pred,
            os.path.join(args.out_dir, "fig05_listening_va_scatter.png"),
        )
        plot_circumplex(
            l_targets, l_pred, l_prompts,
            os.path.join(args.out_dir, "fig06_listening_circumplex.png"),
        )

    print(json.dumps(summary, indent=2))
    print(f"Figures and tables written to {args.out_dir}/")


if __name__ == "__main__":
    main()
