"""
Measure V-A *controllability*: same text prompt, only change (valence, arousal).

The standard evaluate.py CCC can favor identity because captions in DEAM/PMEmo
already encode emotion — MusicGen follows text, and the regressor reads it back.
This script isolates whether *changing V-A alone* shifts predicted emotion.

Usage:
    python scripts/va_controllability_eval.py \
        --checkpoint checkpoints/best.pt \
        --baseline_checkpoint checkpoints/identity.pt \
        --n_prompts 10 \
        --out eval_outputs/va_controllability.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
os.chdir(_ROOT)

from model.musicgen_emotion import MusicGenEmotion
from utils.va_regressor import ClapEmbedder, load_regressor


VA_PAIRS = [
    ("low", -0.9, -0.9),
    ("mid", 0.0, 0.0),
    ("high", 0.9, 0.9),
]


def load_manifest(path, n):
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    return records[:n]


@torch.no_grad()
def run_model(model, embedder, regressor, prompts, guidance_scale, duration, device, out_subdir):
    os.makedirs(out_subdir, exist_ok=True)
    rows = []
    for pi, prompt in enumerate(prompts):
        for tag, v, a in VA_PAIRS:
            audio = model.generate(
                text=prompt, valence=v, arousal=a,
                duration_seconds=duration, guidance_scale=guidance_scale, device=device,
            )
            wav_path = os.path.join(out_subdir, f"p{pi:02d}_{tag}.wav")
            sr = model.musicgen.config.audio_encoder.sampling_rate
            sf.write(wav_path, audio.squeeze().detach().cpu().numpy(), sr)

            embed = embedder.embed_audio_files([wav_path])
            pred = regressor(torch.from_numpy(embed.astype(np.float32)).to(device)).cpu().numpy()[0]

            rows.append({
                "prompt_idx": pi, "prompt": prompt, "va_tag": tag,
                "target_valence": v, "target_arousal": a,
                "pred_valence": float(pred[0]), "pred_arousal": float(pred[1]),
                "wav": wav_path,
            })
    return rows


def summarize(rows):
    """Per-prompt spread: |pred(high) - pred(low)|; mean across prompts."""
    by_prompt = {}
    for r in rows:
        by_prompt.setdefault(r["prompt_idx"], {})[r["va_tag"]] = r

    v_spreads, a_spreads = [], []
    for tags in by_prompt.values():
        if "low" in tags and "high" in tags:
            v_spreads.append(abs(tags["high"]["pred_valence"] - tags["low"]["pred_valence"]))
            a_spreads.append(abs(tags["high"]["pred_arousal"] - tags["low"]["pred_arousal"]))

    # Target-controlled correlation across low/mid/high per prompt
    all_t_v, all_p_v, all_t_a, all_p_a = [], [], [], []
    for tags in by_prompt.values():
        for tag in ("low", "mid", "high"):
            if tag in tags:
                all_t_v.append(tags[tag]["target_valence"])
                all_p_v.append(tags[tag]["pred_valence"])
                all_t_a.append(tags[tag]["target_arousal"])
                all_p_a.append(tags[tag]["pred_arousal"])

    def pearson(x, y):
        x, y = np.asarray(x), np.asarray(y)
        if x.std() < 1e-8 or y.std() < 1e-8:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    return {
        "n_prompts": len(by_prompt),
        "mean_valence_spread_low_vs_high": float(np.mean(v_spreads)) if v_spreads else None,
        "mean_arousal_spread_low_vs_high": float(np.mean(a_spreads)) if a_spreads else None,
        "pearson_target_vs_pred_valence": pearson(all_t_v, all_p_v),
        "pearson_target_vs_pred_arousal": pearson(all_t_a, all_p_a),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--baseline_checkpoint", default="checkpoints/identity.pt")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--n_prompts", type=int, default=10)
    ap.add_argument("--fixed_prompt", default=None,
                    help='If set, ignore manifest and use this single neutral prompt for all runs')
    ap.add_argument("--guidance_scale", type=float, default=1.0)
    ap.add_argument("--duration_seconds", type=float, default=None,
                    help="Override eval.generation_duration_seconds (e.g. 5 for faster runs)")
    ap.add_argument("--out", default="eval_outputs/va_controllability.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.fixed_prompt:
        prompts = [args.fixed_prompt] * args.n_prompts
    else:
        manifest = args.manifest or cfg["data"]["test_manifest"]
        prompts = [r["caption"] for r in load_manifest(manifest, args.n_prompts)]

    def load_ckpt(path):
        m = MusicGenEmotion(
            backbone=cfg["model"]["backbone"],
            freeze_backbone=cfg["model"]["freeze_backbone"],
            use_lora=cfg["model"]["use_lora"],
            lora_config=cfg["model"]["lora"],
            adapter_config=cfg["model"]["emotion_adapter"],
        ).to(args.device)
        ckpt = torch.load(path, map_location=args.device)
        m.adapter.load_state_dict(ckpt["adapter"])
        m.eval()
        return m

    embedder = ClapEmbedder(checkpoint=cfg["eval"]["clap_checkpoint"], device=args.device)
    regressor = load_regressor(
        os.path.join(cfg["train"]["checkpoint_dir"], "va_regressor.pt"), device=args.device
    )
    duration = args.duration_seconds or cfg["eval"]["generation_duration_seconds"]
    out_root = os.path.dirname(args.out) or "eval_outputs"

    print(f"Controllability eval: {len(prompts)} prompts × 3 VA × 2 models, {duration}s each")
    ours_model = load_ckpt(args.checkpoint)
    ours_rows = run_model(
        ours_model, embedder, regressor, prompts, args.guidance_scale, duration,
        args.device, os.path.join(out_root, "va_ctrl_ours"),
    )
    base_model = load_ckpt(args.baseline_checkpoint)
    base_rows = run_model(
        base_model, embedder, regressor, prompts, args.guidance_scale, duration,
        args.device, os.path.join(out_root, "va_ctrl_baseline"),
    )

    result = {
        "guidance_scale": args.guidance_scale,
        "fixed_prompt": args.fixed_prompt,
        "ours": summarize(ours_rows),
        "baseline": summarize(base_rows),
        "ours_rows": ours_rows,
        "baseline_rows": base_rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({"ours": result["ours"], "baseline": result["baseline"]}, indent=2))
    print(f"Written to {args.out}")
    print("\nInterpretation: prioritize Pearson (target vs pred) — positive = correct V-A control.")
    print("Spread alone can mislead on identity (sampling noise). See docs/report_material.md §5.5.")


if __name__ == "__main__":
    main()
