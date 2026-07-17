"""
Phase 4 evaluation (Section 5 of the proposal):

  - Audio quality:            FAD (Fréchet Audio Distance) against a reference set (e.g. MusicCaps/DEAM)
  - Text-music alignment:     CLAP score (cosine similarity between prompt and generated-audio embeddings)
  - Emotion fidelity:         predicted-vs-target (valence, arousal) MAE/RMSE/CCC using the same
                               CLAP-embedding regressor trained in utils/va_regressor.py

Also dumps a random subsample of generations for a small-scale subjective listening test
(`eval.listening_test_sample_size` in the config).

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
"""
import argparse
import json
import os
import random

import numpy as np
import soundfile as sf
import torch
import yaml
from tqdm import tqdm

from model.musicgen_emotion import MusicGenEmotion
from utils.va_regressor import ClapEmbedder, load_regressor


def concordance_correlation_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Lin's CCC — standard metric for affect-regression agreement, stricter than plain Pearson r
    since it also penalizes shifts in mean/scale, not just correlation of direction."""
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    var_true, var_pred = y_true.var(), y_pred.var()
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean()
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)


def load_test_manifest(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="eval_outputs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max_examples", type=int, default=200, help="Cap on number of test prompts to evaluate")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs(args.out_dir, exist_ok=True)
    gen_dir = os.path.join(args.out_dir, "generated_wavs")
    os.makedirs(gen_dir, exist_ok=True)

    # --- load model -----------------------------------------------------------------
    model = MusicGenEmotion(
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        use_lora=cfg["model"]["use_lora"],
        lora_config=cfg["model"]["lora"],
        adapter_config=cfg["model"]["emotion_adapter"],
    ).to(args.device)
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.adapter.load_state_dict(ckpt["adapter"])
    model.eval()

    # --- load test prompts ------------------------------------------------------------
    test_records = load_test_manifest(cfg["data"]["test_manifest"])[: args.max_examples]
    print(f"Evaluating on {len(test_records)} test examples")

    # --- generate ----------------------------------------------------------------------
    generated_paths, target_va, prompts = [], [], []
    for i, rec in enumerate(tqdm(test_records, desc="Generating")):
        audio = model.generate(
            text=rec["caption"],
            valence=rec["valence"],
            arousal=rec["arousal"],
            duration_seconds=cfg["eval"]["generation_duration_seconds"],
        )
        audio = audio.squeeze().detach().cpu().numpy()
        out_path = os.path.join(gen_dir, f"gen_{i:04d}.wav")
        sf.write(out_path, audio, model.musicgen.config.audio_encoder.sampling_rate)
        generated_paths.append(out_path)
        target_va.append((rec["valence"], rec["arousal"]))
        prompts.append(rec["caption"])

    # --- CLAP score (text-music alignment) + emotion fidelity ------------------------
    embedder = ClapEmbedder(checkpoint=cfg["eval"]["clap_checkpoint"], device=args.device)
    regressor = None
    va_reg_ckpt = os.path.join(cfg["train"]["checkpoint_dir"], "va_regressor.pt")
    if os.path.exists(va_reg_ckpt):
        regressor = load_regressor(va_reg_ckpt, device=args.device)
    else:
        print(f"WARNING: no VA regressor found at {va_reg_ckpt}; skipping emotion-fidelity metric. "
              f"Run utils/va_regressor.py first.")

    audio_embeds = embedder.embed_audio_files(generated_paths)
    text_embeds = embedder.embed_text(prompts)

    audio_embeds_n = audio_embeds / np.linalg.norm(audio_embeds, axis=-1, keepdims=True)
    text_embeds_n = text_embeds / np.linalg.norm(text_embeds, axis=-1, keepdims=True)
    clap_scores = (audio_embeds_n * text_embeds_n).sum(axis=-1)
    mean_clap_score = float(np.mean(clap_scores))

    results = {
        "n_examples": len(test_records),
        "mean_clap_score": mean_clap_score,
    }

    if regressor is not None:
        with torch.no_grad():
            pred_va = regressor(torch.from_numpy(audio_embeds.astype(np.float32)).to(args.device))
        pred_va = pred_va.cpu().numpy()
        target_va_arr = np.array(target_va, dtype=np.float32)

        mae = np.mean(np.abs(pred_va - target_va_arr), axis=0)
        rmse = np.sqrt(np.mean((pred_va - target_va_arr) ** 2, axis=0))
        ccc_valence = concordance_correlation_coefficient(target_va_arr[:, 0], pred_va[:, 0])
        ccc_arousal = concordance_correlation_coefficient(target_va_arr[:, 1], pred_va[:, 1])

        results.update({
            "valence_mae": float(mae[0]), "arousal_mae": float(mae[1]),
            "valence_rmse": float(rmse[0]), "arousal_rmse": float(rmse[1]),
            "valence_ccc": float(ccc_valence), "arousal_ccc": float(ccc_arousal),
        })

    # --- FAD (local CLAP embedder; avoids frechet_audio_distance HF downloads) ---
    background_dir = cfg["eval"].get("fad_background_audio_dir")
    if background_dir and os.path.isdir(background_dir):
        try:
            from utils.fad_score import compute_fad_score

            fad_score = compute_fad_score(
                reference_dir=background_dir,
                generated_dir=gen_dir,
                clap_checkpoint=cfg["eval"]["clap_checkpoint"],
                device=args.device,
                embedder=embedder,
            )
            results["fad"] = float(fad_score)
            results["fad_method"] = "clap_embedder_local"
        except Exception as exc:
            print(f"FAD skipped: {exc}")
    else:
        print("No `eval.fad_background_audio_dir` configured / found — skipping FAD. "
              "Point it at a folder of real reference clips (e.g. a MusicCaps/DEAM subset).")

    # --- listening-test sample -------------------------------------------------------
    sample_size = min(cfg["eval"]["listening_test_sample_size"], len(generated_paths))
    sample_idx = random.sample(range(len(generated_paths)), sample_size)
    listening_test_manifest = [
        {"wav": generated_paths[i], "prompt": prompts[i], "target_valence": target_va[i][0],
         "target_arousal": target_va[i][1]}
        for i in sample_idx
    ]
    with open(os.path.join(args.out_dir, "listening_test_sample.json"), "w") as f:
        json.dump(listening_test_manifest, f, indent=2)

    with open(os.path.join(args.out_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Full results + listening-test sample written to {args.out_dir}/")

    # --- optional: compare against the AImoclips external benchmark -------------------
    aimoclips_path = cfg["eval"].get("aimoclips_manifest")
    if aimoclips_path and os.path.exists(aimoclips_path):
        evaluate_against_aimoclips(model, embedder, regressor, aimoclips_path, args.out_dir, args.device)
    elif aimoclips_path:
        print(f"(aimoclips_manifest configured at {aimoclips_path} but file not found — "
              f"run data/prepare_aimoclips.py first if you want this comparison. Skipping.)")


def evaluate_against_aimoclips(model, embedder, regressor, manifest_path, out_dir, device):
    """Generate on AImoclips' own 12 emotion-word intents / target quadrants and compare our
    predicted V-A against the paper's human-rated V-A for the *existing* TTM systems, to see
    whether the emotion adapter narrows the "neutrality bias" AImoclips reports for every system
    it tested (Section 5 external validation, README Sec 2.4)."""
    records = load_test_manifest(manifest_path)
    # one generation per distinct emotion_intent is enough for this comparison — AImoclips itself
    # only varies TTM system per intent, not the prompt content.
    seen_intents = {}
    for r in records:
        seen_intents.setdefault(r["emotion_intent"], r)
    intents = list(seen_intents.values())

    rows = []
    for rec in tqdm(intents, desc="Generating for AImoclips comparison"):
        audio = model.generate(text=rec["emotion_intent"], valence=rec["valence"], arousal=rec["arousal"],
                                duration_seconds=10.0)
        audio_np = audio.squeeze().detach().cpu().numpy()
        tmp_path = os.path.join(out_dir, f"aimoclips_cmp_{rec['emotion_intent'].replace(' ', '_')}.wav")
        sf.write(tmp_path, audio_np, model.musicgen.config.audio_encoder.sampling_rate)

        embed = embedder.embed_audio_files([tmp_path])
        if regressor is not None:
            with torch.no_grad():
                pred_va = regressor(torch.from_numpy(embed.astype(np.float32)).to(device)).cpu().numpy()[0]
        else:
            pred_va = (None, None)

        rows.append({
            "emotion_intent": rec["emotion_intent"],
            "target_valence": rec["valence"], "target_arousal": rec["arousal"],
            "our_predicted_valence": float(pred_va[0]) if pred_va[0] is not None else None,
            "our_predicted_arousal": float(pred_va[1]) if pred_va[1] is not None else None,
            "aimoclips_human_rated_valence_for_reference_systems": rec["valence"],
            "aimoclips_human_rated_arousal_for_reference_systems": rec["arousal"],
        })

    with open(os.path.join(out_dir, "aimoclips_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"AImoclips comparison ({len(rows)} emotion intents) written to "
          f"{out_dir}/aimoclips_comparison.json — compare `our_predicted_*` against the reference "
          f"systems' human ratings reported in Table 2/3 of the AImoclips paper.")


if __name__ == "__main__":
    main()
