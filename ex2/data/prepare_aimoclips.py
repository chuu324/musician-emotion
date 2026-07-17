"""
Build an EVALUATION-ONLY manifest from AImoclips (Go et al., 2025 — "AImoclips: A Benchmark for
Evaluating Emotion Conveyance in Text-to-Music Generation", https://arxiv.org/abs/2509.00813),
the same benchmark the proposal cites in Section 1 as evidence that current TTM systems are
biased toward emotional neutrality.

AImoclips is NOT used for training — it contains AI-generated clips from other TTM systems
(MusicGen, Suno, Udio, ...), not real music, and is far too small (991 clips) and narrow (only
12 emotion-word intents) to fine-tune on. Its value is as an *external, human-rated reference*:

  1. It gives us 12 emotion-word prompts with published, human-perceived (valence, arousal)
     ratings per TTM system. We can prompt OUR system with the same 12 words + target quadrant
     coordinates and compare our CLAP-regressor-predicted V-A (or a small listening test) against
     the paper's human ratings for the existing systems, to see whether the emotion-conditioning
     adapter narrows the "neutrality bias" gap the paper documents.
  2. It's a convenient sanity-check set: independent of DEAM/PMEmo, so it also mildly tests
     generalization of the V-A regressor itself.

Dataset home: https://github.com/HunRotation/HunRotation.github.io
Audio samples: https://hunrotation.github.io/projects/aimoclips.html

Expected layout of the released data (check the repo for the exact filenames/columns, which may
change — this script is written defensively and just needs a CSV with an emotion-intent word/label
column and averaged valence/arousal rating columns; adjust --emotion_col/--valence_col/--arousal_col
if the released column names differ):

    aimoclips_root/
      ratings.csv           <- one row per clip: clip_id, tts_system, emotion_intent,
                               valence_mean, arousal_mean, ...
      clips/                <- audio files, named "<clip_id>.wav" (or .mp3)

Output: data/manifests/aimoclips_eval.jsonl, same schema as the other manifests but with two
extra fields ("tts_system", "emotion_intent") so evaluate.py can break results down per system/intent.
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aimoclips_root", required=True)
    ap.add_argument("--ratings_csv", default=None, help="Auto-discovered under aimoclips_root if omitted")
    ap.add_argument("--audio_subdir", default="clips")
    ap.add_argument("--clip_id_col", default="clip_id")
    ap.add_argument("--system_col", default="tts_system")
    ap.add_argument("--emotion_col", default="emotion_intent")
    ap.add_argument("--valence_col", default="valence_mean")
    ap.add_argument("--arousal_col", default="arousal_mean")
    ap.add_argument(
        "--rating_scale",
        nargs=2,
        type=float,
        default=[1.0, 9.0],
        metavar=("MIN", "MAX"),
        help="The Likert scale AImoclips ratings were collected on (paper uses a 9-point scale); "
             "rescaled to [-1, 1] to match this repo's convention.",
    )
    ap.add_argument("--out", default="data/manifests/aimoclips_eval.jsonl")
    args = ap.parse_args()

    ratings_csv = args.ratings_csv
    if ratings_csv is None:
        candidates = [
            os.path.join(root, f)
            for root, _, files in os.walk(args.aimoclips_root)
            for f in files
            if f.lower().endswith(".csv") and "rating" in f.lower()
        ]
        if not candidates:
            raise FileNotFoundError(
                f"Could not auto-locate a ratings csv under {args.aimoclips_root}; "
                "pass --ratings_csv explicitly (check the released filename in the AImoclips repo)."
            )
        ratings_csv = candidates[0]

    df = pd.read_csv(ratings_csv)
    df.columns = [c.strip() for c in df.columns]

    rmin, rmax = args.rating_scale
    mid = (rmin + rmax) / 2.0
    half_range = (rmax - rmin) / 2.0

    def rescale(x):
        return (float(x) - mid) / half_range

    audio_dir = os.path.join(args.aimoclips_root, args.audio_subdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_written, n_missing = 0, 0
    with open(args.out, "w") as fout:
        for _, row in df.iterrows():
            clip_id = str(row[args.clip_id_col])
            audio_path = None
            for ext in (".wav", ".mp3", ".flac"):
                candidate = os.path.join(audio_dir, f"{clip_id}{ext}")
                if os.path.exists(candidate):
                    audio_path = candidate
                    break
            if audio_path is None:
                n_missing += 1
                continue

            valence = rescale(row[args.valence_col])
            arousal = rescale(row[args.arousal_col])
            emotion_intent = str(row.get(args.emotion_col, "")).strip()

            record = {
                "audio_path": os.path.abspath(audio_path),
                "caption": emotion_intent or "a music clip",
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "source": "aimoclips",
                "is_pseudo_label": False,
                "eval_only": True,
                "tts_system": str(row.get(args.system_col, "")).strip(),
                "emotion_intent": emotion_intent,
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"Wrote {n_written} eval records to {args.out} ({n_missing} rows skipped: audio not found)")
    print("Reminder: AImoclips is for evaluation/benchmarking only — do not add this manifest "
          "to train_manifest / extra_pseudo_labeled_manifest in configs/config.yaml.")


if __name__ == "__main__":
    main()
