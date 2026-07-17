"""
Build a unified manifest.jsonl from the DEAM dataset
(MediaEval Database for Emotional Analysis in Music, http://cvml.unige.ch/databases/DEAM/).

Expected DEAM layout after unzipping the official archives (adjust --deam_root if yours differs):

    DEAM_root/
      MEMD_audio/                      <- 1802 mp3 clips, named "<song_id>.mp3"
      annotations/
        annotations averaged per song/
          song_level/
            static_annotations_averaged_songs_1_2000.csv   <- columns: song_id, valence_mean, arousal_mean, ...
        annotations averaged per song/dynamic (per-second)/  <- optional, not required here

DEAM's static valence/arousal means are on a 1-9 scale; we rescale to [-1, 1] to match the
adapter's expected input range (configs/config.yaml: model.emotion_adapter.va_range).

Output: one JSON object per line:
    {"audio_path": ..., "caption": ..., "valence": float, "arousal": float,
     "source": "deam", "is_pseudo_label": false}

Since DEAM does not ship text captions, we synthesize a light-weight caption from the
valence/arousal quadrant + genre metadata (if available in metadata csv) — good enough as a
weak text-conditioning signal; feel free to replace with a captioning model (e.g. run MusicCaps'
captioner or an audio-captioning model) for higher-quality prompts.
"""
import argparse
import json
import os

import pandas as pd


def rescale_1_9_to_minus1_1(x: float) -> float:
    """DEAM annotations are on a 1-9 Likert scale; map to [-1, 1]."""
    return (x - 5.0) / 4.0


def quadrant_caption(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "upbeat, joyful and energetic music"
    if valence < 0 and arousal >= 0:
        return "tense, aggressive and intense music"
    if valence < 0 and arousal < 0:
        return "melancholic, sad and low-energy music"
    return "calm, peaceful and relaxed music"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deam_root", required=True, help="Path to the unzipped DEAM dataset root")
    ap.add_argument(
        "--static_csv",
        default=None,
        help="Path to static_annotations_averaged_songs_*.csv (auto-discovered if omitted)",
    )
    ap.add_argument("--audio_subdir", default="MEMD_audio")
    ap.add_argument("--out", required=True, help="Output manifest .jsonl path")
    args = ap.parse_args()

    static_csv = args.static_csv
    if static_csv is None:
        candidates = []
        target_full = None
        for root, _, files in os.walk(args.deam_root):
            for f in files:
                if f.startswith("static_annotations_averaged_songs") and f.endswith(".csv"):
                    file_path = os.path.join(root, f)
                    candidates.append(file_path)
                    if "1_2000" in f:
                        target_full = file_path
        if not candidates:
            raise FileNotFoundError(
                "Could not auto-locate static_annotations_averaged_songs_*.csv under "
                f"{args.deam_root}; pass --static_csv explicitly."
            )
        static_csv = target_full if target_full else candidates[0]
    df = pd.read_csv(static_csv)
    df.columns = [c.strip() for c in df.columns]
    # DEAM's csv has columns like: song_id, valence_mean, arousal_mean, valence_std, arousal_std
    audio_dir = os.path.join(args.deam_root, args.audio_subdir)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written, n_missing = 0, 0
    with open(args.out, "w") as fout:
        for _, row in df.iterrows():
            song_id = int(row["song_id"])
            audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
            if not os.path.exists(audio_path):
                n_missing += 1
                continue

            valence = rescale_1_9_to_minus1_1(float(row["valence_mean"]))
            arousal = rescale_1_9_to_minus1_1(float(row["arousal_mean"]))
            record = {
                "audio_path": os.path.abspath(audio_path),
                "caption": quadrant_caption(valence, arousal),
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "source": "deam",
                "is_pseudo_label": False,
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {args.out} ({n_missing} rows skipped: audio not found)")


if __name__ == "__main__":
    main()
