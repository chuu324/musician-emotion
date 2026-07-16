"""
Build a unified manifest.jsonl from PMEmo / PMEmo2019
(https://github.com/HuiZhangDB/PMEmo).

Expected layout (PMEmo2019 official release):

    PMEmo2019_root/
      chorus/                       <- audio clips (mp3), the annotated "chorus" segment per song
      annotations/
        static_annotations.csv      <- columns: musicId, Valence(mean), Arousal(mean), ...
        comments.csv                <- optional metadata (song/artist names) for captions

PMEmo's static annotations are already roughly in [-1, 1] to [0,1]-ish depending on release;
this script assumes the common PMEmo2019 static_annotations.csv range of [0, 1] for both
Valence(mean) and Arousal(mean) and rescales to [-1, 1]. **Check your actual CSV's min/max
and adjust `rescale()` if your copy differs** (PMEmo has had a couple of annotation releases).

Output format matches prepare_deam.py.
"""
import argparse
import json
import os

import pandas as pd


def rescale_0_1_to_minus1_1(x: float) -> float:
    return x * 2.0 - 1.0


def quadrant_caption(valence: float, arousal: float) -> str:
    if valence >= 0 and arousal >= 0:
        return "upbeat pop music, bright and energetic"
    if valence < 0 and arousal >= 0:
        return "dramatic, intense pop music"
    if valence < 0 and arousal < 0:
        return "sad, downtempo pop ballad"
    return "mellow, laid-back pop music"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pmemo_root", required=True, help="Path to the unzipped PMEmo2019 root")
    ap.add_argument("--static_csv", default=None, help="Path to static_annotations.csv (auto-discovered if omitted)")
    ap.add_argument("--audio_subdir", default="chorus")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    static_csv = args.static_csv
    if static_csv is None:
        candidates = []
        for root, _, files in os.walk(args.pmemo_root):
            for f in files:
                if "static" in f.lower() and f.endswith(".csv"):
                    candidates.append(os.path.join(root, f))
        if not candidates:
            raise FileNotFoundError(
                f"Could not auto-locate static_annotations*.csv under {args.pmemo_root}; "
                "pass --static_csv explicitly."
            )
        static_csv = candidates[0]

    df = pd.read_csv(static_csv)
    df.columns = [c.strip() for c in df.columns]
    id_col = next(c for c in df.columns if c.lower() in ("musicid", "music_id", "songid", "song_id"))
    val_col = next(c for c in df.columns if "valence" in c.lower())
    aro_col = next(c for c in df.columns if "arousal" in c.lower())

    audio_dir = os.path.join(args.pmemo_root, args.audio_subdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_written, n_missing = 0, 0
    with open(args.out, "w") as fout:
        for _, row in df.iterrows():
            music_id = int(row[id_col])
            # PMEmo audio filenames are typically "<musicId>.mp3"
            audio_path = os.path.join(audio_dir, f"{music_id}.mp3")
            if not os.path.exists(audio_path):
                n_missing += 1
                continue

            raw_v, raw_a = float(row[val_col]), float(row[aro_col])
            # Heuristic: if values look like they're already roughly centered in [-1,1], don't rescale.
            if -1.05 <= raw_v <= 1.05 and -1.05 <= raw_a <= 1.05:
                valence, arousal = raw_v, raw_a
            else:
                valence, arousal = rescale_0_1_to_minus1_1(raw_v), rescale_0_1_to_minus1_1(raw_a)

            record = {
                "audio_path": os.path.abspath(audio_path),
                "caption": quadrant_caption(valence, arousal),
                "valence": round(valence, 4),
                "arousal": round(arousal, 4),
                "source": "pmemo",
                "is_pseudo_label": False,
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {args.out} ({n_missing} rows skipped: audio not found)")


if __name__ == "__main__":
    main()
