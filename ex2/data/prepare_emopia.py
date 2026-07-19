"""
Build a unified manifest.jsonl from EMOPIA (https://annahung31.github.io/EMOPIA/).

EMOPIA labels each clip with one of Russell's four emotion quadrants (Q1..Q4) rather than a
continuous coordinate. We map each quadrant to an approximate (valence, arousal) centroid so it
can be mixed into training as *extra, coarser* supervision alongside DEAM/PMEmo. Because this is
only a quadrant-level approximation, consider down-weighting these examples or treating them as
"soft" targets (e.g. add annotation noise) rather than exact regression targets.

Expected layout:

    EMOPIA_root/
      songs/                 <- audio clips
      label.csv              <- columns: clip_name (or similar), Q1/Q2/Q3/Q4 label

Output format matches prepare_deam.py, with an extra "coarse_label": true flag.
"""
import argparse
import json
import os

import pandas as pd

QUADRANT_TO_VA = {
    "Q1": (0.6, 0.6),    # high valence, high arousal (happy/excited)
    "Q2": (-0.6, 0.6),   # low valence, high arousal (angry/afraid)
    "Q3": (-0.6, -0.6),  # low valence, low arousal (sad/depressed)
    "Q4": (0.6, -0.6),   # high valence, low arousal (calm/content)
}

QUADRANT_CAPTION = {
    "Q1": "happy, exuberant solo piano music",
    "Q2": "tense, agitated solo piano music",
    "Q3": "sad, sorrowful solo piano music",
    "Q4": "calm, contented solo piano music",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emopia_root", required=True)
    ap.add_argument("--label_csv", default=None)
    ap.add_argument("--audio_subdir", default="songs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    label_csv = args.label_csv
    if label_csv is None:
        candidates = [
            os.path.join(root, f)
            for root, _, files in os.walk(args.emopia_root)
            for f in files
            if f.lower().endswith(".csv") and "label" in f.lower()
        ]
        if not candidates:
            raise FileNotFoundError(f"Could not auto-locate a label csv under {args.emopia_root}")
        label_csv = candidates[0]

    df = pd.read_csv(label_csv)
    df.columns = [c.strip() for c in df.columns]
    name_col = df.columns[0]
    quadrant_col = next(c for c in df.columns if "q" in c.lower() and "quad" in c.lower()) \
        if any("quad" in c.lower() for c in df.columns) else df.columns[1]

    audio_dir = os.path.join(args.emopia_root, args.audio_subdir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    n_written, n_missing = 0, 0
    with open(args.out, "w") as fout:
        for _, row in df.iterrows():
            clip_name = str(row[name_col])
            quadrant = str(row[quadrant_col]).strip().upper()
            if quadrant not in QUADRANT_TO_VA:
                # some releases encode quadrant as an int 1-4
                quadrant = f"Q{quadrant}" if quadrant.isdigit() else quadrant
            if quadrant not in QUADRANT_TO_VA:
                continue

            audio_path = os.path.join(audio_dir, clip_name)
            if not os.path.exists(audio_path):
                # try common extensions
                for ext in (".wav", ".mp3"):
                    if os.path.exists(audio_path + ext):
                        audio_path = audio_path + ext
                        break
            if not os.path.exists(audio_path):
                n_missing += 1
                continue

            valence, arousal = QUADRANT_TO_VA[quadrant]
            record = {
                "audio_path": os.path.abspath(audio_path),
                "caption": QUADRANT_CAPTION[quadrant],
                "valence": valence,
                "arousal": arousal,
                "source": "emopia",
                "is_pseudo_label": False,
                "coarse_label": True,
            }
            fout.write(json.dumps(record) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {args.out} ({n_missing} rows skipped: audio not found)")


if __name__ == "__main__":
    main()
