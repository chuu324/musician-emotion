"""
Auto-label an unlabeled music corpus (primarily MTG-Jamendo's mood/theme subset — see README
Section 2.3 for why MTG-Jamendo rather than a generic corpus like MusicCaps) with pseudo
(valence, arousal) coordinates, using the CLAP-embedding regressor trained on DEAM/PMEmo
(utils/va_regressor.py). This implements Section 4.3 of the proposal:

    "Where annotations are scarce, use an emotion-recognition model / CLAP to auto-label
     unlabeled music with V-A values to expand the training set."

Because MTG-Jamendo tracks already carry (coarse, categorical) mood/theme tags, we get a free
noise filter for the pseudo-labeler: if the regressor predicts, say, strongly negative valence
for a clip tagged "happy", that pseudo-label is suspect and can be dropped via
`--mood_tags_json` + `--tag_disagreement_filter`. This cross-check is specific to using an
emotion-tagged corpus as the auto-labeling target, which is another reason to prefer
MTG-Jamendo's mood subset over an emotion-agnostic corpus.

Usage:
    # 1) first train the regressor on your ground-truth manifests (one-time step):
    python utils/va_regressor.py --manifests data/manifests/deam.jsonl data/manifests/pmemo.jsonl \
                                  --out checkpoints/va_regressor.pt

    # 2) then auto-label MTG-Jamendo's mood/theme subset:
    python data/auto_label.py --audio_dir /path/to/mtg_jamendo/audio \
                               --captions_json /path/to/mtg_jamendo/captions.json \
                               --mood_tags_json /path/to/mtg_jamendo/mood_tags.json \
                               --tag_disagreement_filter \
                               --regressor_checkpoint checkpoints/va_regressor.pt \
                               --out data/manifests/mtg_pseudo.jsonl
"""
import argparse
import json
import os

import numpy as np
from tqdm import tqdm

from utils.va_regressor import ClapEmbedder, load_regressor


AUDIO_EXTS = (".mp3", ".wav", ".flac", ".ogg")

# A small, deliberately conservative lexicon mapping common MTG-Jamendo mood/theme tags to their
# *expected sign* on each axis. `None` means "no strong prior on this axis, don't filter on it".
# Extend this as needed; it only needs to catch egregious disagreements, not be exhaustive.
MOOD_TAG_EXPECTED_SIGN = {
    "happy": (1, 1), "fun": (1, 1), "uplifting": (1, 1), "energetic": (None, 1),
    "sad": (-1, None), "melancholic": (-1, -1), "dark": (-1, None), "depressive": (-1, -1),
    "calm": (None, -1), "relaxing": (None, -1), "meditative": (None, -1), "soft": (None, -1),
    "angry": (-1, 1), "aggressive": (-1, 1), "dramatic": (None, 1), "epic": (None, 1),
    "romantic": (1, None), "love": (1, None),
}


def find_audio_files(audio_dir):
    paths = []
    for root, _, files in os.walk(audio_dir):
        for f in files:
            if f.lower().endswith(AUDIO_EXTS):
                paths.append(os.path.join(root, f))
    return sorted(paths)


def _disagrees_with_tags(valence: float, arousal: float, tags: list, min_disagreements: int = 2) -> bool:
    """Return True if the predicted (valence, arousal) sign contradicts >= `min_disagreements` of
    the clip's own mood/theme tags on the same axis, per MOOD_TAG_EXPECTED_SIGN. This is a coarse
    but free noise filter that only exists because we're auto-labeling an already mood-tagged
    corpus (MTG-Jamendo) rather than a generic one (e.g. MusicCaps)."""
    disagreements = 0
    for tag in tags:
        expected = MOOD_TAG_EXPECTED_SIGN.get(tag.lower())
        if expected is None:
            continue
        exp_v, exp_a = expected
        if exp_v is not None and (valence > 0) != (exp_v > 0) and abs(valence) > 0.15:
            disagreements += 1
        if exp_a is not None and (arousal > 0) != (exp_a > 0) and abs(arousal) > 0.15:
            disagreements += 1
    return disagreements >= min_disagreements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", required=True, help="Directory of unlabeled audio to pseudo-label")
    ap.add_argument(
        "--captions_json",
        default=None,
        help="Optional JSON mapping {filename: caption}. If omitted, a generic caption "
             "'music clip' plus the predicted mood quadrant is used.",
    )
    ap.add_argument(
        "--mood_tags_json",
        default=None,
        help="Optional JSON mapping {filename: [mood/theme tags]} (e.g. MTG-Jamendo's own "
             "tags). When given together with --tag_disagreement_filter, pseudo-labels that "
             "strongly contradict their own mood tags (see MOOD_TAG_EXPECTED_SIGN) are dropped.",
    )
    ap.add_argument(
        "--tag_disagreement_filter",
        action="store_true",
        help="Drop a pseudo-label if it disagrees in sign with >=2 of its own mood tags on the "
             "same axis (only meaningful together with --mood_tags_json).",
    )
    ap.add_argument("--regressor_checkpoint", required=True)
    ap.add_argument("--clap_checkpoint", default="laion/larger_clap_music")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--confidence_threshold", type=float, default=None,
                     help="Optional: drop pseudo-labels whose regressor output magnitude is "
                          "implausibly extreme (|v| or |a| > threshold), a simple noise filter.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    captions = {}
    if args.captions_json:
        with open(args.captions_json) as f:
            captions = json.load(f)

    mood_tags = {}
    if args.mood_tags_json:
        with open(args.mood_tags_json) as f:
            mood_tags = json.load(f)

    audio_paths = find_audio_files(args.audio_dir)
    print(f"Found {len(audio_paths)} unlabeled audio files under {args.audio_dir}")

    embedder = ClapEmbedder(checkpoint=args.clap_checkpoint, device=args.device)
    regressor = load_regressor(args.regressor_checkpoint, device=args.device)

    import torch

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    n_written, n_filtered = 0, 0
    with open(args.out, "w") as fout:
        for i in tqdm(range(0, len(audio_paths), args.batch_size), desc="Auto-labeling"):
            batch_paths = audio_paths[i:i + args.batch_size]
            embeds = embedder.embed_audio_files(batch_paths)
            with torch.no_grad():
                va = regressor(torch.from_numpy(np.asarray(embeds, dtype=np.float32)).to(args.device))
            va = va.cpu().numpy()

            for path, (valence, arousal) in zip(batch_paths, va):
                if args.confidence_threshold is not None and (
                    abs(valence) > args.confidence_threshold or abs(arousal) > args.confidence_threshold
                ):
                    n_filtered += 1
                    continue

                fname = os.path.basename(path)

                if args.tag_disagreement_filter and fname in mood_tags:
                    if _disagrees_with_tags(valence, arousal, mood_tags[fname]):
                        n_filtered += 1
                        continue

                caption = captions.get(fname, "a music clip")
                record = {
                    "audio_path": os.path.abspath(path),
                    "caption": caption,
                    "valence": round(float(valence), 4),
                    "arousal": round(float(arousal), 4),
                    "source": "auto_labeled",
                    "is_pseudo_label": True,
                }
                fout.write(json.dumps(record) + "\n")
                n_written += 1

    print(f"Wrote {n_written} pseudo-labeled records to {args.out} ({n_filtered} filtered out)")


if __name__ == "__main__":
    main()
