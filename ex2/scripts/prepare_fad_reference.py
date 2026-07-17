"""
Copy held-out real audio clips into data/fad_reference for FAD evaluation.

FAD compares the *distribution* of generated wavs vs a folder of reference real music.
Use the test split (never seen during adapter training) as the reference set.

Usage:
    python scripts/prepare_fad_reference.py --config configs/config.yaml
    python scripts/prepare_fad_reference.py --manifest data/manifests/train_val_test.jsonl/test.jsonl --max_files 50
"""
import argparse
import json
import os
import shutil
import sys

import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--manifest", default=None, help="Override test manifest path")
    ap.add_argument("--out_dir", default="data/fad_reference")
    ap.add_argument("--max_files", type=int, default=200)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    manifest_path = args.manifest or cfg["data"]["test_manifest"]
    os.makedirs(args.out_dir, exist_ok=True)

    copied, missing = 0, 0
    with open(manifest_path) as f:
        for line in f:
            if copied >= args.max_files:
                break
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec.get("audio_path")
            if not src or not os.path.isfile(src):
                missing += 1
                continue
            ext = os.path.splitext(src)[1] or ".wav"
            dst = os.path.join(args.out_dir, f"ref_{copied:04d}{ext}")
            shutil.copy2(src, dst)
            copied += 1

    print(f"FAD reference dir: {os.path.abspath(args.out_dir)}")
    print(f"Copied {copied} clips (missing/skipped: {missing})")
    if copied == 0:
        raise SystemExit(
            "No audio copied — check manifest paths exist on this machine. "
            "Run from AutoDL where DEAM/PMEmo audio is stored."
        )


if __name__ == "__main__":
    main()
