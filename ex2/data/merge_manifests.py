"""
Merge one or more manifest.jsonl files (e.g. from prepare_deam.py, prepare_pmemo.py,
prepare_emopia.py, auto_label.py) and split into train/val/test manifests.

Usage:
    python data/merge_manifests.py \
        --inputs data/manifests/deam.jsonl data/manifests/pmemo.jsonl \
        --out data/manifests/train_val_test \
        --split 0.8 0.1 0.1
"""
import argparse
import json
import os
import random


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="One or more manifest .jsonl files")
    ap.add_argument("--out", required=True, help="Output directory for train.jsonl/val.jsonl/test.jsonl")
    ap.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    assert abs(sum(args.split) - 1.0) < 1e-6, "--split must sum to 1.0"

    records = []
    for path in args.inputs:
        recs = load_jsonl(path)
        print(f"  loaded {len(recs)} records from {path}")
        records.extend(recs)

    rng = random.Random(args.seed)
    rng.shuffle(records)

    n = len(records)
    n_train = int(n * args.split[0])
    n_val = int(n * args.split[1])

    train_recs = records[:n_train]
    val_recs = records[n_train:n_train + n_val]
    test_recs = records[n_train + n_val:]

    os.makedirs(args.out, exist_ok=True)
    for name, recs in [("train", train_recs), ("val", val_recs), ("test", test_recs)]:
        out_path = os.path.join(args.out, f"{name}.jsonl")
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {len(recs)} records to {out_path}")

    print(f"Total: {n} records -> train={len(train_recs)} val={len(val_recs)} test={len(test_recs)}")


if __name__ == "__main__":
    main()
