"""
One-shot FAD for ours + baseline using local CLAP (no frechet_audio_distance / no roberta download).

    python scripts/run_fad_all.py

Requires data/fad_reference/ and utils/fad_score.py + utils/clap_ckpt.py
"""
import json
import os
import sys

import torch
import yaml

SCRIPT_VERSION = "2026-07-16-clap-fad-v3-pca"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

JOBS = [
    ("ours", "eval_outputs/generated_wavs", "eval_outputs/results.json"),
    ("baseline", "eval_outputs_baseline/generated_wavs", "eval_outputs_baseline/results.json"),
]


def run_job(name, gen_dir, results_path, background_dir, embedder, cfg):
    if not os.path.isdir(gen_dir):
        print(f"[skip {name}] missing gen_dir: {gen_dir}")
        return

    from utils.fad_score import compute_fad_with_meta, list_audio_files

    n_gen = len(list_audio_files(gen_dir))
    print(f"\n=== FAD: {name} ({n_gen} generated clips) ===")

    meta = compute_fad_with_meta(
        reference_dir=background_dir,
        generated_dir=gen_dir,
        clap_checkpoint=cfg["eval"]["clap_checkpoint"],
        device=embedder.device,
        embedder=embedder,
        use_pca=True,
    )
    score = meta["fad"]
    print(f"FAD ({name}) = {score:.4f}  (PCA dims={meta['fad_pca_components']})")

    results = {}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results.update(meta)
    results["fad_reference_dir"] = os.path.abspath(background_dir)
    results["fad_gen_dir"] = os.path.abspath(gen_dir)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {results_path}")


def main():
    print(f"[run_fad_all {SCRIPT_VERSION}] cwd={os.getcwd()}")

    if not os.environ.get("HF_ENDPOINT"):
        print("Tip: export HF_ENDPOINT=https://hf.mirror.com  (if CLAP init needs HF)")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    background_dir = cfg.get("eval", {}).get("fad_background_audio_dir", "data/fad_reference")
    if not os.path.isdir(background_dir):
        raise SystemExit(
            f"Reference dir not found: {background_dir}\n"
            "Run: python scripts/prepare_fad_reference.py --max_files 50"
        )

    from utils.fad_score import list_audio_files
    from utils.va_regressor import ClapEmbedder

    n_ref = len(list_audio_files(background_dir))
    print(f"Reference: {background_dir} ({n_ref} clips)")
    if n_ref == 0:
        raise SystemExit("data/fad_reference is empty.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLAP once on {device} ...")
    embedder = ClapEmbedder(checkpoint=cfg["eval"]["clap_checkpoint"], device=device)

    for name, gen_dir, results_path in JOBS:
        run_job(name, gen_dir, results_path, background_dir, embedder, cfg)

    print("\nDone:")
    for _, _, rp in JOBS:
        if os.path.isfile(rp):
            fad = json.load(open(rp)).get("fad")
            print(f"  {rp}: fad={fad}")


if __name__ == "__main__":
    main()
