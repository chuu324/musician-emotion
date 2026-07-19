"""
Recompute FAD with PCA (positive scores). No CLI args needed.

    python scripts/rerun_fad_pca.py

Upload together with utils/fad_score.py (PCA version).
"""
import json
import os
import sys

import torch
import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

JOBS = [
    ("ours", "eval_outputs/generated_wavs", "eval_outputs/results.json"),
    ("baseline", "eval_outputs_baseline/generated_wavs", "eval_outputs_baseline/results.json"),
]


def main():
    print("[rerun_fad_pca] PCA-based FAD, positive scores")

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    ref_dir = cfg.get("eval", {}).get("fad_background_audio_dir", "data/fad_reference")
    if not os.path.isdir(ref_dir):
        raise SystemExit(f"Missing {ref_dir}. Run prepare_fad_reference.py first.")

    from utils.fad_score import compute_fad_with_meta, list_audio_files
    from utils.va_regressor import ClapEmbedder

    print(f"Reference: {len(list_audio_files(ref_dir))} clips in {ref_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = ClapEmbedder(checkpoint=cfg["eval"]["clap_checkpoint"], device=device)

    for name, gen_dir, res_path in JOBS:
        if not os.path.isdir(gen_dir):
            print(f"[skip {name}] no {gen_dir}")
            continue
        meta = compute_fad_with_meta(
            reference_dir=ref_dir,
            generated_dir=gen_dir,
            clap_checkpoint=cfg["eval"]["clap_checkpoint"],
            device=device,
            embedder=embedder,
            use_pca=True,
        )
        print(f"FAD ({name}) = {meta['fad']:.4f}  PCA dims={meta['fad_pca_components']}")

        results = json.load(open(res_path)) if os.path.isfile(res_path) else {}
        results.update(meta)
        results["fad_reference_dir"] = os.path.abspath(ref_dir)
        results["fad_gen_dir"] = os.path.abspath(gen_dir)
        json.dump(results, open(res_path, "w"), indent=2)
        print(f"  -> {res_path}")

    print("Done.")


if __name__ == "__main__":
    main()
