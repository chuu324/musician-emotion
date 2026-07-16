"""
Compute FAD on already-generated wavs (no re-generation).

Uses local ClapEmbedder — does NOT call frechet_audio_distance (avoids roberta-base download).

Usage:
    python scripts/compute_fad_only.py
    python scripts/compute_fad_only.py --gen_dir eval_outputs/generated_wavs --results eval_outputs/results.json
"""
import json
import os
import sys

import torch
import yaml

SCRIPT_VERSION = "2026-07-16-clap-fad-v2"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)


def parse_argv(argv):
    gen_dir = "eval_outputs/generated_wavs"
    results = "eval_outputs/results.json"
    background_dir = None
    config = "configs/config.yaml"

    i = 0
    while i < len(argv):
        a = argv[i]
        if a in ("--version", "-V"):
            print(f"compute_fad_only {SCRIPT_VERSION}")
            print(f"  file: {os.path.abspath(__file__)}")
            sys.exit(0)
        if a in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        if a == "--gen_dir" and i + 1 < len(argv):
            gen_dir = argv[i + 1]
            i += 2
            continue
        if a == "--results" and i + 1 < len(argv):
            results = argv[i + 1]
            i += 2
            continue
        if a == "--background_dir" and i + 1 < len(argv):
            background_dir = argv[i + 1]
            i += 2
            continue
        if a == "--config" and i + 1 < len(argv):
            config = argv[i + 1]
            i += 2
            continue
        print(f"Unknown argument: {a!r}", file=sys.stderr)
        sys.exit(2)
        i += 1

    return config, gen_dir, results, background_dir


def main():
    config_path, gen_dir, results_path, background_override = parse_argv(sys.argv[1:])

    print(f"[compute_fad_only {SCRIPT_VERSION}]")
    print(f"  gen_dir={gen_dir}  results={results_path}")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    background_dir = background_override or cfg.get("eval", {}).get("fad_background_audio_dir")
    if not background_dir or not os.path.isdir(background_dir):
        raise SystemExit(f"Reference dir not found: {background_dir}")

    from utils.fad_score import compute_fad_score, list_audio_files

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fad_score = compute_fad_score(
        reference_dir=background_dir,
        generated_dir=gen_dir,
        clap_checkpoint=cfg["eval"]["clap_checkpoint"],
        device=device,
    )
    print(f"FAD = {fad_score:.4f}")

    results = {}
    if os.path.isfile(results_path):
        with open(results_path) as f:
            results = json.load(f)
    results["fad"] = fad_score
    results["fad_method"] = "clap_embedder_local"
    results["fad_reference_dir"] = os.path.abspath(background_dir)
    results["fad_gen_dir"] = os.path.abspath(gen_dir)

    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Updated {results_path}")


if __name__ == "__main__":
    main()
