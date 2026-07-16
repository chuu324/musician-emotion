"""
Inference: generate a BGM clip from a text prompt plus a continuous (valence, arousal) coordinate.

Usage:
    python generate.py --prompt "gentle acoustic guitar for a rainy afternoon" \
                        --valence 0.7 --arousal -0.3 \
                        --checkpoint checkpoints/best.pt \
                        --out out.wav
"""
import argparse

import soundfile as sf
import torch
import yaml

from model.musicgen_emotion import MusicGenEmotion


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--checkpoint", default=None, help="Path to a trained adapter checkpoint (.pt). "
                                                          "If omitted, runs with a randomly-initialized "
                                                          "adapter, which is only useful for a smoke test.")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--valence", type=float, required=True, help="In [-1, 1]: negative=unpleasant, positive=pleasant")
    ap.add_argument("--arousal", type=float, required=True, help="In [-1, 1]: negative=calm, positive=energetic")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--guidance_scale", type=float, default=3.0)
    ap.add_argument("--out", default="out.wav")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGenEmotion(
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        use_lora=cfg["model"]["use_lora"],
        lora_config=cfg["model"]["lora"],
        adapter_config=cfg["model"]["emotion_adapter"],
    ).to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.adapter.load_state_dict(ckpt["adapter"])
        print(f"Loaded adapter weights from {args.checkpoint} (step {ckpt.get('step', '?')})")
    else:
        print("WARNING: no --checkpoint given, using a randomly-initialized adapter (smoke test only).")

    audio = model.generate(
        text=args.prompt,
        valence=args.valence,
        arousal=args.arousal,
        duration_seconds=args.duration,
        guidance_scale=args.guidance_scale,
        device=device,
    )
    audio = audio.squeeze().detach().cpu().numpy()

    sample_rate = model.musicgen.config.audio_encoder.sampling_rate
    sf.write(args.out, audio, sample_rate)
    print(f"Wrote {args.duration}s of audio to {args.out} at {sample_rate} Hz "
          f"(prompt='{args.prompt}', valence={args.valence}, arousal={args.arousal})")


if __name__ == "__main__":
    main()
