"""
Phase 2/3 training entry point: trains only the emotion-conditioning adapter (and, optionally,
a LoRA on the decoder) while keeping the whole MusicGen backbone frozen.

Usage:
    python train.py --config configs/config.yaml
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import EmotionMusicDataset, collate_fn
from model.musicgen_emotion import MusicGenEmotion


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model: MusicGenEmotion, cfg: dict):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=float(cfg["lr"]), weight_decay=float(cfg["weight_decay"]))


def lm_loss_from_batch(model: MusicGenEmotion, batch, device, prefix_reg_weight=0.0):
    """Teacher-forced language-modeling loss on ground-truth EnCodec tokens."""
    waveforms = batch["waveforms"].to(device)
    audio_codes = model.encode_audio_to_codes(waveforms)          # (B, K, T)
    labels = model.build_labels_from_codes(audio_codes)            # (B, T, K)

    outputs = model(
        texts=batch["captions"],
        valence=batch["valence"],
        arousal=batch["arousal"],
        labels=labels,
    )
    lm_loss = outputs.loss

    prefix_reg = torch.tensor(0.0, device=device)
    if prefix_reg_weight > 0 and model.adapter.use_prefix_token:
        valence = batch["valence"].to(device)
        arousal = batch["arousal"].to(device)
        prefix = model.adapter.compute_scaled_prefix(valence, arousal)
        prefix_reg = (prefix ** 2).mean()

    total_loss = lm_loss + prefix_reg_weight * prefix_reg
    return total_loss, lm_loss, prefix_reg, outputs


def alignment_loss_from_batch(model, batch, va_regressor, clap_embedder, device, temperature=1.0):
    """Optional bonus objective (Section 3, innovation #2): push the *predicted emotion* of a
    short generated clip towards the requested (valence, arousal).

    Implementation note: true end-to-end differentiability through EnCodec decoding + CLAP is
    fragile (categorical sampling in between). Here we approximate it with a soft/Gumbel-softmax
    relaxation over the decoder's next-token logits for a short prefix, decode that soft codebook
    embedding with the frozen EnCodec decoder, and score it with the frozen VA regressor. Treat
    this as a research-grade approximation — disable via `use_alignment_loss: false` in the config
    if it destabilizes training, and rely on the post-hoc metric in evaluate.py instead.
    """
    raise NotImplementedError(
        "alignment_loss_from_batch is a research-grade extension left as a documented stub — "
        "see the docstring above and Section 3 of the proposal for the intended approach. "
        "Wire this up once the primary LM-loss training loop is validated end to end."
    )


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            _, lm_loss, _, _ = lm_loss_from_batch(model, batch, device)
            total_loss += lm_loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["train"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MusicGenEmotion(
        backbone=cfg["model"]["backbone"],
        freeze_backbone=cfg["model"]["freeze_backbone"],
        use_lora=cfg["model"]["use_lora"],
        lora_config=cfg["model"]["lora"],
        adapter_config=cfg["model"]["emotion_adapter"],
    ).to(device)

    train_ds = EmotionMusicDataset(
        manifest_paths=[cfg["data"]["train_manifest"]],
        sample_rate=cfg["data"]["sample_rate"],
        clip_seconds=cfg["data"]["clip_seconds"],
        extra_pseudo_labeled_manifest=cfg["data"].get("extra_pseudo_labeled_manifest"),
    )
    val_ds = EmotionMusicDataset(
        manifest_paths=[cfg["data"]["val_manifest"]],
        sample_rate=cfg["data"]["sample_rate"],
        clip_seconds=cfg["data"]["clip_seconds"],
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=cfg["data"]["num_workers"], collate_fn=collate_fn,
    )

    optimizer = build_optimizer(model, cfg["train"])
    total_steps = len(train_loader) * cfg["train"]["epochs"] // cfg["train"]["grad_accum_steps"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(cfg["train"]["lr"]), total_steps=max(total_steps, 1),
        pct_start=min(cfg["train"]["warmup_steps"] / max(total_steps, 1), 0.3),
    )

    os.makedirs(cfg["train"]["checkpoint_dir"], exist_ok=True)
    best_val_loss = float("inf")
    global_step = 0
    prefix_reg_weight = float(cfg["train"].get("prefix_reg_weight", 0.0))

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        pbar = tqdm(train_loader, desc=f"epoch {epoch}")
        optimizer.zero_grad()
        for step, batch in enumerate(pbar):
            loss, lm_loss, prefix_reg, _ = lm_loss_from_batch(
                model, batch, device, prefix_reg_weight=prefix_reg_weight
            )
            (loss / cfg["train"]["grad_accum_steps"]).backward()

            if (step + 1) % cfg["train"]["grad_accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % cfg["train"]["log_every"] == 0:
                    log_kwargs = {
                        "loss": loss.item(),
                        "lm": lm_loss.item(),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    if prefix_reg_weight > 0 and model.adapter.use_prefix_token:
                        log_kwargs["p_reg"] = prefix_reg.item()
                        log_kwargs["p_scale"] = model.adapter.prefix_scale.item()
                    pbar.set_postfix(**log_kwargs)

                if global_step % cfg["train"]["eval_every_steps"] == 0:
                    val_loss = evaluate(model, val_loader, device)
                    print(f"[step {global_step}] val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(
                            {"adapter": model.adapter.state_dict(), "config": cfg, "step": global_step},
                            os.path.join(cfg["train"]["checkpoint_dir"], "best.pt"),
                        )

                if global_step % cfg["train"]["save_every_steps"] == 0:
                    torch.save(
                        {"adapter": model.adapter.state_dict(), "config": cfg, "step": global_step},
                        os.path.join(cfg["train"]["checkpoint_dir"], f"step_{global_step}.pt"),
                    )

    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
