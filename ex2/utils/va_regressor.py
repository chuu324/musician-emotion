"""
A small (valence, arousal) regression probe on top of frozen CLAP audio embeddings.

Used in two places:
  1. data/auto_label.py   - pseudo-label a large unlabeled corpus (e.g. MTG-Jamendo) so it can be
                             added to training (Section 4.3 of the proposal: "use an emotion-recognition
                             model / CLAP to auto-label unlabeled music with V-A values").
  2. evaluate.py           - "emotion fidelity" metric (Section 5): predict V-A of *generated* audio
                             and compare it against the target V-A that was requested.

The regressor itself is a 2-layer MLP trained on (CLAP embedding -> ground-truth V-A) pairs from
DEAM + PMEmo. CLAP and the regressor are both frozen at generation-evaluation time.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class VARegressor(nn.Module):
    """CLAP embedding (default 512-d) -> (valence, arousal), both in [-1, 1]."""

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
            nn.Tanh(),  # outputs constrained to [-1, 1]
        )

    def forward(self, clap_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(clap_embeddings)


class ClapEmbedder:
    """Thin wrapper around laion_clap so callers don't need to know its API."""

    DEFAULT_MUSIC_CKPT = "checkpoints/clap_weights/music_audioset_epoch_15_esc_90.14.pt"

    def __init__(self, checkpoint: str = "laion/larger_clap_music", device: str = "cuda"):
        import sys

        # laion_clap's training entry point parses sys.argv on import; strip CLI args first.
        _argv = sys.argv
        sys.argv = [_argv[0]]
        try:
            import laion_clap  # imported lazily so this file can be inspected without the dep installed
        finally:
            sys.argv = _argv

        ckpt_path, amodel = self._resolve_clap_checkpoint(checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"CLAP checkpoint not found: {ckpt_path}\n"
                "Download it first (uses HF mirror):\n"
                "  export HF_ENDPOINT=https://hf-mirror.com\n"
                "  mkdir -p checkpoints/clap_weights\n"
                "  huggingface-cli download lukewys/laion_clap "
                "music_audioset_epoch_15_esc_90.14.pt "
                "--local-dir checkpoints/clap_weights\n"
                "Or set env CLAP_CKPT_PATH=/path/to/music_audioset_epoch_15_esc_90.14.pt"
            )

        module_kwargs = {"enable_fusion": False}
        if amodel is not None:
            module_kwargs["amodel"] = amodel

        self.device = device
        self.model = laion_clap.CLAP_Module(**module_kwargs)
        self._load_clap_checkpoint(self.model, ckpt_path)
        self.model.eval()
        self.model.to(device)

    @staticmethod
    def _load_clap_checkpoint(clap_module, ckpt_path: str):
        """Load laion_clap weights; tolerate transformers version drift (e.g. position_ids)."""
        try:
            clap_module.load_ckpt(ckpt_path)
            return
        except RuntimeError as exc:
            msg = str(exc)
            if "state_dict" not in msg and "Unexpected key" not in msg:
                raise
            print("[ClapEmbedder] load_ckpt failed; retrying with strict=False ...")

        raw = torch.load(ckpt_path, map_location="cpu")
        state = raw
        if isinstance(raw, dict):
            state = raw.get("state_dict", raw.get("model", raw))

        try:
            from laion_clap.hook import load_state_dict

            state = load_state_dict(raw, skip_params=True)
        except Exception:
            pass

        if not isinstance(state, dict):
            raise RuntimeError(f"Unrecognized CLAP checkpoint format: {ckpt_path}")

        state = {k: v for k, v in state.items() if not k.endswith("position_ids")}
        missing, unexpected = clap_module.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[ClapEmbedder] missing keys ({len(missing)}), e.g. {missing[:3]}")
        if unexpected:
            print(f"[ClapEmbedder] ignored unexpected keys: {unexpected}")

    @classmethod
    def _resolve_clap_checkpoint(cls, checkpoint: str):
        """Map config/HF-style ids to a local .pt file + audio encoder variant."""
        env_ckpt = os.environ.get("CLAP_CKPT_PATH")
        if env_ckpt:
            return env_ckpt, "HTSAT-base"

        if os.path.isfile(checkpoint):
            use_htsat = "HTSAT" in checkpoint or "music_audioset" in checkpoint
            return checkpoint, ("HTSAT-base" if use_htsat else None)

        if checkpoint in ("laion/larger_clap_music", "larger_clap_music"):
            return cls.DEFAULT_MUSIC_CKPT, "HTSAT-base"

        # Allow passing the .pt filename only (resolved under checkpoints/clap_weights).
        if checkpoint.endswith(".pt"):
            return os.path.join("checkpoints/clap_weights", checkpoint), "HTSAT-base"

        return cls.DEFAULT_MUSIC_CKPT, "HTSAT-base"

    @staticmethod
    def _to_numpy(embeds):
        if torch.is_tensor(embeds):
            return embeds.detach().cpu().numpy()
        return np.asarray(embeds)

    def _embed_audio(self, audio_paths: list) -> np.ndarray:
        """Call laion_clap across versions (older builds omit `use_tensor`)."""
        try:
            embeds = self.model.get_audio_embedding_from_filelist(
                x=audio_paths, use_tensor=False
            )
        except TypeError:
            embeds = self.model.get_audio_embedding_from_filelist(x=audio_paths)
        return self._to_numpy(embeds)

    def _embed_text(self, texts: list) -> np.ndarray:
        try:
            embeds = self.model.get_text_embedding(texts, use_tensor=False)
        except TypeError:
            embeds = self.model.get_text_embedding(texts)
        return self._to_numpy(embeds)

    @torch.no_grad()
    def embed_audio_files(self, audio_paths: list) -> np.ndarray:
        return self._embed_audio(audio_paths)  # (N, 512)

    @torch.no_grad()
    def embed_text(self, texts: list) -> np.ndarray:
        return self._embed_text(texts)


def train_regressor(
    manifest_paths,
    clap_checkpoint: str,
    out_path: str,
    device: str = "cuda",
    epochs: int = 50,
    lr: float = 1e-3,
):
    """Fit the VARegressor on ground-truth-labeled manifests (DEAM/PMEmo), caching CLAP embeddings."""
    records = []
    for p in manifest_paths:
        with open(p) as f:
            records.extend(json.loads(l) for l in f if l.strip())
    records = [r for r in records if not r.get("is_pseudo_label", False)]
    print(f"Training VA regressor on {len(records)} ground-truth-labeled clips")

    embedder = ClapEmbedder(checkpoint=clap_checkpoint, device=device)
    audio_paths = [r["audio_path"] for r in records]
    targets = np.array([[r["valence"], r["arousal"]] for r in records], dtype=np.float32)

    embeds = []
    batch_size = 16
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Embedding audio with CLAP"):
        embeds.append(embedder.embed_audio_files(audio_paths[i:i + batch_size]))
    embeds = np.concatenate(embeds, axis=0).astype(np.float32)

    x_tr, x_val, y_tr, y_val = train_test_split(embeds, targets, test_size=0.1, random_state=42)

    model = VARegressor(embedding_dim=embeds.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    x_tr_t = torch.from_numpy(x_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    x_val_t = torch.from_numpy(x_val).to(device)
    y_val_t = torch.from_numpy(y_val).to(device)

    best_val = float("inf")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        pred = model(x_tr_t)
        loss = loss_fn(pred, y_tr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"state_dict": model.state_dict(), "embedding_dim": embeds.shape[1]}, out_path)
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")

    print(f"Best val MSE={best_val:.4f}; saved regressor to {out_path}")
    return out_path


def load_regressor(checkpoint_path: str, device: str = "cuda") -> VARegressor:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = VARegressor(embedding_dim=ckpt["embedding_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the CLAP -> (valence, arousal) regressor probe")
    ap.add_argument("--manifests", nargs="+", required=True)
    ap.add_argument("--clap_checkpoint", default="laion/larger_clap_music")
    ap.add_argument("--out", default="checkpoints/va_regressor.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=200)
    args = ap.parse_args()
    train_regressor(args.manifests, args.clap_checkpoint, args.out, device=args.device, epochs=args.epochs)
