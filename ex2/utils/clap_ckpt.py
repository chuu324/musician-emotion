"""Shared laion_clap checkpoint loading; patch for frechet_audio_distance + transformers drift."""
import torch

_PATCHED = False


def _load_state_dict_lenient(clap_module, ckpt_path: str) -> None:
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
        print(f"[clap_ckpt] missing keys ({len(missing)}), e.g. {missing[:3]}")
    if unexpected:
        print(f"[clap_ckpt] ignored unexpected keys: {unexpected}")


def load_clap_checkpoint(clap_module, ckpt_path: str) -> None:
    """Load weights onto a laion_clap.CLAP_Module; tolerate position_ids drift."""
    try:
        clap_module.load_ckpt(ckpt_path)
        return
    except RuntimeError as exc:
        msg = str(exc)
        if "state_dict" not in msg and "Unexpected key" not in msg:
            raise
        print("[clap_ckpt] load_ckpt failed; retrying with strict=False ...")
    _load_state_dict_lenient(clap_module, ckpt_path)


def patch_laion_clap_load_ckpt() -> None:
    """Monkey-patch CLAP_Module.load_ckpt so frechet_audio_distance works on newer transformers."""
    global _PATCHED
    if _PATCHED:
        return

    import laion_clap

    _orig = laion_clap.CLAP_Module.load_ckpt

    def load_ckpt_patched(self, ckpt_path, *args, **kwargs):
        try:
            return _orig(self, ckpt_path, *args, **kwargs)
        except RuntimeError as exc:
            msg = str(exc)
            if "state_dict" not in msg and "Unexpected key" not in msg:
                raise
            print("[clap_ckpt] FAD/CLAP load_ckpt failed; retrying with strict=False ...")
        _load_state_dict_lenient(self, ckpt_path)

    laion_clap.CLAP_Module.load_ckpt = load_ckpt_patched
    _PATCHED = True
