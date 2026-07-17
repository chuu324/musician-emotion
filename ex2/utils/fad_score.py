"""
CLAP-based Fréchet Audio Distance using local laion_clap weights (no HF download for FAD).

Reuses ClapEmbedder from va_regressor.py — same path as evaluate.py / CCC metrics.

With N=50 clips and 512-d CLAP embeddings, raw covariance FAD can be negative (ill-conditioned).
We PCA-reduce (fit on reference audio) before Fréchet distance — same idea as
frechet_audio_distance use_pca=True — yielding stable non-negative scores.
"""
import os
from typing import List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA

from utils.va_regressor import ClapEmbedder


def list_audio_files(directory: str) -> List[str]:
    exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
    paths = []
    for name in sorted(os.listdir(directory)):
        if name.lower().endswith(exts):
            paths.append(os.path.join(directory, name))
    return paths


def _frechet_distance_gaussians(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    """Fréchet distance between two multivariate Gaussians (standard FAD formula)."""
    try:
        from scipy.linalg import sqrtm
    except ImportError:
        raise SystemExit("pip install scipy")

    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)

    diff = mu1 - mu2
    offset = np.eye(sigma1.shape[0]) * eps
    covmean, _ = sqrtm((sigma1 + offset).dot(sigma2 + offset), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fad = float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean))
    return max(0.0, fad)


def _pca_reduce(
    ref_embeds: np.ndarray,
    gen_embeds: np.ndarray,
    n_components: Optional[int] = None,
    max_components: int = 64,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Fit PCA on reference embeddings; transform both sets (FID/FAD convention)."""
    n_ref, dim = ref_embeds.shape
    n_gen = gen_embeds.shape[0]
    if n_components is None:
        n_components = min(n_ref - 1, n_gen - 1, dim, max_components)
    n_components = max(2, int(n_components))

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(ref_embeds)
    return pca.transform(ref_embeds), pca.transform(gen_embeds), n_components


def embed_directory(
    embedder: ClapEmbedder,
    directory: str,
    batch_size: int = 8,
) -> np.ndarray:
    paths = list_audio_files(directory)
    if not paths:
        raise ValueError(f"No audio files in {directory}")
    chunks = []
    for i in range(0, len(paths), batch_size):
        chunks.append(embedder.embed_audio_files(paths[i : i + batch_size]))
    return np.concatenate(chunks, axis=0).astype(np.float64)


def compute_fad_score(
    reference_dir: str,
    generated_dir: str,
    clap_checkpoint: str = "laion/larger_clap_music",
    device: str = "cuda",
    batch_size: int = 8,
    eps: float = 1e-6,
    embedder: Optional[ClapEmbedder] = None,
    use_pca: bool = True,
    pca_components: Optional[int] = None,
    max_pca_components: int = 64,
) -> float:
    """
    FAD between reference (real) and generated audio folders in CLAP embedding space.
    """
    if embedder is None:
        embedder = ClapEmbedder(checkpoint=clap_checkpoint, device=device)

    ref_embeds = embed_directory(embedder, reference_dir, batch_size=batch_size)
    gen_embeds = embed_directory(embedder, generated_dir, batch_size=batch_size)

    if use_pca:
        ref_embeds, gen_embeds, _ = _pca_reduce(
            ref_embeds, gen_embeds, n_components=pca_components, max_components=max_pca_components
        )

    mu_ref, mu_gen = ref_embeds.mean(axis=0), gen_embeds.mean(axis=0)
    sigma_ref = np.cov(ref_embeds, rowvar=False)
    sigma_gen = np.cov(gen_embeds, rowvar=False)

    return _frechet_distance_gaussians(mu_ref, sigma_ref, mu_gen, sigma_gen, eps=eps)


def compute_fad_with_meta(
    reference_dir: str,
    generated_dir: str,
    clap_checkpoint: str = "laion/larger_clap_music",
    device: str = "cuda",
    batch_size: int = 8,
    embedder: Optional[ClapEmbedder] = None,
    use_pca: bool = True,
    pca_components: Optional[int] = None,
    max_pca_components: int = 64,
) -> dict:
    """Like compute_fad_score but returns fad + metadata for results.json."""
    if embedder is None:
        embedder = ClapEmbedder(checkpoint=clap_checkpoint, device=device)

    ref_embeds = embed_directory(embedder, reference_dir, batch_size=batch_size)
    gen_embeds = embed_directory(embedder, generated_dir, batch_size=batch_size)
    n_pca = None

    if use_pca:
        ref_embeds, gen_embeds, n_pca = _pca_reduce(
            ref_embeds, gen_embeds, n_components=pca_components, max_components=max_pca_components
        )

    mu_ref, mu_gen = ref_embeds.mean(axis=0), gen_embeds.mean(axis=0)
    sigma_ref = np.cov(ref_embeds, rowvar=False)
    sigma_gen = np.cov(gen_embeds, rowvar=False)
    fad = _frechet_distance_gaussians(mu_ref, sigma_ref, mu_gen, sigma_gen)

    return {
        "fad": fad,
        "fad_method": "clap_embedder_local_pca" if use_pca else "clap_embedder_local",
        "fad_pca_components": n_pca,
        "fad_n_reference": len(list_audio_files(reference_dir)),
        "fad_n_generated": len(list_audio_files(generated_dir)),
    }
