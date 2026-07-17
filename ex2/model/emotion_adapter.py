"""
The core innovation of the proposal (Section 3, innovation #1): a lightweight, continuous
emotion-conditioning module that turns a 2-D Valence-Arousal coordinate into a modulation of
MusicGen's (frozen) T5 text-encoder hidden states.

Two complementary, jointly-trained mechanisms:

  1. FiLM modulation:  h' = gamma(v,a) ⊙ h + beta(v,a)
     applied per-channel to every token of the T5 encoder output. Initialized to the identity
     transform (gamma≈1, beta≈0) so training starts from "behave like vanilla MusicGen" and only
     gradually learns to warp the text representation towards the requested emotion.

  2. Soft emotion-prefix token: one extra pseudo text-token embedding, purely a function of
     (v,a), prepended to the encoder sequence so cross-attention has a dedicated slot for the
     emotion coordinate (mirrors prefix-tuning).

Both are pure functions of (valence, arousal) -> continuous by construction, so interpolating
between two points in V-A space at inference time yields smoothly interpolated conditioning.
"""
import torch
import torch.nn as nn


class _VAEncoder(nn.Module):
    """Small MLP shared by both FiLM and the prefix-token head: (v, a) -> hidden feature."""

    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )

    def forward(self, va: torch.Tensor) -> torch.Tensor:
        return self.net(va)


class EmotionConditioningModule(nn.Module):
    def __init__(
        self,
        text_hidden_size: int = 768,
        mlp_hidden_size: int = 128,
        use_film: bool = True,
        use_prefix_token: bool = True,
        va_range=(-1.0, 1.0),
    ):
        super().__init__()
        self.text_hidden_size = text_hidden_size
        self.use_film = use_film
        self.use_prefix_token = use_prefix_token
        self.va_min, self.va_max = va_range

        self.va_encoder = _VAEncoder(mlp_hidden_size)

        if use_film:
            self.film_head = nn.Linear(mlp_hidden_size, 2 * text_hidden_size)
            # Initialize so the FiLM transform starts as (approximately) the identity function:
            # gamma = 1 + 0*x, beta = 0*x  at the start of training.
            nn.init.zeros_(self.film_head.weight)
            nn.init.zeros_(self.film_head.bias)
            self.film_head.bias.data[:text_hidden_size] = 1.0  # gamma starts at 1

        if use_prefix_token:
            self.prefix_head = nn.Linear(mlp_hidden_size, text_hidden_size)
            nn.init.zeros_(self.prefix_head.weight)
            nn.init.zeros_(self.prefix_head.bias)
            # Starts at 0 → identical to an untrained adapter; grows only as needed.
            self.prefix_scale = nn.Parameter(torch.zeros(1))

    def compute_scaled_prefix(
        self, valence: torch.Tensor, arousal: torch.Tensor
    ) -> torch.Tensor:
        """Return the emotion prefix embedding (B, D), scaled by the learnable gate."""
        if not self.use_prefix_token:
            raise RuntimeError("compute_scaled_prefix called but use_prefix_token is False")
        va = torch.stack([self._clip_va(valence), self._clip_va(arousal)], dim=-1)
        va_feat = self.va_encoder(va)
        return self.prefix_scale * self.prefix_head(va_feat)

    def _clip_va(self, va: torch.Tensor) -> torch.Tensor:
        return va.clamp(self.va_min, self.va_max)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,   # (B, T, D)
        attention_mask: torch.Tensor,           # (B, T)
        valence: torch.Tensor,                  # (B,)
        arousal: torch.Tensor,                  # (B,)
    ):
        """Returns (new_encoder_hidden_states, new_attention_mask)."""
        va = torch.stack([self._clip_va(valence), self._clip_va(arousal)], dim=-1)  # (B, 2)
        va_feat = self.va_encoder(va)  # (B, H)

        h = encoder_hidden_states
        mask = attention_mask

        if self.use_film:
            gamma_beta = self.film_head(va_feat)  # (B, 2D)
            gamma, beta = gamma_beta.chunk(2, dim=-1)  # each (B, D)
            h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)

        if self.use_prefix_token:
            prefix = self.compute_scaled_prefix(valence, arousal).unsqueeze(1)  # (B, 1, D)
            h = torch.cat([prefix, h], dim=1)
            prefix_mask = torch.ones(mask.shape[0], 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([prefix_mask, mask], dim=1)

        return h, mask

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
