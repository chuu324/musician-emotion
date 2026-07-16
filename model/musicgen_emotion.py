"""
Wraps HF `transformers.MusicgenForConditionalGeneration`:
  - freezes the entire pretrained backbone (T5 text encoder, decoder transformer, EnCodec),
  - injects `EmotionConditioningModule` between the text encoder and the decoder,
  - optionally attaches LoRA adapters (via peft) on the decoder's attention projections for a
    bit of extra trainable capacity, still far cheaper than full fine-tuning.

NOTE on HF internals: this uses the public `encoder_outputs=` argument of
`MusicgenForConditionalGeneration.forward` (the same mechanism T5/BART-style seq2seq models use)
to inject our modified hidden states without re-implementing the rest of the forward pass. This
keeps the wrapper robust to minor internal changes in `transformers` across versions, but you
should double check tensor shapes against your installed `transformers` version the first time
you run this (`pip show transformers`), since MusicGen's training-time codebook/labels reshaping
has changed slightly between releases.
"""
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

from model.emotion_adapter import EmotionConditioningModule


class MusicGenEmotion(nn.Module):
    def __init__(
        self,
        backbone: str = "facebook/musicgen-small",
        freeze_backbone: bool = True,
        use_lora: bool = False,
        lora_config: Optional[dict] = None,
        adapter_config: Optional[dict] = None,
    ):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(backbone)
        self.musicgen = MusicgenForConditionalGeneration.from_pretrained(backbone)

        # HF musicgen-small config may omit decoder_start_token_id; training with labels needs it.
        dec_cfg = self.musicgen.config.decoder
        if dec_cfg.decoder_start_token_id is None:
            dec_cfg.decoder_start_token_id = dec_cfg.bos_token_id

        text_hidden_size = self.musicgen.config.text_encoder.hidden_size
        adapter_config = adapter_config or {}
        self.adapter = EmotionConditioningModule(
            text_hidden_size=text_hidden_size,
            mlp_hidden_size=adapter_config.get("mlp_hidden_size", 128),
            use_film=adapter_config.get("use_film", True),
            use_prefix_token=adapter_config.get("use_prefix_token", True),
            va_range=tuple(adapter_config.get("va_range", (-1.0, 1.0))),
        )

        if freeze_backbone:
            for p in self.musicgen.parameters():
                p.requires_grad_(False)

        if use_lora:
            self._attach_lora(lora_config or {})

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.parameters())
        print(f"[MusicGenEmotion] trainable params: {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / n_total:.3f}%)")

    def _attach_lora(self, lora_config: dict):
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=lora_config.get("r", 8),
            lora_alpha=lora_config.get("alpha", 16),
            lora_dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "out_proj"]
            ),
            bias="none",
        )
        # Only wrap the decoder (the autoregressive transformer) with LoRA; the text encoder and
        # EnCodec stay fully frozen since we only want to adapt how the model *generates*, not
        # how it *understands* text or encodes audio.
        self.musicgen.decoder = get_peft_model(self.musicgen.decoder, cfg)

    def encode_text_with_emotion(self, texts, valence: torch.Tensor, arousal: torch.Tensor, device):
        """Run the frozen T5 text encoder, then apply the emotion adapter. Returns a
        `BaseModelOutput` suitable for passing as `encoder_outputs=` to `self.musicgen(...)`."""
        tok = self.processor.tokenizer(texts, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():  # text encoder is frozen -> no need to track grads through it
            enc_out = self.musicgen.text_encoder(
                input_ids=tok.input_ids, attention_mask=tok.attention_mask, return_dict=True
            )
        hidden_states, attention_mask = self.adapter(
            enc_out.last_hidden_state, tok.attention_mask, valence, arousal
        )
        return BaseModelOutput(last_hidden_state=hidden_states), attention_mask

    def forward(self, texts, valence, arousal, labels=None, decoder_input_ids=None, **kwargs):
        device = next(self.musicgen.parameters()).device
        valence, arousal = valence.to(device), arousal.to(device)
        encoder_outputs, attention_mask = self.encode_text_with_emotion(texts, valence, arousal, device)

        outputs = self.musicgen(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )
        return outputs

    @torch.no_grad()
    def generate(
        self,
        text: str,
        valence: float,
        arousal: float,
        duration_seconds: float = 10.0,
        guidance_scale: float = 3.0,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate a single clip. Returns a (1, num_samples) waveform tensor at the model's
        native sample rate (`self.musicgen.config.audio_encoder.sampling_rate`, 32kHz by default)."""
        self.eval()
        valence_t = torch.tensor([valence], dtype=torch.float32, device=device)
        arousal_t = torch.tensor([arousal], dtype=torch.float32, device=device)
        # Keep original-batch input_ids so HF initializes the decoder at batch=1.
        # (If omitted, batch_size is inferred from the CFG-doubled encoder and the
        # decoder gets doubled a second time → shape mismatch.)
        tok = self.processor.tokenizer([text], padding=True, return_tensors="pt").to(device)
        encoder_outputs, attention_mask = self.encode_text_with_emotion([text], valence_t, arousal_t, device)

        # Classifier-free guidance expects [cond; null] along the batch dim.
        # HF does this in `_prepare_text_encoder_kwargs_for_generation`, but that path is
        # skipped when we already pass `encoder_outputs=`.
        if guidance_scale is not None and guidance_scale > 1:
            h = encoder_outputs.last_hidden_state
            encoder_outputs = BaseModelOutput(
                last_hidden_state=torch.cat([h, torch.zeros_like(h)], dim=0)
            )
            attention_mask = torch.cat(
                [attention_mask, torch.zeros_like(attention_mask)], dim=0
            )

        frame_rate = self.musicgen.config.audio_encoder.frame_rate
        max_new_tokens = int(duration_seconds * frame_rate)

        audio_values = self.musicgen.generate(
            input_ids=tok.input_ids,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            guidance_scale=guidance_scale,
        )
        return audio_values  # (1, 1, num_samples) or (1, num_samples) depending on transformers version

    def encode_audio_to_codes(self, waveforms: torch.Tensor):
        """Encode ground-truth waveforms to EnCodec discrete codes for teacher-forced training.

        Args:
            waveforms: (B, num_samples) float tensor at the backbone's native sample rate.
        Returns:
            audio_codes: (B, num_codebooks, seq_len) long tensor, suitable for building `labels`.
        """
        device = next(self.musicgen.parameters()).device
        waveforms = waveforms.to(device).unsqueeze(1)  # audio_encoder expects (B, 1, num_samples)
        with torch.no_grad():  # EnCodec is frozen
            encoder_outputs = self.musicgen.audio_encoder.encode(waveforms)
            # `encoder_outputs.audio_codes` shape can be (1, B, num_codebooks, seq_len) in some
            # transformers versions (leading dim = number of RVQ "segments"); squeeze if present.
            audio_codes = encoder_outputs.audio_codes
            if audio_codes.dim() == 4:
                audio_codes = audio_codes.squeeze(0)
        return audio_codes

    def build_labels_from_codes(self, audio_codes: torch.Tensor) -> torch.Tensor:
        """Build `labels` for MusicgenForConditionalGeneration.

        HF transformers >= 4.56 expects shape (B, seq_len, num_codebooks). `audio_codes` from
        `encode_audio_to_codes` is (B, num_codebooks, seq_len), so we transpose the last two dims.
        """
        return audio_codes.transpose(1, 2)
