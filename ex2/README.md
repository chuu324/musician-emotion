# Emotion-Controllable Personalized BGM Generation
### Continuous Valence–Arousal Emotion Control on MusicGen

This repository implements the system described in the project proposal *"Emotion-Controllable
Personalized BGM Generation — Continuous Emotion Control on MusicGen."* It fine-tunes a frozen,
pretrained MusicGen backbone with a lightweight **emotion-conditioning adapter** that lets a user
specify mood as a continuous 2-D **Valence–Arousal (V-A)** coordinate instead of a discrete label
("happy" / "sad"), together with an optional **emotion-fidelity alignment loss**.

```
text prompt + (valence, arousal) ──▶ Emotion Conditioning Module ──▶ MusicGen (frozen backbone)
                                                                    ──▶ EnCodec decoder ──▶ audio
```

Only the conditioning module (a few million parameters) is trained. Everything else (T5 text
encoder, MusicGen transformer, EnCodec) stays frozen, which is what makes this feasible on a
single consumer/university GPU, matching the proposal's feasibility argument (Section 8).

---

## 1. Repository layout

```
musicgen-emotion/
├── README.md                     <- you are here
├── requirements.txt
├── configs/
│   └── config.yaml               <- all hyperparameters / paths in one place
├── data/
│   ├── prepare_deam.py           <- builds a manifest from the DEAM dataset
│   ├── prepare_pmemo.py          <- builds a manifest from the PMEmo dataset
│   ├── prepare_emopia.py         <- builds a manifest from the EMOPIA dataset (optional extra data)
│   ├── prepare_aimoclips.py      <- builds an EVAL-ONLY manifest from the AImoclips benchmark
│   ├── auto_label.py             <- CLAP + regressor auto-labeling for unlabeled corpora (MTG-Jamendo primarily)
│   ├── merge_manifests.py        <- merges/splits multiple manifests into train/val/test
│   └── dataset.py                <- PyTorch Dataset/DataLoader (audio -> EnCodec tokens, on the fly or cached)
├── model/
│   ├── emotion_adapter.py        <- the core innovation: VA -> FiLM + soft prefix token
│   └── musicgen_emotion.py       <- wraps HF MusicGen, freezes backbone, injects adapter (+ optional LoRA)
├── utils/
│   ├── audio_utils.py            <- audio IO / resampling helpers
│   └── va_regressor.py           <- small CLAP-embedding -> (valence, arousal) probe, used for
│                                     auto-labeling (Sec 4.3) and for the emotion-fidelity metric (Sec 5)
├── train.py                      <- Phase 2/3 training entry point
├── generate.py                   <- inference: text + (V,A) -> wav
├── evaluate.py                   <- FAD, CLAP score, emotion-fidelity (Sec 5 of proposal)
└── scripts/
    └── run_baseline.sh           <- Phase 1: reproduce a stock MusicGen inference baseline
```

## 2. Datasets (Section 4.3 of the proposal)

The proposal needs (a) a modest, high-quality set of music **with continuous V-A annotations** to
train/validate the conditioning module, and (b) a much larger, **unlabeled** corpus that gets
auto-labeled to scale up training. Below is what I'd actually use and why.

### 2.1 Primary annotated data (continuous V-A, dynamic + static)

| Dataset | Size | Annotation | Why use it | Link |
|---|---|---|---|---|
| **DEAM** (MediaEval Database for Emotional Analysis in Music) | 1,802 clips (mostly 45 s) | Continuous valence & arousal, both per-second (2 Hz) and per-clip average, collected via crowdsourcing | The de-facto standard dynamic V-A benchmark; used in almost all recent MER papers; free, non-commercial CC-BY-NC license | http://cvml.unige.ch/databases/DEAM/ |
| **PMEmo / PMEmo2019** | 794 pop songs (with chorus-focused clips) | Continuous dynamic valence & arousal + EDA physiological signal + lyrics | Different genre distribution (contemporary pop, not the older/soundtrack-heavy DEAM), good complement to DEAM | https://github.com/HuiZhangDB/PMEmo |

These two together give ~2,600 emotion-labeled clips spanning multiple genres — enough to validate
that the conditioning module actually moves the generated music's perceived affect in the intended
direction (Phase 2/3 in the proposal timeline), which is the main claim we need to support.

### 2.2 Optional extra labeled data

| Dataset | Size | Annotation | Notes |
|---|---|---|---|
| **EMOPIA** | 1,087 piano clips (~1.5 h) | Discrete emotion quadrant (Russell's circumplex Q1–Q4) + tempo/key | Not continuous, but each quadrant maps to an approximate (V, A) centroid, e.g. Q1=(+,+), Q2=(-,+), Q3=(-,-), Q4=(+,-); useful as *extra weak supervision*, and useful if you want a piano-only BGM subset. `data/prepare_emopia.py` converts quadrant labels to approximate coordinates. |

### 2.3 Large unlabeled corpus (for auto-labeling, Section 4.3 "expand the training set")

| Dataset | Size | Why |
|---|---|---|
| **MTG-Jamendo (mood/theme subset)** | ~18,000 tracks with mood/theme tags (55k total tracks) | **Primary auto-labeling target.** Large, freely licensed (Creative Commons), and — unlike a generic captioned corpus — its mood/theme tags ("happy", "dark", "melancholic", "energetic", ...) are already affect-related, so they double as a cheap sanity check on the auto-labeler's pseudo V-A output (see `--mood_tags_json` in `data/auto_label.py`). Good genre diversity for BGM use cases. | https://github.com/MTG/mtg-jamendo-dataset |

`data/auto_label.py` trains a lightweight regressor (`utils/va_regressor.py`) that maps **CLAP audio
embeddings → (valence, arousal)** using DEAM + PMEmo as ground truth, then applies it to MTG-Jamendo
audio to produce pseudo V-A labels. This is exactly the "emotion-recognition model / CLAP
auto-labeling" step described in the proposal. When MTG-Jamendo's own mood tags are supplied, the
script cross-checks each pseudo-label against a small tag→expected-sign lexicon and drops clips
where the regressor's output strongly disagrees with its own mood tags — a free noise filter that a
generic (non-emotion) corpus wouldn't give us.

> **Why not MusicCaps for training?** MusicCaps (5,521 clips, AudioSet-derived) is the dataset the
> proposal already cites (Sec 2.3) for its detailed *captions*, but its captions describe
> instrumentation/genre/production rather than affect, it carries no emotion labels at all, and the
> audio itself is only distributable as YouTube IDs (increasingly unreliable to fetch at scale). For
> an emotion-focused project, MTG-Jamendo's mood tags are a strictly better fit for the auto-labeling
> step. MusicCaps still has a legitimate, narrower role — see 2.4 below.

### 2.4 Evaluation-only resources

| Dataset | Size | Role |
|---|---|---|
| **AImoclips** (Go et al., 2025 — the same benchmark the proposal cites in Section 1) | 991 AI-generated clips from 6 TTM systems, each rated by ~5+ of 111 human participants on a continuous valence/arousal scale, across 12 emotion-word prompts spanning all 4 V-A quadrants | **Use this as an external emotion-fidelity benchmark**, not training data. It is exactly the kind of ground truth Section 5 needs: generate our own system's output on the same 12 emotion-word/quadrant combinations and compare (a) our CLAP-regressor-predicted V-A against the paper's human-rated V-A for the *same* intents, and (b) whether our system still shows the "neutrality bias" the paper found in every existing TTM system. Open-sourced at https://github.com/HunRotation/HunRotation.github.io (audio samples: https://hunrotation.github.io/projects/aimoclips.html). `data/prepare_aimoclips.py` turns it into an eval manifest. |
| **MusicCaps** | 5,521 clips w/ detailed captions | Optional: only as a generic reference set for FAD's background statistics (a common convention in TTM papers) or a sanity-check on general (non-emotion) CLAP text–audio alignment. Not required — DEAM/PMEmo held-out audio works fine as an FAD reference too, and avoids the YouTube-download dependency entirely. |

> **A note on audio files:** none of these datasets' raw audio is redistributed here — the prepare_*.py
> scripts download official archives / use `youtube-dl`-style retrieval where the dataset only ships
> IDs (e.g. MTG-Jamendo, MusicCaps), and write out a unified `manifest.jsonl` with
> `{"audio_path", "caption", "valence", "arousal", "source", "is_pseudo_label"}` per line, which every
> other script consumes.

## 3. Model design (Section 3 & 4 of the proposal)

### 3.1 Emotion Conditioning Module (`model/emotion_adapter.py`) — the core innovation

Two complementary, lightweight mechanisms, both trainable, both cheap:

1. **FiLM modulation of the T5 text embeddings.** A small MLP maps `(valence, arousal) ∈ [-1,1]²`
   to per-channel scale/shift vectors `(γ, β)` that modulate the frozen T5 encoder's hidden states
   before they enter MusicGen's cross-attention (`h' = γ ⊙ h + β`). This lets the emotion coordinate
   continuously warp the *meaning* of the text prompt rather than just appending information.
2. **Soft emotion-prefix token.** The same MLP (different head) also produces one extra "pseudo
   text token" embedding that is prepended to the encoder sequence, so cross-attention has a
   dedicated slot purely about the emotion coordinate. This is analogous to prefix-tuning and is
   very cheap to add/remove.

Both are pure functions of `(valence, arousal)` — continuous by construction, so interpolating
between two emotion points at inference time gives smoothly interpolated conditioning, satisfying
the "continuous, quantifiable" requirement in the proposal.

### 3.2 Backbone (`model/musicgen_emotion.py`)

* Wraps HF `transformers.MusicgenForConditionalGeneration` (`facebook/musicgen-small` by default;
  swap to `-medium` if compute allows).
* Freezes **everything** (T5 encoder, decoder transformer, EnCodec) except the emotion adapter.
* Optionally (config flag `use_lora: true`) also attaches a small **LoRA** (via `peft`) to the
  decoder's cross-/self-attention projections for a bit more capacity — still cheap, still avoids
  full fine-tuning, matching "adapter or LoRA" in Section 4.2.

### 3.3 Losses

* **Primary — LM loss**: standard MusicGen next-codebook-token cross-entropy against ground-truth
  EnCodec tokens of the training audio (teacher forcing), computed only over the trainable
  parameters' gradient path (backbone frozen).
* **Bonus — emotion-fidelity alignment loss** (Section 3, innovation #2): during training, every
  `align_every_n_steps`, take a soft (Gumbel/softmax-relaxed) reconstruction of a short generated
  clip, decode it with the frozen EnCodec decoder, run it through the frozen CLAP-embedding + VA
  regressor from `utils/va_regressor.py`, and add `MSE(predicted_VA, target_VA)` to the loss. This
  is flagged clearly in the code as an approximate/optional term (Gumbel-softmax through a frozen
  audio codec is inherently noisy) — disable with `use_alignment_loss: false` if it destabilizes
  training, and fall back to only evaluating emotion fidelity post-hoc (Section 5).

## 4. Evaluation (Section 5 of the proposal)

`evaluate.py` computes, for a held-out set of (prompt, target V, target A):

| Dimension | Metric | Implementation |
|---|---|---|
| Audio quality | FAD (Fréchet Audio Distance) | `frechet_audio_distance` package (VGGish/CLAP embeddings) vs. MusicCaps/DEAM reference set |
| Text–music alignment | CLAP score | `laion_clap` cosine similarity between prompt text embedding and generated-audio embedding |
| Emotion fidelity | (generated vs. target V-A) consistency | frozen `utils/va_regressor.py` predicts V-A from generated audio's CLAP embedding, reports MAE/RMSE and Concordance Correlation Coefficient (CCC) vs. target; script also dumps clips for a small-scale subjective listening test |

## 5. Quickstart

```bash
pip install -r requirements.txt

# Phase 1 — reproduce a stock MusicGen baseline
bash scripts/run_baseline.sh

# Phase 2 — build data
python data/prepare_deam.py   --deam_root /path/to/DEAM --out data/manifests/deam.jsonl
python data/prepare_pmemo.py  --pmemo_root /path/to/PMEmo2019 --out data/manifests/pmemo.jsonl
python data/merge_manifests.py --inputs data/manifests/deam.jsonl data/manifests/pmemo.jsonl \
                                --out data/manifests/train_val_test --split 0.8 0.1 0.1

# optional: scale up with auto-labeled data
python data/auto_label.py --unlabeled_audio_dir /path/to/mtg_jamendo/audio \
                           --labeled_manifest data/manifests/train_val_test/train.jsonl \
                           --out data/manifests/mtg_pseudo.jsonl

# Phase 2/3 — train the emotion adapter
python train.py --config configs/config.yaml

# Inference
python generate.py --prompt "gentle acoustic guitar for a rainy afternoon" \
                    --valence 0.7 --arousal -0.3 --out out.wav

# Phase 4 — evaluate
python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best.pt
```

## 6. Mapping back to the proposal's timeline

| Phase | This repo |
|---|---|
| Phase 1 | `scripts/run_baseline.sh` |
| Phase 2 | `data/*.py`, `model/emotion_adapter.py` |
| Phase 3 | `train.py` |
| Phase 4 | `evaluate.py`, `generate.py` |
