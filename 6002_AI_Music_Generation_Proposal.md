# AI Music Generation — Project Proposal 

Emotion-Controllable Personalized BGM Generation — Continuous Emotion Control 
on MusicGen 
 
Abstract 
This project designs and implements an end-to-end AI music generation system that produces 
personalized background music (BGM) from a user's emotional intent. The system fine-tunes the 
pretrained MusicGen model rather than training from scratch, keeping it feasible within limited 
compute and time. Its core innovation is continuous emotion control, addressing a common 
weakness of current text-to-music (TTM) models—output biased toward emotional neutrality with 
weak controllability—so that abstract affective intent is mapped into stylistically coherent music. 
1. Background & Motivation 
Recent models such as MusicGen, MusicLM, and AudioLDM can already generate high-quality 
music from natural-language descriptions. However, emotional controllability remains a clear 
weakness: a recent benchmark (AImoclips, 2025) finds that current TTM systems are generally 
biased toward emotional neutrality and struggle to convey a specified emotion accurately and 
adjustably. 
This is exactly what our topic—“turning emotion into bespoke BGM”—calls for. Rather than 
spreading thin across emotion, style, and structure, this project focuses on the emotion dimension 
in depth: letting users specify emotion in a continuous, quantifiable way and making the generated 
music faithfully reflect it. 
2. Preliminary Literature Review 
2.1 Audio Discretization & Coding (EnCodec / RVQ) 
EnCodec is a convolutional auto-encoder that compresses continuous audio waveforms into 
discrete tokens via Residual Vector Quantization (RVQ): multiple codebooks encode the signal 
stage by stage, with each quantizer encoding the residual error of the previous one (8 codebooks 
in MusicGen). This turns audio into a text-like discrete sequence, enabling language-model-style 
modeling. 
2.2 Autoregressive Generation (MusicGen) 
MusicGen uses a single-stage Transformer to autoregressively model EnCodec tokens, predicting 
multiple codebooks in parallel via a delay/interleaving pattern. Its text conditioning is provided by 
a T5 text encoder, without relying on diffusion. 
2.3 Conditioning & Evaluation (CLAP / MusicCaps) 
CLAP maps text and audio into a shared semantic space via contrastive learning; here it is used 
mainly for evaluation (CLAP score, measuring text–music alignment) and automatic emotion 
labeling. MusicCaps is a music dataset with detailed text captions, often used as an evaluation 
benchmark.

---

## Page 2

2.4 Unified Framework (AudioCraft) 
Meta's open-source AudioCraft provides unified implementations of MusicGen and EnCodec, 
serving as the engineering basis for our reproduction and extension. 
3. Objectives & Innovations 
Overall objective: Build an end-to-end, self-contained system that generates stylistically coherent, 
emotionally accurate personalized BGM from continuous emotion coordinates. 
Innovations (deliberately focused, not scattered): 
1. Continuous emotion control (core). Replace discrete labels (“happy/sad”) with a 2-D 
continuous Valence-Arousal coordinate for smooth, continuous emotion adjustment. We 
freeze the MusicGen backbone and train only a lightweight emotion-conditioning module 
(adapter), which is compute-friendly. 
2. Emotion-fidelity alignment (bonus). Introduce a “generated emotion → target emotion” 
alignment loss during training, directly targeting the neutrality-bias problem. 
4. Technical Approach 
4.1 System Pipeline 
Text description + Emotion coordinate (V-A)  →  conditioning-fusion module  →  MusicGen 
Transformer (frozen backbone)  →  EnCodec decoder  →  output audio 
4.2 Training Strategy 
• Freeze the pretrained backbone and train only the emotion-conditioning module (adapter or 
LoRA), greatly reducing compute and training time. 
• Validate in stages: first verify controllable emotion conditioning on a small dataset, then 
scale up the data and tune. 
4.3 Data 
• Primarily music data with emotion annotations; 
• Where annotations are scarce, use an emotion-recognition model / CLAP to auto-label 
unlabeled music with V-A values to expand the training set (data feasibility is a later focus). 
5. Evaluation 
Dimension 
Metric / Method 
Audio quality 
FAD (Fréchet Audio Distance, lower is better) 
Text–music alignment 
CLAP score 
Emotion fidelity 
Emotion-recognition model measuring “generated vs. target 
emotion” consistency + small-scale subjective listening test 
 
 
 
 
 


---

## Page 3

6. Timeline 
Phase 
Main Work 
Phase 1 
Literature review; reproduce a MusicGen inference baseline on AudioCraft 
Phase 2 
Data preparation & emotion labeling; implement the continuous emotion-
conditioning module 
Phase 3 
Fine-tuning & hyperparameter tuning; validate emotion controllability 
Phase 4 
Evaluation & comparison experiments; system integration, demo, and report 
 
7. Potential Extensions 
To be attempted only after the core goals are met and if time permits. 
• Melodic-cue conditioning (primary extension). Support user-provided melodic cues 
(humming / MIDI), generating BGM with a specified emotion while preserving the melodic 
contour—corresponding to the “melodic cues” in the topic. This requires an additional 
melody-alignment mechanism in the conditioning module and is more complex to 
implement, hence listed as an extension. 
• Multimodal emotion conditioning. Infer emotion automatically from an image or short 
video, then generate matching BGM. 
8. Feasibility 
This project fine-tunes an open-source pretrained model rather than training from scratch; 
with a frozen backbone and only a lightweight conditioning module to train, both compute and 
timeline are manageable. The literature review and baseline reproduction have a clear path, the 
innovations are focused and can be validated in stages, and the risks are controllable. 
