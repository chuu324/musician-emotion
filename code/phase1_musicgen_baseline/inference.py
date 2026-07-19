"""
Phase 1 — MusicGen 推理基线复现
================================
目标：加载预训练 MusicGen 模型，用文本 prompt 生成音频并保存。

支持两种后端：
  1. HuggingFace Transformers（推荐，开箱即用）
  2. AudioCraft（Meta 官方，需额外安装）

用法：
  python inference.py --prompt "calm piano melody" --output output.wav
  python inference.py --prompt_file prompts.txt --output_dir outputs/
"""

import argparse
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 后端 1：HuggingFace Transformers（默认，推荐）
# ---------------------------------------------------------------------------
def generate_with_transformers(prompt: str, duration: float = 8.0,
                               model_size: str = "small") -> tuple:
    """
    使用 HuggingFace Transformers 加载 MusicGen 生成音频。

    返回: (audio_tensor: FloatTensor[1, channels, samples], sample_rate: int)
    """
    import torch
    from transformers import MusicgenForConditionalGeneration, AutoProcessor

    model_name = f"facebook/musicgen-{model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[HF] 加载 {model_name} 到 {device} ...")

    model = MusicgenForConditionalGeneration.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    # 编码 prompt
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    ).to(device)

    # 生成音频
    print(f"[HF] 生成中 ... (prompt: '{prompt}', duration: {duration}s)")
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=3.0,
        max_new_tokens=int(256 * duration / 5),  # 约 256 tokens / 5s
    )

    audio = audio_values[0].cpu()          # [channels, samples]
    sampling_rate = model.config.audio_encoder.sampling_rate
    return audio, sampling_rate


# ---------------------------------------------------------------------------
# 后端 2：AudioCraft（Meta 官方）
# ---------------------------------------------------------------------------
def generate_with_audiocraft(prompt: str, duration: float = 8.0,
                             model_size: str = "small") -> tuple:
    """
    使用 AudioCraft 的 MusicGen API 生成音频。

    返回: (audio_tensor: FloatTensor[1, channels, samples], sample_rate: int)
    """
    import torch
    from audiocraft.models import MusicGen

    model_name = f"facebook/musicgen-{model_size}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AudioCraft] 加载 {model_name} 到 {device} ...")

    model = MusicGen.get_pretrained(model_name, device=device)
    model.set_generation_params(duration=duration)

    print(f"[AudioCraft] 生成中 ... (prompt: '{prompt}')")
    wav = model.generate([prompt], progress=True)  # [B, C, T]

    audio = wav[0].cpu()
    sampling_rate = model.sample_rate
    return audio, sampling_rate


# ---------------------------------------------------------------------------
# 保存音频（44.1kHz 16-bit 立体声 PCM WAV，最大兼容性）
# ---------------------------------------------------------------------------
def save_audio(audio, sample_rate: int, output_path: str, normalize: bool = True):
    """保存音频为 44.1kHz 16-bit 立体声 PCM WAV，确保所有播放器兼容。

    Args:
        audio: FloatTensor [channels, samples] 或 [samples]
        sample_rate: 原始采样率
        output_path: 输出路径
        normalize: 是否做归一化提升音量
    """
    import numpy as np
    from scipy import signal
    import soundfile as sf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Tensor → numpy
    if hasattr(audio, 'detach'):
        audio = audio.detach().cpu()
    if hasattr(audio, 'numpy'):
        audio_np = audio.numpy().astype(np.float64)
    else:
        audio_np = np.asarray(audio, dtype=np.float64)

    # shape: [channels, samples] → [samples, channels]
    is_mono = False
    if audio_np.ndim == 2:
        audio_np = audio_np.T  # 现在 [samples, channels]
    else:
        is_mono = True
        audio_np = audio_np.reshape(-1, 1)  # [samples, 1]

    # 重采样到 44100 Hz（标准 CD 采样率，兼容性最好）
    target_sr = 44100
    if sample_rate != target_sr:
        num_samples = int(len(audio_np) * target_sr / sample_rate)
        audio_np = signal.resample(
            audio_np, num_samples, axis=0
        ).astype(np.float64)
        sample_rate = target_sr

    # 轻度归一化提升音量
    if normalize:
        peak = np.max(np.abs(audio_np))
        if peak > 1e-8:
            audio_np = audio_np * (0.85 / peak)  # 峰值放大到 0.85
        rms = np.sqrt(np.mean(audio_np ** 2))
        if 1e-8 < rms < 0.12:
            gain = min(0.12 / rms, 4.0)
            audio_np = audio_np * gain
            peak = np.max(np.abs(audio_np))
            if peak > 0.95:
                audio_np = audio_np * (0.95 / peak)

    # 转立体声：单声道 → 复制到双声道
    if is_mono or audio_np.shape[1] == 1:
        audio_np = np.repeat(audio_np, 2, axis=1)

    # 转 16-bit PCM
    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(str(output_path), audio_np, sample_rate, subtype="PCM_16")
    print(f"[保存] → {output_path}  ({sample_rate}Hz, 16-bit 立体声 PCM)")


# ---------------------------------------------------------------------------
# 批量处理
# ---------------------------------------------------------------------------
def batch_generate(prompts: list[str], output_dir: str, backend: str,
                   duration: float, model_size: str):
    """批量生成多个 prompt。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generate_fn = (
        generate_with_transformers if backend == "hf"
        else generate_with_audiocraft
    )

    for i, prompt in enumerate(prompts):
        prompt = prompt.strip()
        if not prompt:
            continue

        print(f"\n{'='*60}")
        print(f"样本 {i+1}/{len(prompts)}: '{prompt}'")
        print(f"{'='*60}")

        try:
            audio, sr = generate_fn(prompt, duration, model_size)
            output_path = output_dir / f"sample_{i+1:03d}.wav"
            save_audio(audio, sr, str(output_path))
        except Exception as e:
            print(f"[错误] 生成失败: {e}")


# ---------------------------------------------------------------------------
# 演示 prompt
# ---------------------------------------------------------------------------
DEMO_PROMPTS = [
    "calm piano melody with soft strings",
    "upbeat electronic dance music with strong bass",
    "sad ambient soundscape with slow pads",
    "happy acoustic guitar folk song",
    "tense orchestral movie soundtrack",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="MusicGen Phase 1 — 推理基线复现"
    )
    parser.add_argument("--prompt", type=str, default=None,
                        help="单个文本 prompt")
    parser.add_argument("--prompt_file", type=str, default=None,
                        help="包含 prompt 的文本文件（每行一个）")
    parser.add_argument("--output", type=str, default="output.wav",
                        help="输出音频路径（单个 prompt 时使用）")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="批量输出目录")
    parser.add_argument("--duration", type=float, default=8.0,
                        help="生成音频时长（秒）")
    parser.add_argument("--backend", type=str, default="hf",
                        choices=["hf", "audiocraft"],
                        help="推理后端：hf (Transformers) / audiocraft")
    parser.add_argument("--model", type=str, default="small",
                        choices=["small", "medium", "large"],
                        help="MusicGen 模型大小")
    parser.add_argument("--demo", action="store_true",
                        help="运行内置 demo prompt")

    args = parser.parse_args()

    # ── 运行 demo ──
    if args.demo:
        print("=" * 60)
        print("MusicGen Phase 1 — 推理基线复现 (Demo)")
        print("=" * 60)
        batch_generate(DEMO_PROMPTS, args.output_dir, args.backend, args.duration, args.model)
        return

    # ── 读取 prompt ──
    prompts = []
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompts = [line.strip() for line in f if line.strip()]
    elif args.prompt:
        prompts = [args.prompt]
    else:
        parser.print_help()
        print("\n[提示] 请提供 --prompt / --prompt_file / --demo")
        return

    # ── 生成 ──
    if len(prompts) == 1:
        generate_fn = (
            generate_with_transformers if args.backend == "hf"
            else generate_with_audiocraft
        )
        print(f"[MusicGen] prompt: '{prompts[0]}' | backend: {args.backend} | model: {args.model}")
        audio, sr = generate_fn(prompts[0], args.duration, args.model)
        save_audio(audio, sr, args.output)
    else:
        batch_generate(prompts, args.output_dir, args.backend, args.duration, args.model)


if __name__ == "__main__":
    main()
