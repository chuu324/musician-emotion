"""
generate.py — 情感控制 BGM 生成 CLI
=====================================
三种模式:
  1. prompt 模式（推荐）— V-A → 文本 → MusicGen 原生生成（音质最好）
  2. adapter 模式       — 使用训练好的情感 adapter 模型
  3. dsp 模式           — 两阶段: MusicGen 生成 + 情感 DSP 处理

用法:
  # prompt 模式（音质最好）
  python generate.py --preset happy -o output.wav

  # dsp 模式（两阶段，音质好 + 情感控制）
  python generate.py --preset sad --mode dsp -o output.wav
  python generate.py --text "piano" --valence 0.2 --arousal 0.8 --mode dsp -o tense.wav

  # adapter 模式（训练模型）
  python generate.py --preset happy --mode adapter -o output.wav

  # 批量生成
  python generate.py --batch --mode dsp -o batch_dsp/
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emotion_adapter import build_model, generate_with_emotion
from emotion_prompt import va_to_prompt
from emotion_dsp import emotion_dsp_process


def librosa_load_for_dsp(path, sr, duration=8.0):
    """加载 DSP 处理后的音频返回 tensor (形状 [1, T]，同 MusicGen 输出)。"""
    import librosa
    import numpy as np
    y, _ = librosa.load(path, sr=sr, mono=True)
    audio_tensor = torch.from_numpy(y).unsqueeze(0)  # [1, T]
    # 确保长度匹配 duration
    expected_len = int(sr * duration)
    if audio_tensor.shape[-1] < expected_len:
        pad = expected_len - audio_tensor.shape[-1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, pad))
    elif audio_tensor.shape[-1] > expected_len:
        audio_tensor = audio_tensor[..., :expected_len]
    return audio_tensor, sr


# 预置情感模板
EMOTION_PRESETS = {
    "happy":       (0.85, 0.75, "Happy and energetic music"),
    "sad":         (0.20, 0.25, "Sad melancholic melody"),
    "calm":        (0.80, 0.15, "Calm relaxing peaceful music"),
    "tense":       (0.25, 0.75, "Tense dramatic soundtrack"),
    "angry":       (0.15, 0.85, "Angry aggressive intense music"),
    "romantic":    (0.80, 0.40, "Romantic tender love song"),
    "epic":        (0.40, 0.80, "Epic orchestral cinematic music"),
    "dreamy":      (0.55, 0.30, "Dreamy ethereal ambient soundscape"),
    "dark":        (0.20, 0.45, "Dark mysterious atmosphere"),
    "neutral":     (0.50, 0.50, "Neutral background ambient music"),
    "uplifting":   (0.70, 0.65, "Uplifting cheerful uplifting tune"),
    "boring":      (0.30, 0.15, "Boring monotonous dull drone"),
}


def save_audio(audio, sample_rate: int, output_path: str):
    """保存音频为 WAV 文件，带归一化防止爆音。"""
    import soundfile as sf
    import numpy as np
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    # 归一化到峰值 0.85，留余量防爆音
    peak = np.max(np.abs(audio_np))
    if peak > 0.85:
        audio_np = audio_np * (0.85 / peak)
    sf.write(output_path, audio_np.T, sample_rate, subtype="PCM_16")
    return output_path


def list_presets():
    """打印所有预置情感模板。"""
    print("\n情感预设模板:")
    print(f"{'名称':>12} | Valence | Arousal | 描述")
    print("-" * 60)
    for name, (v, a, desc) in sorted(EMOTION_PRESETS.items()):
        print(f"{name:>12} | {v:>7.2f} | {a:>7.2f} | {desc}")
    print()


def load_checkpoint(checkpoint_path: str, device: str):
    """加载模型和 checkpoint。"""
    import argparse as ap

    model, va_encoder, injector = build_model(device=device, freeze_backbone=True)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        with torch.serialization.safe_globals([ap.Namespace]):
            ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        va_encoder.load_state_dict(ckpt["va_encoder_state_dict"])
        if "injector_state_dict" in ckpt:
            injector.load_state_dict(ckpt["injector_state_dict"])
        elif "adapter_state_dict" in ckpt:
            injector.load_state_dict(ckpt["adapter_state_dict"])  # 兼容旧
        print(f"  Checkpoint: {checkpoint_path}")
    else:
        print("  ⚠️  使用未训练的随机初始化参数")
    va_encoder.eval()
    injector.eval()
    return model, va_encoder, injector


def generate_single(args):
    """生成单个音频。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"模式: {args.mode}")

    # 预设
    if args.preset:
        if args.preset not in EMOTION_PRESETS:
            print(f"❌ 未知预设: {args.preset}")
            list_presets()
            return
        v, a, default_text = EMOTION_PRESETS[args.preset]
        args.valence = v
        args.arousal = a
        if not args.text:
            args.text = default_text

    if args.text is None:
        print("❌ 请指定 --text 或 --preset")
        return

    if args.mode in ("prompt", "dsp"):
        # ======== prompt / dsp 第一阶段：MusicGen 原生生成 ========
        from transformers import MusicgenForConditionalGeneration, AutoProcessor

        print("加载 MusicGen ...")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

        prompt = va_to_prompt(args.valence, args.arousal,
                              base_prompt=args.text,
                              detail_level="medium")
        print(f"生成 prompt: '{prompt}'")

        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
        t0 = time.time()
        audio_values = model.generate(**inputs, do_sample=True,
                                       guidance_scale=args.guidance,
                                       max_new_tokens=int(256 * args.duration / 5))
        audio = audio_values[0].cpu()
        sr = model.config.audio_encoder.sampling_rate
        elapsed = time.time() - t0

        if args.mode == "dsp":
            # ======== dsp 第二阶段：情感 DSP 处理 ========
            print(f"应用情感 DSP: V={args.valence:.2f}, A={args.arousal:.2f}")
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp_path = f.name
            save_audio(audio, sr, tmp_path)
            dsp_t0 = time.time()
            emotion_dsp_process(tmp_path, tmp_path,
                                valence=args.valence,
                                arousal=args.arousal,
                                sr=sr)
            audio, _ = librosa_load_for_dsp(tmp_path, sr=sr, duration=args.duration)
            os.unlink(tmp_path)
            elapsed = time.time() - t0
            print(f"  DSP 耗时: {time.time() - dsp_t0:.1f}s")
    elif args.mode == "adapter":
        # ======== adapter 模式 ========
        print("加载模型...")
        model, va_encoder, injector = load_checkpoint(args.checkpoint, device)

        print(f"生成中: '{args.text}' (V={args.valence:.2f}, A={args.arousal:.2f})")
        t0 = time.time()
        audio, sr = generate_with_emotion(
            model, va_encoder, injector,
            text_prompt=args.text,
            valence=args.valence,
            arousal=args.arousal,
            duration=args.duration,
            guidance_scale=args.guidance,
            device=device,
            generate_scale=args.scale,
        )
        elapsed = time.time() - t0
    else:
        print(f"❌ 未知模式: {args.mode}")
        return

    # 保存
    output_path = args.output
    if not output_path:
        name = f"gen_{args.preset or 'custom'}_v{args.valence:.2f}_a{args.arousal:.2f}.wav"
        output_path = name

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    save_audio(audio, sr, output_path)
    print(f"✅ 已保存: {output_path} ({elapsed:.1f}s)")
    print(f"   时长: {args.duration}s | 采样率: {sr}Hz | 大小: {os.path.getsize(output_path) / 1024:.0f}KB")

    # 播放
    if args.play:
        import subprocess
        try:
            subprocess.Popen(["aplay", output_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   🎵 正在播放...")
        except FileNotFoundError:
            print("   ⚠️  aplay 未找到，无法播放")


def generate_batch(args):
    """批量生成，遍历所有预设情感。"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"模式: {args.mode}")

    output_dir = args.output or "batch_output"
    os.makedirs(output_dir, exist_ok=True)

    if args.mode in ("prompt", "dsp"):
        from transformers import MusicgenForConditionalGeneration, AutoProcessor
        print("加载 MusicGen ...")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").to(device)
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        dsp_mode = (args.mode == "dsp")
    else:
        print("加载模型...")
        model, va_encoder, injector = load_checkpoint(args.checkpoint, device)
        dsp_mode = False

    for name, (v, a, desc) in sorted(EMOTION_PRESETS.items()):
        output_path = os.path.join(output_dir, f"{name}_v{v:.2f}_a{a:.2f}.wav")
        if os.path.exists(output_path) and not args.force:
            print(f"  [{name}] 已存在，跳过")
            continue

        print(f"  [{name}] V={v:.2f}, A={a:.2f} ...", end=" ", flush=True)
        t0 = time.time()
        try:
            if args.mode in ("prompt", "dsp"):
                prompt = va_to_prompt(v, a, base_prompt="", detail_level="medium")
                inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)
                audio_values = model.generate(**inputs, do_sample=True,
                                               guidance_scale=args.guidance,
                                               max_new_tokens=int(256 * args.duration / 5))
                audio = audio_values[0].cpu()
                sr = model.config.audio_encoder.sampling_rate
                if dsp_mode:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        tmp_path = f.name
                    save_audio(audio, sr, tmp_path)
                    emotion_dsp_process(tmp_path, tmp_path, valence=v, arousal=a, sr=sr)
                    audio, _ = librosa_load_for_dsp(tmp_path, sr=sr, duration=args.duration)
                    os.unlink(tmp_path)
            else:
                audio, sr = generate_with_emotion(
                    model, va_encoder, injector,
                    text_prompt=desc, valence=v, arousal=a,
                    duration=args.duration, guidance_scale=args.guidance,
                    device=device, generate_scale=args.scale,
                )
            save_audio(audio, sr, output_path)
            print(f"{time.time() - t0:.1f}s")
        except Exception as e:
            print(f"失败: {e}")

    print(f"\n✅ 批量生成完成! 文件保存在: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="🎵 情感控制 BGM 生成器")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="文本描述 (如 'calm piano')")
    parser.add_argument("--mode", "-m", type=str, default="prompt",
                        choices=["prompt", "dsp", "adapter"],
                        help="生成模式: prompt(默认,音质好) / dsp(两阶段,情感DSP) / adapter(训练模型)")
    parser.add_argument("--preset", "-p", type=str, default=None,
                        choices=list(EMOTION_PRESETS.keys()) + [None],
                        help=f"情感预设 ({', '.join(EMOTION_PRESETS.keys())})")
    parser.add_argument("--valence", "-v", type=float, default=0.5,
                        help="Valence 0~1 (积极/消极)")
    parser.add_argument("--arousal", "-a", type=float, default=0.5,
                        help="Arousal 0~1 (兴奋/平静)")
    parser.add_argument("--duration", "-d", type=float, default=8.0,
                        help="生成时长 (秒)")
    parser.add_argument("--scale", "-s", type=float, default=0.3,
                        help="调制幅度 (0~1, 越小音质越好但控制越弱)")
    parser.add_argument("--guidance", "-g", type=float, default=3.0,
                        help="guidance scale (越高越贴合文本)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--checkpoint", "-c", type=str,
                        default="checkpoints/adapter_v3.pth",
                        help="checkpoint 路径")
    parser.add_argument("--batch", "-b", action="store_true",
                        help="批量生成所有预设情感")
    parser.add_argument("--list", "-l", action="store_true",
                        help="列出所有情感预设")
    parser.add_argument("--play", action="store_true",
                        help="生成后播放 (Linux aplay)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="强制覆盖已存在的文件")

    args = parser.parse_args()

    if args.list:
        list_presets()
        return

    if args.batch:
        generate_batch(args)
    else:
        generate_single(args)


if __name__ == "__main__":
    main()
