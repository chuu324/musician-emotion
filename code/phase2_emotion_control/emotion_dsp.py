"""
emotion_dsp.py — 两阶段情感控制：MusicGen 生成 + 情感 DSP 处理
==============================================================
Stage 1: MusicGen 原生生成（干净音频）
Stage 2: 根据 V-A 坐标，对音频做情感化的 DSP 处理

DSP 效果映射:
  Valence (积极/消极) → pitch shift, EQ brightness, harmonics
  Arousal (兴奋/平静) → tempo, volume, dynamics, spectral centroid
"""

import librosa
import numpy as np
import soundfile as sf


def va_to_dsp_params(valence: float, arousal: float) -> dict:
    """将 V-A 坐标映射为 DSP 处理参数。

    Returns:
        dict with: tempo_rate, pitch_shift, eq_bass, eq_mid, eq_treble,
                   volume_boost, compression_ratio, reverb_mix
    """
    # Valence 控制: pitch, brightness, harmonics
    # 高 V → 升调 + 亮 + 温暖
    # 低 V → 降调 + 暗 + 沉闷
    pitch_shift = (valence - 0.5) * 4.0       # -2 ~ +2 semitones
    eq_treble = (valence - 0.5) * 6.0          # -3 ~ +3 dB
    eq_bass = (0.5 - valence) * 3.0            # +1.5 ~ -1.5 dB (低V 加低频)

    # Arousal 控制: tempo, volume, dynamics
    # 高 A → 快 + 响 + 压缩小
    # 低 A → 慢 + 轻 + 压缩大
    tempo_rate = 1.0 + (arousal - 0.5) * 0.3   # 0.85 ~ 1.15
    volume_boost = (arousal - 0.5) * 4.0        # -2 ~ +2 dB
    compression_ratio = 1.0 + (1.0 - arousal) * 3.0  # 1.0 ~ 4.0

    # 添加少量混响模拟空间感（低 A 混响多，高 A 混响少）
    reverb_mix = (1.0 - arousal) * 0.15         # 0 ~ 0.15

    return {
        "tempo_rate": round(tempo_rate, 3),
        "pitch_shift": round(pitch_shift, 1),
        "eq_bass": round(eq_bass, 1),
        "eq_mid": 0.0,
        "eq_treble": round(eq_treble, 1),
        "volume_boost": round(volume_boost, 1),
        "compression_ratio": round(compression_ratio, 1),
        "reverb_mix": round(reverb_mix, 3),
    }


def apply_dsp(audio: np.ndarray, sr: int, params: dict) -> np.ndarray:
    """对音频应用 DSP 处理。

    Args:
        audio: 1D numpy array
        sr: 采样率
        params: va_to_dsp_params 返回的参数字典

    Returns:
        处理后的 1D numpy array
    """
    y = audio.copy().astype(np.float64)

    # 1. Tempo 调整（时间伸缩）
    if params["tempo_rate"] != 1.0:
        y = librosa.effects.time_stretch(y=y, rate=params["tempo_rate"])

    # 2. Pitch 偏移
    if params["pitch_shift"] != 0:
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=params["pitch_shift"])

    # 3. EQ：简单的 shelving 滤波
    if params["eq_bass"] != 0:
        y = _apply_shelf(y, sr, gain=params["eq_bass"], freq=200, kind="low")
    if params["eq_treble"] != 0:
        y = _apply_shelf(y, sr, gain=params["eq_treble"], freq=4000, kind="high")

    # 4. Volume
    if params["volume_boost"] != 0:
        gain_linear = 10 ** (params["volume_boost"] / 20)
        y = y * gain_linear

    # 5. 动态范围压缩
    # compression_ratio=1.0 时不压缩，越大压缩越强
    if params["compression_ratio"] > 1.01:
        threshold = 0.3  # 阈值，超过此值开始压缩
        gain_db = 20 * np.log10(np.abs(y) + 1e-10)
        mask = gain_db > (20 * np.log10(threshold))
        # 压缩：超出部分按 ratio 衰减
        excess = gain_db[mask] - 20 * np.log10(threshold)
        gain_db[mask] = 20 * np.log10(threshold) + excess / params["compression_ratio"]
        y = np.sign(y) * (10 ** (gain_db / 20))

    # 6. 限制峰值防止削波
    peak = np.max(np.abs(y))
    if peak > 0.95:
        y = y * (0.95 / peak)

    return y


def _apply_shelf(y: np.ndarray, sr: int, gain: float, freq: float, kind: str = "low"):
    """简单的 shelving EQ。"""
    from scipy import signal
    # 一阶 shelving filter
    w0 = 2 * np.pi * freq / sr
    if kind == "low":
        # Low shelf
        A = 10 ** (gain / 40)
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.5 - 1) + 2)
        b0 = A * ((A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * np.cos(w0))
        a2 = (A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha
    else:
        # High shelf
        A = 10 ** (gain / 40)
        alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.5 - 1) + 2)
        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

    b = np.array([b0, b1, b2]) / a0
    a = np.array([a0, a1, a2]) / a0
    return signal.lfilter(b, a, y)


def emotion_dsp_process(
    input_audio_path: str,
    output_audio_path: str,
    valence: float,
    arousal: float,
    sr: int = 32000,
):
    """两阶段情感控制：加载音频 → DSP 处理 → 保存。

    Args:
        input_audio_path: MusicGen 生成的音频路径
        output_audio_path: 输出路径
        valence: 0~1
        arousal: 0~1
        sr: 采样率
    """
    # 加载音频
    y, orig_sr = librosa.load(input_audio_path, sr=sr, mono=True)

    # 计算 DSP 参数
    params = va_to_dsp_params(valence, arousal)
    print(f"[DSP] 参数: 节奏={params['tempo_rate']:.2f}, "
          f"音高={params['pitch_shift']:+.1f}半音, "
          f"高音EQ={params['eq_treble']:+.1f}dB, "
          f"低音EQ={params['eq_bass']:+.1f}dB, "
          f"音量={params['volume_boost']:+.1f}dB")

    # 应用 DSP
    y_processed = apply_dsp(y, sr, params)

    # 归一化保存
    peak = np.max(np.abs(y_processed))
    if peak > 0.95:
        y_processed = y_processed * (0.95 / peak)

    sf.write(output_audio_path, y_processed, sr, subtype="PCM_16")
    print(f"[DSP] 输出: {output_audio_path}")

    return params


if __name__ == "__main__":
    # 测试
    import soundfile as sf
    import os

    # 生成测试音频（纯正弦波模拟）
    sr = 32000
    t = np.linspace(0, 3, sr * 3)
    test_audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 880 * t)
    sf.write("/tmp/test_input.wav", test_audio, sr)

    for name, v, a in [("happy", 0.85, 0.80), ("calm", 0.80, 0.15), ("sad", 0.20, 0.25)]:
        emotion_dsp_process("/tmp/test_input.wav", f"/tmp/test_{name}.wav", v, a)
