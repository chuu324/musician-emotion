"""
data_prep.py — 数据预处理与自动情感标注
=========================================
功能：
  1. 下载/加载带 V-A 标注的音乐数据集（DEAM, EmoMusic）
  2. 用 CLAP 给无标签音乐自动标注 V-A 值
  3. 输出结构化的训练数据（.parquet / .json），直接供 train_adapter.py 使用

输出格式（每条数据）:
  {
    "audio_path": "path/to/audio.wav",
    "text": "calm piano melody",          # 文本描述（可选）
    "valence": 0.8,                        # 0~1
    "arousal": 0.3,                        # 0~1
    "duration": 8.0,                       # 秒
  }

用法：
  # 演示模式（生成合成数据测试流程）
  python data_prep.py --demo --output_dir ./data

  # 使用 DEAM 数据集（需先下载音频文件）
  python data_prep.py --dataset deam --deam_dir ./DEAM --output_dir ./data

  # 使用 MTG-Jamendo mood/theme 元数据（无需音频也可生成标注）
  python data_prep.py --mtg_metadata /path/to/autotagging_moodtheme.tsv --output_dir ./data
  python data_prep.py --mtg_metadata /path/to/autotagging_moodtheme.tsv \
                      --mtg_audio /path/to/audio_dir --output_dir ./data

  # 自动标注无标签音乐
  python data_prep.py --auto_label --input_dir ./raw_music --output_dir ./data
"""

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch


# ===================================================================
# 1. 合成演示数据
# ===================================================================
def generate_demo_data(output_dir: str, num_samples: int = 10):
    """生成合成音频 + 标注，用于快速验证训练流程。

    用不同频率的正弦波模拟不同情感的音乐:
      - 高 Valence + 高 Arousal  → 快节奏、高音（开心/兴奋）
      - 低 Valence + 低 Arousal  → 慢节奏、低音（悲伤/平静）
      - 混合                      → 中间状态
    """
    import librosa

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(exist_ok=True)

    sr = 16000
    records = []

    for i in range(num_samples):
        # 随机 V-A
        valence = np.random.uniform(0.1, 0.9)
        arousal = np.random.uniform(0.1, 0.9)

        # 用 V-A 控制合成参数
        duration = 2.0 + arousal * 2.0          # 高 arousal → 更长（感觉更快）
        base_freq = 200 + valence * 400          # 高 valence → 更高音
        beat_freq = 1.0 + arousal * 3.0          # 高 arousal → 更快节拍

        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # 主音
        tone = 0.5 * np.sin(2 * np.pi * base_freq * t)
        # 节拍感（振幅调制）
        beat = 0.3 * np.sin(2 * np.pi * beat_freq * t)
        # 谐波
        harm = 0.2 * np.sin(2 * np.pi * base_freq * 2 * t)
        audio = tone * (1 + beat) + harm

        # 归一化
        audio = audio / np.max(np.abs(audio) + 1e-8)

        filename = f"synth_{i:03d}_v{valence:.2f}_a{arousal:.2f}.wav"
        filepath = audio_dir / filename
        sf.write(str(filepath), audio, sr)

        # 生成文本描述
        text = _va_to_text(valence, arousal)

        records.append({
            "audio_path": str(filepath.absolute()),
            "text": text,
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "duration": round(duration, 2),
        })

    # 保存标注
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"[demo] 生成 {num_samples} 条演示数据:")
    print(f"      音频: {audio_dir}/")
    print(f"      标注: {meta_path}")
    return records


def _va_to_text(valence: float, arousal: float) -> str:
    """根据 V-A 值生成简单的文本描述。"""
    if valence > 0.6 and arousal > 0.6:
        return "Happy and energetic music with fast tempo"
    elif valence > 0.6 and arousal <= 0.4:
        return "Calm and peaceful melody, relaxing"
    elif valence <= 0.4 and arousal > 0.6:
        return "Tense and intense dramatic music"
    elif valence <= 0.4 and arousal <= 0.4:
        return "Sad and melancholic slow piano piece"
    elif valence > 0.5 and arousal > 0.4:
        return "Uplifting cheerful background music"
    else:
        return "Mellow ambient instrumental music"


# ===================================================================
# 2. DEAM 数据集处理
# ===================================================================
def prepare_deam(deam_dir: str, output_dir: str):
    """处理 DEAM 数据集。

    DEAM 结构要求:
      deam_dir/
        ├── audio/              # .mp3 或 .wav 音频文件
        ├── annotations/
        │   ├── annotations.csv        # 文件级 V-A 标注（平均值）
        │   └── annotations_per_second.csv  # 每秒 V-A 标注

    如果 annotations 不存在，会尝试从官方源下载。
    注意: 音频文件需自行获取（DEAM 音频有版权限制）。
    """
    deam_dir = Path(deam_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 尝试下载标注文件
    annot_dir = deam_dir / "annotations"
    annot_dir.mkdir(parents=True, exist_ok=True)

    annot_file = annot_dir / "annotations.csv"
    if not annot_file.exists():
        _download_if_missing(
            "https://raw.githubusercontent.com/JoycexR/DEAM/master/annotations/annotations.csv",
            annot_file,
        )

    # 解析标注
    records = []
    if annot_file.exists():
        import csv
        with open(annot_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                song_id = row.get("song_id", "").strip()
                valence = float(row.get("valence_mean", 0))
                arousal = float(row.get("arousal_mean", 0))

                # 寻找对应的音频文件
                audio_path = None
                for ext in [".mp3", ".wav", ".ogg"]:
                    candidate = deam_dir / "audio" / f"{song_id}{ext}"
                    if candidate.exists():
                        audio_path = str(candidate.absolute())
                        break

                if audio_path is None:
                    continue  # 音频文件不存在，跳过

                # V-A 归一化到 0~1（DEAM 原始范围是 1~9）
                valence_norm = max(0.0, min(1.0, (valence - 1) / 8))
                arousal_norm = max(0.0, min(1.0, (arousal - 1) / 8))

                records.append({
                    "audio_path": audio_path,
                    "text": "",  # DEAM 无文本描述
                    "valence": round(valence_norm, 3),
                    "arousal": round(arousal_norm, 3),
                    "duration": 0.0,
                })

    # 保存
    if records:
        meta_path = output_dir / "metadata_deam.json"
        with open(meta_path, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"[DEAM] 处理完成: {len(records)} 条数据 → {meta_path}")
    else:
        print(f"[DEAM] 未找到有效数据。请确保 {deam_dir}/audio/ 中有音频文件。")

    return records


def _download_if_missing(url: str, dest: Path):
    """如果目标文件不存在，从 URL 下载。"""
    if dest.exists():
        return
    print(f"  下载 {url} → {dest} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"  下载完成")


# ===================================================================
# 3. CLAP 自动情感标注
# ===================================================================
class CLAPAutoLabeler:
    """使用 HuggingFace CLAP 模型给无标签音乐自动标注 V-A。

    策略:
      1. 定义一组情感锚点文本（描述不同 V-A 组合）
      2. 用 CLAP 计算音频与各锚点文本的相似度
      3. 用加权平均估算 V-A 值

    参考锚点:
      - "happy exciting music"    → V=0.85, A=0.80
      - "calm relaxing music"     → V=0.75, A=0.20
      - "sad melancholic music"   → V=0.20, A=0.25
      - "tense dramatic music"    → V=0.25, A=0.75
      - "neutral background music" → V=0.50, A=0.50
    """

    _ANCHORS = [
        ("happy exciting upbeat music", 0.85, 0.80),
        ("calm relaxing peaceful music", 0.75, 0.20),
        ("sad melancholic depressing music", 0.20, 0.25),
        ("tense dramatic aggressive music", 0.25, 0.75),
        ("neutral ordinary background music", 0.50, 0.50),
        ("romantic tender loving music", 0.80, 0.40),
        ("angry furious intense music", 0.15, 0.85),
        ("boring dull monotonous music", 0.30, 0.15),
    ]

    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def _lazy_init(self):
        if self.model is not None:
            return
        from transformers import ClapModel, ClapProcessor

        print(f"[CLAP] 加载模型到 {self.device} ...")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model.to(self.device)
        self.model.eval()

        # 编码锚点文本
        texts = [a[0] for a in self._ANCHORS]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            self.text_embeddings = self.model.get_text_features(**inputs)
            self.text_embeddings = self.text_embeddings / self.text_embeddings.norm(
                dim=-1, keepdim=True
            )
        print(f"[CLAP] 锚点文本已编码: {len(texts)} 条")

    @torch.no_grad()
    def predict_va(self, audio_path: str) -> tuple[float, float]:
        """用 CLAP 预测一段音频的 V-A 值。

        返回: (valence, arousal) 均在 0~1 范围。
        """
        import librosa

        self._lazy_init()

        # 加载音频（CLAP 期望 48kHz）
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=10.0)

        # 处理
        inputs = self.processor(
            audios=audio, sampling_rate=48000, return_tensors="pt"
        ).to(self.device)

        # 编码音频
        audio_emb = self.model.get_audio_features(**inputs)
        audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)

        # 计算与各锚点的相似度
        sim = (audio_emb @ self.text_embeddings.T).squeeze(0)  # (num_anchors,)
        weights = torch.softmax(sim / 0.1, dim=0)  # temperature scaling

        # 加权平均 V-A
        anchors = torch.tensor(
            [(v, a) for _, v, a in self._ANCHORS], device=self.device
        )
        pred = (weights.unsqueeze(1) * anchors).sum(dim=0)

        return pred[0].item(), pred[1].item()

    def label_directory(self, input_dir: str, output_dir: str):
        """标注整个目录下的音频文件。"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
        audio_files = sorted([
            f for f in input_dir.iterdir()
            if f.suffix.lower() in audio_exts
        ])

        if not audio_files:
            print(f"[CLAP] {input_dir} 中未找到音频文件")
            return []

        print(f"[CLAP] 开始标注 {len(audio_files)} 个文件 ...")
        records = []
        for i, fpath in enumerate(audio_files):
            try:
                valence, arousal = self.predict_va(str(fpath))
                records.append({
                    "audio_path": str(fpath.absolute()),
                    "text": "",
                    "valence": round(valence, 3),
                    "arousal": round(arousal, 3),
                    "duration": 0.0,
                })
                print(f"  [{i+1}/{len(audio_files)}] {fpath.name} → V={valence:.3f}, A={arousal:.3f}")
            except Exception as e:
                print(f"  [{i+1}/{len(audio_files)}] {fpath.name} → 失败: {e}")

        # 保存
        if records:
            meta_path = output_dir / "metadata_auto.json"
            with open(meta_path, "w") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            print(f"[CLAP] 标注完成: {len(records)} 条 → {meta_path}")

        return records


# ===================================================================
# 4. MTG-Jamendo 数据集处理（mood tag → V-A 映射）
# ===================================================================

# MTG-Jamendo mood/theme 标签 → V-A 映射表
# 基于 Russell 的 Valence-Arousal 情感模型
MOODTAG_TO_VA = {
    # --- 高 Valence / 高 Arousal (快乐/兴奋) ---
    "mood/theme---happy":        (0.85, 0.75),
    "mood/theme---joyful":       (0.88, 0.70),
    "mood/theme---fun":          (0.80, 0.78),
    "mood/theme---festive":      (0.82, 0.80),
    "mood/theme---party":        (0.78, 0.85),
    "mood/theme---excited":      (0.75, 0.88),
    "mood/theme---energetic":    (0.70, 0.85),
    "mood/theme---groovy":       (0.75, 0.70),
    "mood/theme---lively":       (0.78, 0.72),
    "mood/theme---dynamic":      (0.65, 0.78),
    "mood/theme---upbeat":       (0.80, 0.75),
    "mood/theme---bright":       (0.82, 0.65),
    "mood/theme---cheerful":     (0.85, 0.68),
    "mood/theme---summer":       (0.80, 0.70),

    # --- 高 Valence / 低 Arousal (平静/放松) ---
    "mood/theme---calm":         (0.78, 0.18),
    "mood/theme---relaxing":     (0.75, 0.15),
    "mood/theme---peaceful":     (0.80, 0.12),
    "mood/theme---serene":       (0.82, 0.10),
    "mood/theme---soft":         (0.72, 0.20),
    "mood/theme---tender":       (0.78, 0.25),
    "mood/theme---romantic":     (0.80, 0.35),
    "mood/theme---love":         (0.82, 0.38),
    "mood/theme---dreamy":       (0.65, 0.28),
    "mood/theme---meditative":   (0.60, 0.08),
    "mood/theme---ambient":      (0.55, 0.12),
    "mood/theme---chill":        (0.70, 0.22),
    "mood/theme---cool":         (0.68, 0.25),
    "mood/theme---beautiful":    (0.78, 0.30),
    "mood/theme---melodic":      (0.72, 0.35),

    # --- 低 Valence / 低 Arousal (悲伤/忧郁) ---
    "mood/theme---sad":          (0.18, 0.22),
    "mood/theme---melancholic":  (0.20, 0.20),
    "mood/theme---depressing":   (0.10, 0.18),
    "mood/theme---dark":         (0.12, 0.35),
    "mood/theme---gloomy":       (0.15, 0.25),
    "mood/theme---somber":       (0.18, 0.15),
    "mood/theme---nostalgic":    (0.35, 0.30),
    "mood/theme---sentimental":  (0.30, 0.28),
    "mood/theme---lonely":       (0.12, 0.12),
    "mood/theme---ballad":       (0.35, 0.25),

    # --- 低 Valence / 高 Arousal (紧张/愤怒) ---
    "mood/theme---aggressive":   (0.15, 0.82),
    "mood/theme---angry":        (0.08, 0.88),
    "mood/theme---intense":      (0.22, 0.85),
    "mood/theme---dramatic":     (0.28, 0.78),
    "mood/theme---dark":         (0.12, 0.35),  # 也出现在低V低A，取折中
    "mood/theme---drama":        (0.25, 0.75),
    "mood/theme---tense":        (0.18, 0.80),
    "mood/theme---epic":         (0.40, 0.82),
    "mood/theme---powerful":     (0.35, 0.80),
    "mood/theme---heavy":        (0.20, 0.78),

    # --- 中等 Valence / 中等 Arousal (中性) ---
    "mood/theme---neutral":      (0.50, 0.50),
    "mood/theme---background":   (0.50, 0.25),
    "mood/theme---commercial":   (0.55, 0.45),
    "mood/theme---corporate":    (0.52, 0.30),
    "mood/theme---advertising":  (0.60, 0.55),
    "mood/theme---film":         (0.45, 0.50),
    "mood/theme---documentary":  (0.50, 0.35),
    "mood/theme---nature":       (0.65, 0.20),
    "mood/theme---travel":       (0.60, 0.55),
    "mood/theme---space":        (0.45, 0.30),
    "mood/theme---retro":        (0.55, 0.45),
    "mood/theme---minimal":      (0.45, 0.15),

    # --- 其他特殊标签 ---
    "mood/theme---funny":        (0.78, 0.65),
    "mood/theme---weird":        (0.35, 0.60),
    "mood/theme---adventure":    (0.55, 0.70),
    "mood/theme---children":     (0.82, 0.68),
    "mood/theme---emotional":    (0.50, 0.55),
    "mood/theme---hopeful":      (0.75, 0.45),
    "mood/theme---inspiring":    (0.70, 0.55),
    "mood/theme---motivational": (0.65, 0.70),
    "mood/theme---mysterious":   (0.30, 0.45),
    "mood/theme---suspense":     (0.20, 0.65),
    "mood/theme---action":       (0.35, 0.82),
    "mood/theme---sport":        (0.55, 0.78),
    "mood/theme---soundscape":   (0.50, 0.20),
    "mood/theme---piano":        (0.55, 0.25),  # 乐器标签，温和
    "mood/theme---guitar":       (0.50, 0.40),
    "mood/theme---synth":        (0.45, 0.35),
    "mood/theme---orchestral":   (0.50, 0.55),
}


def _moodtags_to_va(tags: list[str]) -> tuple[float, float]:
    """将一组 mood/theme 标签加权平均为连续 V-A 值。

    策略:
      1. 对每个已知的 mood tag 查表得到 (V, A)
      2. 加权平均（权重 = 标签的"信息量"，默认均等）
      3. 若无任何匹配标签，返回中性值 (0.5, 0.5)
    """
    matched = []
    for tag in tags:
        tag = tag.strip()
        if tag in MOODTAG_TO_VA:
            matched.append(MOODTAG_TO_VA[tag])

    if not matched:
        return 0.5, 0.5  # 中性默认值

    v = sum(m[0] for m in matched) / len(matched)
    a = sum(m[1] for m in matched) / len(matched)
    return round(v, 3), round(a, 3)


def prepare_mtg_jamendo(
    metadata_path: str,
    audio_dir: str | None = None,
    output_dir: str = "./data",
    max_samples: int | None = None,
):
    """处理 MTG-Jamendo 数据集（仅需 TSV 元数据，音频可选）。

    将 mood/theme 标签通过 MOODTAG_TO_VA 映射表转为连续 V-A 值。

    Args:
        metadata_path: autotagging_moodtheme.tsv 路径
        audio_dir: 音频目录（可选）。如果提供，会拼接完整 audio_path
        output_dir: 输出目录
        max_samples: 最多处理的样本数（None=全部）

    返回: records list
    """
    import csv

    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []

    with open(metadata_path) as f:
        # 跳过可能的不规则列
        reader = csv.reader(f, delimiter="\t")
        headers = next(reader)
        # 找到各列的索引
        col_map = {}
        for i, h in enumerate(headers):
            col_map[h.strip()] = i

        track_id_col = col_map.get("TRACK_ID", 0)
        path_col = col_map.get("PATH", 3)
        duration_col = col_map.get("DURATION", 4)
        # TAGS 列及其之后的所有列都是标签

        for row_idx, row in enumerate(reader):
            if max_samples and len(records) >= max_samples:
                break

            if len(row) < 6:
                continue

            # 提取所有 mood/theme 标签（第 5 列开始）
            tags = [t.strip() for t in row[5:] if t.strip()]

            if not tags:
                continue

            # 映射到 V-A
            valence, arousal = _moodtags_to_va(tags)

            # 音频路径
            track_path = row[path_col]  # e.g. "22/949222.mp3"
            if audio_dir:
                audio_path = str(Path(audio_dir) / track_path)
                # 检查文件是否存在
                if not Path(audio_path).exists():
                    # 尝试低质量版本
                    alt_path = audio_path.replace(".mp3", ".low.mp3")
                    if Path(alt_path).exists():
                        audio_path = alt_path
                    else:
                        # 音频不存在时仍生成记录，仅标记路径
                        pass
            else:
                audio_path = track_path  # 仅路径，无实际文件

            # 从 tags 生成文本描述
            text = _moodtags_to_text(tags)

            try:
                duration = float(row[duration_col])
            except (ValueError, IndexError):
                duration = 0.0

            records.append({
                "audio_path": audio_path,
                "track_id": row[track_id_col],
                "text": text,
                "valence": valence,
                "arousal": arousal,
                "tags": tags,
                "duration": round(duration, 1),
            })

    # 保存
    meta_path = output_dir / "metadata_mtg.json"
    with open(meta_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 统计
    tag_usage = {}
    for r in records:
        for t in r["tags"]:
            tag_usage[t] = tag_usage.get(t, 0) + 1

    print(f"[MTG-Jamendo] 处理完成: {len(records)} 条数据")
    print(f"[MTG-Jamendo] 涉及 {len(tag_usage)} 种 mood 标签")
    print(f"[MTG-Jamendo] V 范围: {min(r['valence'] for r in records):.3f} ~ "
          f"{max(r['valence'] for r in records):.3f}")
    print(f"[MTG-Jamendo] A 范围: {min(r['arousal'] for r in records):.3f} ~ "
          f"{max(r['arousal'] for r in records):.3f}")
    print(f"[MTG-Jamendo] 标注保存到: {meta_path}")

    # 打印 top-10 最常用标签
    top_tags = sorted(tag_usage.items(), key=lambda x: -x[1])[:10]
    print(f"[MTG-Jamendo] Top-10 标签: ", end="")
    for tag, cnt in top_tags:
        short = tag.replace("mood/theme---", "")
        v, a = MOODTAG_TO_VA.get(tag, (0.5, 0.5))
        print(f"{short}(V={v},A={a}):{cnt}  ", end="")
    print()

    return records


def _moodtags_to_text(tags: list[str]) -> str:
    """从 mood 标签生成简短文本描述。"""
    # 取前 3 个最有表现力的标签（去掉前缀）
    short_tags = [t.replace("mood/theme---", "") for t in tags[:3]]
    return f"{' '.join(short_tags)} music".strip()


# ===================================================================
# 5. 数据合并与格式转换
# ===================================================================
def merge_and_split(
    records_list: list[list[dict]],
    output_dir: str,
    split_ratio: tuple[float, float, float] = (0.8, 0.1, 0.1),
):
    """合并多个来源的数据，按比例划分 train/val/test。

    Args:
        records_list: 多个来源的数据列表（如 [deam_records, auto_labeled_records]）
        output_dir: 输出目录
        split_ratio: (train, val, test) 比例

    输出:
        output_dir/
            train.json
            val.json
            test.json
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 合并
    all_records = []
    for records in records_list:
        all_records.extend(records)

    if not all_records:
        print("[merge] 没有数据可合并")
        return

    # 随机打乱
    rng = np.random.RandomState(42)
    rng.shuffle(all_records)

    # 划分
    n = len(all_records)
    n_train = int(n * split_ratio[0])
    n_val = int(n * split_ratio[1])

    splits = {
        "train": all_records[:n_train],
        "val": all_records[n_train:n_train + n_val],
        "test": all_records[n_train + n_val:],
    }

    # 保存 JSON
    for split_name, split_data in splits.items():
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)
        print(f"[merge] {split_name}: {len(split_data)} 条 → {path}")

    # 同时保存为 PyTorch 格式，供 train_adapter.py 直接加载
    for split_name, split_data in splits.items():
        if not split_data:
            continue
        # 为了节省内存，只保存元信息路径
        path = output_dir / f"{split_name}_meta.json"
        with open(path, "w") as f:
            json.dump(split_data, f, indent=2, ensure_ascii=False)

    print(f"[merge] 总计: {n} 条数据")

    # 打印统计信息
    valences = [r["valence"] for r in all_records]
    arousals = [r["arousal"] for r in all_records]
    print(f"[merge] V 范围: {min(valences):.3f}~{max(valences):.3f}, "
          f"均值: {np.mean(valences):.3f}")
    print(f"[merge] A 范围: {min(arousals):.3f}~{max(arousals):.3f}, "
          f"均值: {np.mean(arousals):.3f}")

    return splits


# ===================================================================
# 6. 加载数据供 train_adapter.py 使用
# ===================================================================
class EmotionMusicDataset(torch.utils.data.Dataset):
    """从预处理好的 JSON 文件加载数据，供 train_adapter.py 使用。

    替换 train_adapter.py 中的 DummyMusicDataset 即可。
    """

    def __init__(
        self,
        meta_path: str,
        audio_cache_dir: str | None = None,
        max_audio_len: int = 16000 * 8,  # 8秒 @ 16kHz
    ):
        with open(meta_path) as f:
            self.records = json.load(f)
        self.audio_cache_dir = audio_cache_dir
        self.max_audio_len = max_audio_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        import librosa
        from transformers import AutoProcessor

        record = self.records[idx]

        # 加载音频
        audio_path = record["audio_path"]
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # 截断或填充到固定长度
        if len(audio) > self.max_audio_len:
            audio = audio[:self.max_audio_len]
        else:
            audio = np.pad(audio, (0, self.max_audio_len - len(audio)))

        # 用 MusicGen 的 processor 处理文本
        text = record.get("text", "")
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        inputs = processor(text=[text], padding=True, return_tensors="pt")

        valence = torch.tensor(record["valence"], dtype=torch.float)
        arousal = torch.tensor(record["arousal"], dtype=torch.float)

        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "valence": valence,
            "arousal": arousal,
            "audio_path": audio_path,
        }


# ===================================================================
# 7. 主入口
# ===================================================================
def main(args):
    all_records = []

    if args.demo:
        print("=" * 50)
        print("演示模式: 生成合成数据")
        print("=" * 50)
        records = generate_demo_data(args.output_dir, num_samples=args.num_samples)
        all_records.append(records)

    if args.deam_dir:
        print("=" * 50)
        print("DEAM 数据集处理")
        print("=" * 50)
        records = prepare_deam(args.deam_dir, args.output_dir)
        if records:
            all_records.append(records)

    if args.mtg_metadata:
        print("=" * 50)
        print("MTG-Jamendo 数据集处理 (mood tag → V-A)")
        print("=" * 50)
        records = prepare_mtg_jamendo(
            metadata_path=args.mtg_metadata,
            audio_dir=args.mtg_audio,
            output_dir=args.output_dir,
            max_samples=args.num_samples,
        )
        if records:
            all_records.append(records)

    if args.auto_label and args.input_dir:
        print("=" * 50)
        print("CLAP 自动标注")
        print("=" * 50)
        labeler = CLAPAutoLabeler()
        # 只在 demo 模式下不重复标注演示数据
        records = labeler.label_directory(args.input_dir, args.output_dir)
        if records:
            all_records.append(records)
    elif args.auto_label:
        print("[CLAP] 请指定 --input_dir 指定待标注音频目录")

    # 合并并划分
    if all_records:
        merge_and_split(all_records, args.output_dir)
        print(f"\n✅ 全部完成！数据保存在: {args.output_dir}/")
    else:
        print("\n⚠️  未生成任何数据。使用 --demo 或提供数据集路径。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="数据预处理与自动情感标注")
    parser.add_argument("--demo", action="store_true",
                        help="演示模式: 生成合成数据")
    parser.add_argument("--deam_dir", type=str, default=None,
                        help="DEAM 数据集路径")
    parser.add_argument("--mtg_metadata", type=str, default=None,
                        help="MTG-Jamendo autotagging_moodtheme.tsv 路径")
    parser.add_argument("--mtg_audio", type=str, default=None,
                        help="MTG-Jamendo 音频目录（可选，仅用于拼接路径）")
    parser.add_argument("--auto_label", action="store_true",
                        help="用 CLAP 自动标注无标签音乐")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="待标注的音频目录")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出目录")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="处理的样本数上限（None=全部）")
    args = parser.parse_args()

    main(args)
