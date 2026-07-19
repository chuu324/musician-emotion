# 🎵 基于 MusicGen Adapter 的连续情感可控 BGM 生成

> **课程项目 6002** — 通过 Valence-Arousal 连续坐标控制 MusicGen-small 生成情感化背景音乐

---

## 📋 项目概述

本项目实现了一个端到端的 AI 音乐生成系统，能够从连续的 **Valence-Arousal (V-A)** 情感坐标生成个性化背景音乐 (BGM)。核心思路是冻结预训练的 **MusicGen-small** 骨干网络，仅训练轻量级情感适配器 (adapter)，在不对骨干网络做全量微调的前提下实现情感控制。

### 项目结构

```
ddd/
├── code/
│   ├── phase1_musicgen_baseline/       # Phase 1: MusicGen 基线推理
│   │   ├── inference.py                # 加载 MusicGen，文本 prompt 生成音频
│   │   ├── requirements.txt
│   │   └── outputs_demo/               # 5 个示例输出音频
│   │
│   └── phase2_emotion_control/         # Phase 2: 情感控制模块
│       ├── config.yaml                 # 训练配置（参看用）
│       ├── requirements.txt
│       ├── download_mtg_subset.py      # 下载 MTG-Jamendo 数据集音频
│       ├── download_mtg_subset.sh      # Shell 版下载脚本
│       ├── data_prep.py                # 数据预处理 & 自动 V-A 标注
│       ├── emotion_adapter.py          # 情感 Adapter 核心模块
│       ├── emotion_dsp.py              # 两阶段 DSP 情感处理流水线
│       ├── emotion_prompt.py           # V-A → 文本 prompt 映射
│       ├── train_adapter.py            # 训练情感 Adapter
│       ├── generate.py                 # 情感控制 BGM 生成 CLI
│       ├── evaluate.py                 # 评估脚本
│       ├── data/                       # 数据集元数据 (JSON)
│       ├── checkpoints/                # 训练好的模型权重
│       ├── eval_results/               # 评估结果 & 生成样本
│       └── report/                     # 报告图表生成脚本 & PNG
│
├── mtg_data/audio/                     # MTG-Jamendo 数据集音频
├── mtg-jamendo-dataset/                # 数据集官方仓库（元数据 & 工具脚本）
│
├── 6002_AI_Music_Generation_Proposal.md
├── 6002_AI_Music_Generation_Progress.md
├── 6002_AI_Music_Generation_Report_Draft.md       # 最终报告（英文）
├── 6002_AI_Music_Generation_Report_Draft_CN.md    # 最终报告（中文）
├── .gitignore
└── README.md
```

---

## 🚀 快速开始

### 环境要求

- Python 3.10+
- PyTorch 2.0+ (CUDA 推荐)
- 12GB+ VRAM (RTX 3060)

### 安装依赖

```bash
# Phase 1 基线
pip install -r code/phase1_musicgen_baseline/requirements.txt

# Phase 2 情感控制
pip install -r code/phase2_emotion_control/requirements.txt
```

### 下载数据集（可选）

训练需要 MTG-Jamendo 数据集音频：

```bash
cd code/phase2_emotion_control
python download_mtg_subset.py
```

数据将下载到 `~/ddd/mtg_data/audio/`。

---

## 🎮 使用指南

### 1. 情感控制生成

```bash
cd code/phase2_emotion_control

# Prompt 模式（音质最好，推荐）
python generate.py --preset happy --mode prompt -o happy.wav

# DSP 模式（两阶段：生成 + 后处理）
python generate.py --preset sad --mode dsp -o sad.wav

# Adapter 模式（使用训练好的模型）
python generate.py --preset calm --mode adapter -o calm.wav
```

**支持的情感预设：** `happy`, `sad`, `calm`, `tense`, `angry`, `romantic`, `epic`, `dreamy`, `dark`, `neutral`, `uplifting`, `boring`

也可自定义 V-A 值：

```bash
python generate.py --text "piano melody" --valence 0.2 --arousal 0.8 --mode dsp -o tense.wav
```

### 2. 训练

```bash
# Dummy 模式（快速验证训练流程）
python train_adapter.py

# Demo 模式（使用合成音频）
python train_adapter.py --data_mode demo --data_path ./data/train.json

# MTG 模式（真实数据）
python train_adapter.py --data_mode mtg --data_path ./data/train.json --epochs 30

# 启用情感对齐损失
python train_adapter.py --use_fidelity
```

### 3. 评估

```bash
# 全面评估（FAD + CLAP Score + Emotion Fidelity）
python evaluate.py --checkpoint checkpoints/adapter_v3.pth

# 仅评估情感 Fidelity（最快）
python evaluate.py --checkpoint checkpoints/adapter_v3.pth --fidelity_only
```

---

## 🧠 模型架构

| 组件 | 说明 |
|:-----|:------|
| **基座模型** | MusicGen-small (586M, 24 层 Transformer 解码器) |
| **VAEncoder** | 2(VA) → 128 → 1024 情感嵌入 |
| **注入策略** | 解码器层间偏置注入 (v3) |
| **训练参数** | ~132K (仅 0.02% 总参数) |

### 注入策略版本

| 版本 | 方法 | 注入点 |
|:----|:----|:-------|
| v1 | 加性注入 | 编码器输出 (768d) |
| v2 | 前缀注入 | 8 个可学习 prefix token |
| **v3** ✅ | **加性注入** | **解码器 (1024d) — 综合最优** |
| v4 | 恒等保护 | 编码器 + 残差 |
| v5 | 恒等保护加性 | 编码器 |
| v6 | Scale 分离 | 解码器 + 自适应缩放 |
| v7 | 解码器 Hook | 24 层逐层 Tanh 偏置 |

---

## 📊 实验结果

| 版本 | V Pearson | A Pearson | FAD ↓ | CLAP ↑ |
|:----|:---------:|:---------:|:-----:|:------:|
| **v3 (平衡适配器)** | **0.446** | **0.554** | 1.715 | -0.054 |
| v7_10tar (解码器 Hook) | 0.373 | 0.154 | **1.520** | **0.009** |

图表详见 `code/phase2_emotion_control/report/figures/`。

---

## 📁 数据

| 数据集 | 曲目数 | 标签 | 来源 |
|:-------|:------:|:----|:-----|
| MTG-Jamendo mood/theme | 1,990 | 59 情感标签 → V-A | [GitHub](https://github.com/MTG/mtg-jamendo-dataset) |

数据划分: 训练 1,590 / 验证 193 / 测试 208

---

## 📄 报告

- [`6002_AI_Music_Generation_Report_Draft.md`](./6002_AI_Music_Generation_Report_Draft.md) — 英文报告
- [`6002_AI_Music_Generation_Report_Draft_CN.md`](./6002_AI_Music_Generation_Report_Draft_CN.md) — 中文报告
- [`6002_AI_Music_Generation_Proposal.md`](./6002_AI_Music_Generation_Proposal.md) — 项目提案
- [`6002_AI_Music_Generation_Progress.md`](./6002_AI_Music_Generation_Progress.md) — 进展报告

---

## 📌 备注

- 数据集音频 (`mtg_data/audio/`) 因体积较大未包含在 Git 仓库中，如需使用请运行 `download_mtg_subset.py`
- 检查点 `adapter_v3.pth` (102MB) 超过 GitHub 单文件 100MB 限制，未纳入版本控制
- 所有评估图表可通过 `code/phase2_emotion_control/report/generate_charts.py` 重新生成

---

## 👥 贡献者

课程项目 6002
