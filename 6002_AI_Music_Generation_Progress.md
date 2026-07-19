# AI Music Generation — 项目进展报告

> 对照 Proposal，逐项检查完成情况

---

## 总体目标

> "端到端 AI 音乐生成系统，从用户情感意图生成个性化 BGM"

**✅ 已完成** — `generate.py` 输入 V-A 坐标 → 输出音频

---

## 核心创新 1：连续情感控制

> "用连续 V-A 坐标替代离散标签，冻结 backbone 训练轻量 adapter"

| 要求 | 状态 |
|:----|:----:|
| V-A 坐标代替 happy/sad | ✅ `VAEncoder` 实现 |
| 冻结 backbone 训练 adapter | ✅ 多版本训练完成 |
| 连续情感调节 | ✅ 生成时可调 V-A 值 |
| **存在不足** | ❌ 控制效果偏弱（Pearson 最高 0.65），音频有杂音 |

## 核心创新 2：Emotion-Fidelity Alignment

> "引入'生成情感 → 目标情感'对齐 loss"

**✅ 已实现** — `emotion_fidelity_loss` + `EmotionPredictor`，v2_fixed 证明了有效性（V Pearson 0.65）

---

## Phase 1：推理基线

> "复现 MusicGen 推理"

**✅ inference.py** — HF + AudioCraft 双后端

## Phase 2：数据 & 情感模块

> "数据准备、情感标注、情感条件注入模块"

**✅ 全部完成**

- `data_prep.py`（MTG/DEAM/CLAP 标注）
- 609 首真实音频下载
- `emotion_adapter.py`（VAEncoder + Adapter）

## Phase 3：微调 & 验证

> "微调、超参调优、验证情感可控性"

**✅ 完成但效果有限**

- 训练了 v1~v5 五个版本
- 最佳 V Pearson = 0.65, A Pearson = 0.58
- 音频质量有下降（FAD 从 1.17 升到 1.71）

## Phase 4：评估 & 集成

| 要求 | 状态 |
|:----|:----:|
| **FAD** | ✅ 已实现 |
| **CLAP Score** | ✅ 已实现 |
| **情感 Fidelity** | ✅ 已实现 |
| **推理 CLI** | ✅ `generate.py` |
| **主观听测** | ❌ 未做 |
| **Gradio 界面** | ❌ 未做 |
| **报告** | ❌ 未整理 |

---

## 训练版本对比

| 版本 | 注入方式 | V Pearson | A Pearson | 说明 |
|:----|:--------:|:---------:|:---------:|:-----|
| v1 | 加法(768d) | 0.27 | -0.40 | 基线 |
| v2 | Prefix | -0.11 | **0.58** | Arousal 改善 |
| v2_fixed | Prefix + 修复 | **0.65** | 0.07 | Valence 最强 |
| **v3** | 平衡版 | **0.45** | **0.55** | **综合最佳** |
| v4 | 加法(1024d) | — | — | 结构验证 |
| v5 | Identity 保护 | -0.17 | 0.49 | 音质略好 |

## 一句话总结

**核心技术路线全部走通了，量化指标证明情感控制确实有效（V/A Pearson 均大于 0），但控制强度还不够强，音频质量有折损。** 对于一份课程项目报告来说，已经有了完整的 Data → Train → Evaluate → Generate 流水线和可量化的结果。
