"""
evaluate.py — Phase 4 评估脚本
================================
评估训练好的情感 Adapter 的三个指标：
  1. 情感 Fidelity — 生成音频的情感与目标 V-A 的一致性
  2. CLAP Score   — 文本描述与生成音频的对齐程度
  3. FAD          — 生成音频的质量（Fréchet Audio Distance）

用法:
  # 全面评估
  python evaluate.py --checkpoint checkpoints/adapter_mtg_real.pth

  # 只评估情感 fidelity（最快）
  python evaluate.py --checkpoint checkpoints/adapter_mtg_real.pth --fidelity_only

  # 指定输出目录
  python evaluate.py --checkpoint checkpoints/adapter_mtg_real.pth --output_dir ./eval_results
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from emotion_adapter import build_model, generate_with_emotion


# ===================================================================
# 工具：保存音频
# ===================================================================
def save_audio(audio, sample_rate: int, output_path: str):
    """保存音频为 WAV 文件（带归一化）。"""
    import soundfile as sf
    import numpy as np
    audio_np = audio.detach().cpu().numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(1, -1)
    peak = np.max(np.abs(audio_np))
    if peak > 0.85:
        audio_np = audio_np * (0.85 / peak)
    sf.write(output_path, audio_np.T, sample_rate, subtype="PCM_16")
    return output_path


# ===================================================================
# 1. 情感 Fidelity 评估
# ===================================================================
class EmotionFidelityEvaluator:
    """用 CLAP + 锚点文本预测生成音频的 V-A，与目标 V-A 比较。

    指标:
      - VAE (Valence Absolute Error): |pred_V - target_V|
      - AAE (Arousal Absolute Error): |pred_A - target_A|
      - VAE_std, AAE_std
      - Pearson 相关系数
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.clap_model = None
        self.processor = None
        self.text_embeddings = None

        # 8 个情感锚点 (text, V, A)
        self.anchors = [
            ("happy exciting upbeat music", 0.85, 0.80),
            ("calm relaxing peaceful music", 0.75, 0.20),
            ("sad melancholic depressing music", 0.20, 0.25),
            ("tense dramatic aggressive music", 0.25, 0.75),
            ("neutral ordinary background music", 0.50, 0.50),
            ("romantic tender loving music", 0.80, 0.40),
            ("angry furious intense music", 0.15, 0.85),
            ("boring dull monotonous music", 0.30, 0.15),
        ]

    def _lazy_init(self):
        if self.clap_model is not None:
            return
        from transformers import ClapModel, ClapProcessor

        print("[Fidelity] 加载 CLAP 模型 ...")
        self.clap_model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.clap_model.to(self.device)
        self.clap_model.eval()

        texts = [a[0] for a in self.anchors]
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.clap_model.get_text_features(**inputs)
            # 新版 CLAP 返回 BaseModelOutputWithPooling，text features 在 pooler_output
            if hasattr(outputs, 'pooler_output'):
                self.text_embeddings = outputs.pooler_output
            elif hasattr(outputs, 'text_features'):
                self.text_embeddings = outputs.text_features
            else:
                self.text_embeddings = outputs.last_hidden_state.mean(dim=1)
            self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)

    @torch.no_grad()
    def predict_va(self, audio_path: str) -> tuple[float, float]:
        """从音频文件预测 V-A 值。"""
        import librosa

        self._lazy_init()
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=10.0)
        inputs = self.processor(audio=audio, sampling_rate=48000, return_tensors="pt").to(self.device)
        outputs = self.clap_model.get_audio_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            audio_emb = outputs.pooler_output
        elif hasattr(outputs, 'audio_features'):
            audio_emb = outputs.audio_features
        else:
            audio_emb = outputs.last_hidden_state.mean(dim=1)
        audio_emb = F.normalize(audio_emb, dim=-1)

        sim = (audio_emb @ self.text_embeddings.T).squeeze(0)
        weights = torch.softmax(sim / 0.1, dim=0)

        anchors_t = torch.tensor([(v, a) for _, v, a in self.anchors], device=self.device)
        pred = (weights.unsqueeze(1) * anchors_t).sum(dim=0)
        return pred[0].item(), pred[1].item()

    def evaluate(
        self,
        generated_audio_dir: str,
        metadata: list[dict],
    ) -> dict:
        """评估一批生成音频的情感 fidelity。"""
        print("\n" + "=" * 50)
        print("1. 情感 Fidelity 评估")
        print("=" * 50)

        v_errors, a_errors = [], []
        v_targets, a_targets = [], []
        v_preds, a_preds = [], []

        for i, item in enumerate(metadata):
            audio_path = item["gen_path"]
            target_v = item["valence"]
            target_a = item["arousal"]

            if not os.path.isfile(audio_path):
                print(f"  [{i+1}/{len(metadata)}] 文件不存在: {audio_path}")
                continue

            pred_v, pred_a = self.predict_va(audio_path)
            v_err = abs(pred_v - target_v)
            a_err = abs(pred_a - target_a)

            v_errors.append(v_err)
            a_errors.append(a_err)
            v_targets.append(target_v)
            a_targets.append(target_a)
            v_preds.append(pred_v)
            a_preds.append(pred_a)

            print(f"  [{i+1}/{len(metadata)}] V:目标{target_v:.2f},预测{pred_v:.2f},误差{v_err:.3f}  "
                  f"A:目标{target_a:.2f},预测{pred_a:.2f},误差{a_err:.3f}")

        if not v_errors:
            return {"error": "无有效数据"}

        # 汇总
        results = {
            "VAE_mean": float(np.mean(v_errors)),
            "VAE_std": float(np.std(v_errors)),
            "AAE_mean": float(np.mean(a_errors)),
            "AAE_std": float(np.std(a_errors)),
            "V_correlation": float(np.corrcoef(v_targets, v_preds)[0, 1]) if len(v_targets) > 2 else 0,
            "A_correlation": float(np.corrcoef(a_targets, a_preds)[0, 1]) if len(a_targets) > 2 else 0,
            "n_samples": len(v_errors),
        }
        print(f"\n  结果:")
        print(f"    VAE (Valence Absolute Error):  {results['VAE_mean']:.4f} ± {results['VAE_std']:.4f}")
        print(f"    AAE (Arousal Absolute Error):  {results['AAE_mean']:.4f} ± {results['AAE_std']:.4f}")
        print(f"    V Pearson r:                   {results['V_correlation']:.4f}")
        print(f"    A Pearson r:                   {results['A_correlation']:.4f}")
        return results


# ===================================================================
# 2. CLAP Score 评估
# ===================================================================
class CLAPScoreEvaluator:
    """用 CLAP 计算文本-音频相似度（CLAP Score）。

    CLAP Score 越高，表示生成音频与文本描述越匹配。
    参考值: 随机匹配 ~0.10, 良好匹配 ~0.30+
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.processor = None

    def _lazy_init(self):
        if self.model is not None:
            return
        from transformers import ClapModel, ClapProcessor

        print("[CLAP Score] 加载 CLAP 模型 ...")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def score(self, audio_path: str, text: str) -> float:
        """计算单条音频-文本对的 CLAP Score。"""
        import librosa

        self._lazy_init()
        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=10.0)
        inputs = self.processor(
            audio=audio, sampling_rate=48000,
            text=text, padding=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        sim = F.cosine_similarity(outputs.text_embeds, outputs.audio_embeds)
        return sim.item()

    def evaluate(self, generated_audio_dir: str, metadata: list[dict]) -> dict:
        """评估一批生成音频的 CLAP Score。"""
        print("\n" + "=" * 50)
        print("2. CLAP Score 评估")
        print("=" * 50)

        scores = []
        for i, item in enumerate(metadata):
            audio_path = item["gen_path"]
            text = item.get("text", "")

            if not os.path.isfile(audio_path) or not text:
                continue

            score = self.score(audio_path, text)
            scores.append(score)
            print(f"  [{i+1}/{len(metadata)}] score={score:.4f}  text='{text}'")

        if not scores:
            return {"error": "无有效数据"}

        results = {
            "CLAP_score_mean": float(np.mean(scores)),
            "CLAP_score_std": float(np.std(scores)),
            "n_samples": len(scores),
        }
        print(f"\n  结果:")
        print(f"    CLAP Score: {results['CLAP_score_mean']:.4f} ± {results['CLAP_score_std']:.4f}")
        return results


# ===================================================================
# 3. FAD (Fréchet Audio Distance) 评估
# ===================================================================
class FADEvaluator:
    """用 CLAP 嵌入计算 Fréchet Audio Distance。

    FAD 越低越好。参考值:
      - 真实音频 vs 真实音频: ~0
      - 高质量生成: <5
      - 低质量生成: >10
    """

    def __init__(self, device: str = "cuda", embedding_dim: int = 512):
        self.device = device
        self.embedding_dim = embedding_dim
        self.model = None
        self.processor = None

    def _lazy_init(self):
        if self.model is not None:
            return
        from transformers import ClapModel, ClapProcessor

        print("[FAD] 加载 CLAP 模型用于特征提取 ...")
        self.model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        self.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def _get_embedding(self, audio_path: str) -> np.ndarray:
        """提取音频的 CLAP 嵌入向量。"""
        import librosa

        audio, sr = librosa.load(audio_path, sr=48000, mono=True, duration=10.0)
        inputs = self.processor(audio=audio, sampling_rate=48000, return_tensors="pt").to(self.device)
        outputs = self.model.get_audio_features(**inputs)
        if hasattr(outputs, 'pooler_output'):
            emb = outputs.pooler_output
        elif hasattr(outputs, 'audio_features'):
            emb = outputs.audio_features
        else:
            emb = outputs.last_hidden_state.mean(dim=1)
        return emb.cpu().numpy().flatten()

    def _compute_frechet(self, mu1, sigma1, mu2, sigma2):
        """计算 Fréchet Distance（使用 scipy 的矩阵平方根）。"""
        from scipy.linalg import sqrtm
        diff = mu1 - mu2
        covmean = sqrtm((sigma1 @ sigma2).cpu().numpy())
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        covmean = torch.from_numpy(covmean).to(mu1.device)
        return diff @ diff + torch.trace(sigma1 + sigma2 - 2 * covmean).real

    def evaluate(
        self,
        generated_dir: str,
        reference_dir: str | None = None,
        generated_files: list[str] | None = None,
        reference_files: list[str] | None = None,
    ) -> dict:
        """计算生成音频与参考音频的 FAD。

        Args:
            generated_dir: 生成音频目录
            reference_dir: 参考音频目录（真实音频）
            generated_files: 或直接提供生成文件列表
            reference_files: 或直接提供参考文件列表
        """
        print("\n" + "=" * 50)
        print("3. FAD (Fréchet Audio Distance) 评估")
        print("=" * 50)

        self._lazy_init()

        # 收集文件
        if generated_files is None:
            gen_dir = Path(generated_dir)
            generated_files = [str(f) for f in gen_dir.glob("*.wav")] + \
                              [str(f) for f in gen_dir.glob("*.mp3")]

        if reference_files is None and reference_dir:
            ref_dir = Path(reference_dir)
            reference_files = [str(f) for f in ref_dir.glob("*.low.mp3")][:200]  # 取 200 首

        print(f"  生成音频: {len(generated_files)} 个")
        print(f"  参考音频: {len(reference_files) if reference_files else 0} 个")

        # 提取嵌入
        def get_embeddings(file_list, name):
            embs = []
            for i, fpath in enumerate(file_list):
                try:
                    emb = self._get_embedding(fpath)
                    embs.append(emb)
                except Exception as e:
                    print(f"    [{i+1}/{len(file_list)}] 失败: {Path(fpath).name} - {e}")
                if (i + 1) % 50 == 0:
                    print(f"    [{i+1}/{len(file_list)}] 已处理 {name}")
            return np.array(embs)

        gen_embs = get_embeddings(generated_files, "生成")
        if reference_files:
            ref_embs = get_embeddings(reference_files, "参考")

        if len(gen_embs) < 2:
            return {"error": "生成音频不足 (<2)"}

        # 计算统计量
        gen_mu = torch.from_numpy(gen_embs.mean(axis=0)).float()
        gen_sigma = torch.from_numpy(np.cov(gen_embs, rowvar=False)).float()

        results = {"n_generated": len(gen_embs)}

        if reference_files and len(ref_embs) >= 2:
            ref_mu = torch.from_numpy(ref_embs.mean(axis=0)).float()
            ref_sigma = torch.from_numpy(np.cov(ref_embs, rowvar=False)).float()
            fad = self._compute_frechet(gen_mu, gen_sigma, ref_mu, ref_sigma)
            results["FAD"] = float(fad)
            results["n_reference"] = len(ref_embs)
            print(f"\n  FAD = {fad:.4f}")
        else:
            # 没有参考音频时计算生成音频内部的多样性
            print("  无参考音频，跳过 FAD 计算")
            # 计算内部平均 pairwise distance
            from scipy.spatial.distance import pdist
            if len(gen_embs) >= 5:
                intra_dist = pdist(gen_embs, metric='cosine').mean()
                results["intra_cosine_div"] = float(intra_dist)
                print(f"  生成音频内部多样性 (cosine): {intra_dist:.4f}")

        return results


# ===================================================================
# 4. 生成评估用的音频样本
# ===================================================================
def generate_eval_samples(
    model, va_encoder, injector,
    output_dir: str,
    device: str = "cuda",
) -> list[dict]:
    """生成一组覆盖不同 V-A 值的评估音频。

    返回 metadata list，每项含 {text, valence, arousal, gen_path}
    """
    os.makedirs(output_dir, exist_ok=True)

    # 定义评估用例：覆盖 V-A 空间的 8 个象限
    eval_cases = [
        # (text, valence, arousal, description)
        ("Happy energetic music", 0.85, 0.80, "happy_energetic"),
        ("Calm relaxing piano", 0.80, 0.15, "calm_relaxing"),
        ("Sad melancholic melody", 0.20, 0.25, "sad_melancholic"),
        ("Tense dramatic soundtrack", 0.25, 0.75, "tense_dramatic"),
        ("Neutral background ambient", 0.50, 0.50, "neutral"),
        ("Romantic soft love song", 0.80, 0.40, "romantic"),
        ("Angry aggressive rock", 0.15, 0.85, "angry_aggressive"),
        ("Boring monotonous drone", 0.30, 0.15, "boring"),
        # 中间值
        ("Uplifting cheerful tune", 0.70, 0.65, "uplifting"),
        ("Misty dreamy soundscape", 0.55, 0.30, "dreamy"),
        ("Epic orchestral battle", 0.40, 0.80, "epic"),
        ("Dark mysterious atmosphere", 0.20, 0.45, "dark"),
    ]

    metadata = []
    print(f"\n生成 {len(eval_cases)} 个评估音频样本 ...")

    for i, (text, v, a, desc) in enumerate(eval_cases):
        output_path = os.path.join(output_dir, f"eval_{desc}_v{v:.2f}_a{a:.2f}.wav")
        if os.path.exists(output_path):
            print(f"  [{i+1}/{len(eval_cases)}] 已存在: {desc}")
            metadata.append({"text": text, "valence": v, "arousal": a,
                            "gen_path": output_path, "desc": desc})
            continue

        try:
            t0 = time.time()
            audio, sr = generate_with_emotion(
                model, va_encoder, injector,
                text_prompt=text,
                valence=v,
                arousal=a,
                duration=5.0,
                guidance_scale=3.0,
                device=device,
            )
            save_audio(audio, sr, output_path)
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(eval_cases)}] {desc} (V={v},A={a}) {elapsed:.1f}s")
            metadata.append({"text": text, "valence": v, "arousal": a,
                            "gen_path": output_path, "desc": desc})
        except Exception as e:
            print(f"  [{i+1}/{len(eval_cases)}] 失败 {desc}: {e}")

    return metadata


# ===================================================================
# 5. 主函数
# ===================================================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    print(f"PyTorch: {torch.__version__}")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载模型 + checkpoint
    print("\n" + "=" * 50)
    print("加载模型与 checkpoint")
    print("=" * 50)
    model, va_encoder, injector = build_model(
        model_name=args.model,
        device=device,
        freeze_backbone=True,
    )

    if args.checkpoint and os.path.isfile(args.checkpoint):
        import argparse
        with torch.serialization.safe_globals([argparse.Namespace]):
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
        va_encoder.load_state_dict(ckpt["va_encoder_state_dict"])
        if "injector_state_dict" in ckpt:
            injector.load_state_dict(ckpt["injector_state_dict"])
        elif "adapter_state_dict" in ckpt:
            injector.load_state_dict(ckpt["adapter_state_dict"])  # 兼容旧 checkpoint
        print(f"  Checkpoint 已加载: {args.checkpoint}")
    else:
        print(f"  使用未训练的 adapter（随机初始化）")

    va_encoder.eval()
    injector.eval()
    model.eval()

    # 2. 生成评估样本
    eval_dir = os.path.join(output_dir, "generated")
    metadata = generate_eval_samples(model, va_encoder, injector, eval_dir, device)

    if not metadata:
        print("❌ 未生成任何评估样本")
        return

    # 保存 metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # 3. 评估
    all_results = {}
    report_path = os.path.join(output_dir, "results.json")

    if not args.fidelity_only:
        # FAD 评估
        try:
            fad_eval = FADEvaluator(device=device)
            # 参考音频：使用 MTG 验证集中的真实音频
            ref_dir = os.path.expanduser("~/ddd/mtg_data/audio")
            gen_files = [m["gen_path"] for m in metadata]
            fad_results = fad_eval.evaluate(
                generated_dir=eval_dir,
                reference_dir=ref_dir,
                generated_files=gen_files,
            )
            all_results["FAD"] = fad_results
        except Exception as e:
            print(f"  FAD 评估失败: {e}")
            import traceback
            traceback.print_exc()

        # CLAP Score 评估
        try:
            clap_eval = CLAPScoreEvaluator(device=device)
            clap_results = clap_eval.evaluate(eval_dir, metadata)
            all_results["CLAP_Score"] = clap_results
        except Exception as e:
            print(f"  CLAP Score 评估失败: {e}")

    # 情感 Fidelity 评估（始终执行）
    try:
        fid_eval = EmotionFidelityEvaluator(device=device)
        fid_results = fid_eval.evaluate(eval_dir, metadata)
        all_results["Emotion_Fidelity"] = fid_results
    except Exception as e:
        print(f"  Emotion Fidelity 评估失败: {e}")
        import traceback
        traceback.print_exc()

    # 4. 保存和报告
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("📊 评估报告")
    print("=" * 50)

    if "Emotion_Fidelity" in all_results:
        r = all_results["Emotion_Fidelity"]
        print(f"  情感 Fidelity:")
        print(f"    VAE: {r.get('VAE_mean', 'N/A'):.4f}")
        print(f"    AAE: {r.get('AAE_mean', 'N/A'):.4f}")
        print(f"    V Pearson: {r.get('V_correlation', 'N/A'):.4f}")

    if "CLAP_Score" in all_results:
        r = all_results["CLAP_Score"]
        print(f"  CLAP Score: {r.get('CLAP_score_mean', 'N/A'):.4f}")

    if "FAD" in all_results:
        r = all_results["FAD"]
        print(f"  FAD: {r.get('FAD', 'N/A')}")

    print(f"\n  完整报告已保存: {report_path}")
    print(f"  生成音频: {eval_dir}/")
    print("\n✅ 评估完成！")


# ===================================================================
# 入口
# ===================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="情感 Adapter 评估")
    parser.add_argument("--model", type=str, default="facebook/musicgen-small")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/adapter_mtg_real.pth",
                        help="训练好的 checkpoint 路径")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="评估结果输出目录")
    parser.add_argument("--fidelity_only", action="store_true",
                        help="只评估情感 Fidelity（跳过 FAD 和 CLAP Score）")
    args = parser.parse_args()

    main(args)
