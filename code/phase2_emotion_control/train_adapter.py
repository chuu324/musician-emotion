"""
train_adapter.py — 训练情感 Adapter
=====================================
支持三种数据模式：
  1. dummy (默认) — 随机张量，快速验证训练流程
  2. demo        — 使用 data_prep.py 生成的合成音频
  3. mtg         — 使用 MTG-Jamendo mood tag 映射数据

用法：
  # dummy 模式（快速验证）
  python train_adapter.py

  # demo 模式（使用合成音频）
  python train_adapter.py --data_mode demo --data_path ./data/train.json

  # MTG 模式（无音频，仅元数据）
  python train_adapter.py --data_mode mtg --data_path ./data/train.json

  # 启用 emotion-fidelity loss
  python train_adapter.py --use_fidelity
"""

import argparse
import hashlib
import json
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import numpy as np

from emotion_adapter import build_model, forward_with_emotion


# ---------------------------------------------------------------------------
# 情感预测器（用于 fidelity loss，v2 改进）
# ---------------------------------------------------------------------------
class EmotionPredictor(nn.Module):
    """从情感 prefix 嵌入中预测 V-A 值。

    作为 fidelity loss 的判别器，确保 prefix token 包含情感信息。
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, 2)


# ---------------------------------------------------------------------------
# Dummy 数据集（模拟真实数据格式）
# ---------------------------------------------------------------------------
class DummyMusicDataset(Dataset):
    """生成随机 dummy 数据，模拟真实训练数据格式。

    每条数据包含:
        input_ids, attention_mask, decoder_input_ids,
        valence, arousal, labels

    注意: decoder_input_ids 形状为 (num_codebooks, audio_len)，
    因为 MusicGen decoder 期望 (batch * num_codebooks, seq_len)。
    """

    def __init__(
        self,
        num_samples: int = 64,
        text_len: int = 12,
        audio_len: int = 32,
        vocab_size: int = 2048,
        text_vocab_size: int = 32000,
        num_codebooks: int = 4,
    ):
        self.num_samples = num_samples
        self.text_len = text_len
        self.audio_len = audio_len
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.num_codebooks = num_codebooks
        torch.manual_seed(42)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return (
            torch.randint(0, self.text_vocab_size, (self.text_len,)),
            torch.ones(self.text_len, dtype=torch.long),
            torch.randint(0, self.vocab_size, (self.num_codebooks, self.audio_len)),
            torch.tensor(torch.rand(1).item(), dtype=torch.float),
            torch.tensor(torch.rand(1).item(), dtype=torch.float),
            torch.randint(0, self.vocab_size, (self.audio_len, self.num_codebooks)),
        )


def collate_fn(batch):
    """将 batch 中的各条数据拼接成模型输入格式。"""
    input_ids = torch.stack([b[0] for b in batch])
    attention_mask = torch.stack([b[1] for b in batch])
    # decoder_input_ids: (num_codebooks, audio_len) per item → cat to (B*num_codebooks, audio_len)
    decoder_input_ids = torch.cat([b[2] for b in batch], dim=0)
    valence = torch.stack([b[3] for b in batch])
    arousal = torch.stack([b[4] for b in batch])
    labels = torch.stack([b[5] for b in batch])
    return input_ids, attention_mask, decoder_input_ids, valence, arousal, labels


# ---------------------------------------------------------------------------
# 真实数据数据集（从 JSON 元数据加载，编码音频生成 token）
# ---------------------------------------------------------------------------
class RealMusicDataset(Dataset):
    """从 data_prep.py 产出的 JSON 加载真实数据。

    对于有音频文件的条目（demo 模式），编码音频生成 decoder token；
    对于无音频文件条目（MTG 模式），生成占位 token。

    输出格式与 DummyMusicDataset 一致，可无缝替换。
    """

    def __init__(
        self,
        meta_path: str,
        model=None,
        text_processor=None,
        device: str = "cpu",
        audio_len: int = 50,       # token 序列长度
        num_codebooks: int = 4,
        vocab_size: int = 2048,
        text_vocab_size: int = 32000,
        mode: str = "demo",        # "demo" = 有音频, "mtg" = 无音频
        cache_dir: str | None = None,  # 音频编码缓存目录
    ):
        self.device = device
        self.audio_len = audio_len
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.text_vocab_size = text_vocab_size
        self.mode = mode
        self.model = model
        self.text_processor = text_processor
        self.cache_dir = cache_dir

        # 加载元数据，只保留有真实音频的文件
        with open(meta_path) as f:
            all_records = json.load(f)

        self.records = [r for r in all_records if os.path.isfile(r.get("audio_path", ""))]

        print(f"[RealMusicDataset] 加载 {len(self.records)}/{len(all_records)} 条数据 "
              f"(mode={mode}, 无音频的已跳过)")

        if mode == "demo" and len(self.records) == 0:
            print("[RealMusicDataset] ❌ demo 模式但没有找到任何音频文件！")

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # 延迟初始化 text processor（避免加载两次）
        if self.text_processor is None:
            from transformers import AutoProcessor
            self.text_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    def __len__(self):
        return len(self.records)

    def _encode_audio(self, audio_path: str):
        """加载音频并用 EnCodec 编码为离散 token（带缓存）。"""
        # 检查缓存
        cache_key = hashlib.md5(audio_path.encode()).hexdigest()
        if self.cache_dir:
            cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")
            if os.path.isfile(cache_path):
                cached = torch.load(cache_path, weights_only=True)
                return cached["decoder_ids"], cached["labels"]
        else:
            cache_path = None

        import librosa

        # 加载音频
        audio, sr = librosa.load(audio_path, sr=32000, mono=True)
        target_len = 32000 * 5  # 5 秒
        if len(audio) > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - len(audio)))

        audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
        audio_tensor = audio_tensor.to(self.device)

        # EnCodec 编码
        with torch.no_grad():
            encoded = self.model.audio_encoder.encode(audio_tensor)
            audio_codes = encoded.audio_codes  # (1, num_codebooks, seq_len)

        # 截断或填充到 audio_len
        seq_len = audio_codes.shape[-1]
        if seq_len >= self.audio_len:
            audio_codes = audio_codes[..., :self.audio_len]
        else:
            pad = torch.zeros(1, self.num_codebooks, self.audio_len - seq_len,
                              dtype=torch.long, device=self.device)
            audio_codes = torch.cat([audio_codes, pad], dim=-1)

        # 格式转换
        decoder_ids = audio_codes.squeeze(0)  # (num_codebooks, audio_len)
        labels = decoder_ids.transpose(0, 1)  # (audio_len, num_codebooks)
        decoder_ids_cpu, labels_cpu = decoder_ids.cpu(), labels.cpu()

        # 保存缓存
        if cache_path is not None:
            torch.save({"decoder_ids": decoder_ids_cpu, "labels": labels_cpu}, cache_path)

        return decoder_ids_cpu, labels_cpu

    def _generate_dummy_tokens(self):
        """生成占位 token（用于无音频的 MTG 模式）。"""
        decoder_ids = torch.randint(0, self.vocab_size, (self.num_codebooks, self.audio_len))
        labels = torch.randint(0, self.vocab_size, (self.audio_len, self.num_codebooks))
        return decoder_ids, labels

    def __getitem__(self, idx):
        record = self.records[idx]

        # 文本编码（固定长度 12，不足补 0）
        text = record.get("text", "")
        if not text.strip():
            text = "music"
        inputs = self.text_processor(text=[text], padding="max_length",
                                     truncation=True, max_length=12,
                                     return_tensors="pt")
        input_ids = inputs.input_ids.squeeze(0)        # (12,)
        attention_mask = inputs.attention_mask.squeeze(0)  # (12,)

        # V-A
        valence = torch.tensor(record["valence"], dtype=torch.float)
        arousal = torch.tensor(record["arousal"], dtype=torch.float)

        # 音频 token（所有记录都有真实音频）
        decoder_ids, labels = self._encode_audio(record["audio_path"])

        return input_ids, attention_mask, decoder_ids, valence, arousal, labels


# ---------------------------------------------------------------------------
# Emotion-Fidelity Alignment Loss（辅助损失）
# ---------------------------------------------------------------------------
def emotion_fidelity_loss(
    encoder_hidden_states: torch.Tensor,
    target_valence: torch.Tensor,
    target_arousal: torch.Tensor,
    predictor: torch.nn.Module,
    num_prefix: int = 4,
) -> torch.Tensor:
    """情感忠实度对齐损失（v2 改进版）。

    从情感 prefix token 中预测 V-A，与目标 V-A 计算 MSE。
    相比 v1 的改进:
      1. 只取 prefix tokens 预测，不含文本噪声
      2. predictor 作为独立模块传入 optimizer
      3. 更深的 predictor 结构

    Args:
        encoder_hidden_states: (B, N+T, 1024) — prefix + text 投影后的嵌入
        target_valence: (B,)
        target_arousal: (B,)
        predictor: EmotionPredictor 模块
        num_prefix: prefix token 数量
    """
    # 取 encoder_hidden_states 预测 V-A
    # num_prefix=0 时取全部，>0 时只取 prefix 部分
    if num_prefix > 0:
        prefix_states = encoder_hidden_states[:, :num_prefix, :]
        pooled = prefix_states.mean(dim=1)
    else:
        pooled = encoder_hidden_states.mean(dim=1)

    pred_va = predictor(pooled)  # (B, 2)
    target_va = torch.stack([target_valence, target_arousal], dim=1)

    # 分别计算 V 和 A 的 loss，平衡权重防止 Valence 主导
    loss_v = F.mse_loss(pred_va[:, 0], target_va[:, 0])
    loss_a = F.mse_loss(pred_va[:, 1], target_va[:, 1])
    return loss_v + loss_a  # 均等权重


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_one_epoch(
    model, va_encoder, injector,
    dataloader, optimizer, device,
    use_fidelity_loss: bool = False,
    fidelity_weight: float = 0.1,
    emotion_predictor: EmotionPredictor | None = None,
):
    model.eval()  # backbone 冻结，只训练 injector
    va_encoder.train()
    injector.train()

    total_loss = 0.0
    total_fid_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids, attn_mask, decoder_ids, valence, arousal, labels = batch
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        decoder_ids = decoder_ids.to(device)
        valence = valence.to(device)
        arousal = arousal.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ---- 手动前向（Decoder 层注入，encoder 路径不动） ----
        encoder_outputs = model.text_encoder(
            input_ids=input_ids, attention_mask=attn_mask, return_dict=True,
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = model.enc_to_dec_proj(encoder_hidden_states)

        # 设置情感（hook 在 decoder 前向时自动注入每层输出）
        emotion_emb = va_encoder(valence, arousal)
        injector.set_emotion(emotion_emb)

        decoder_outputs = model.decoder(
            input_ids=decoder_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attn_mask,
            labels=labels,
        )

        loss = decoder_outputs.loss

        # ---- Emotion-Fidelity 辅助损失 ----
        if use_fidelity_loss and emotion_predictor is not None:
            fid_loss = emotion_fidelity_loss(
                encoder_hidden_states, valence, arousal,
                emotion_predictor, num_prefix=0,
            )
            loss = loss + fidelity_weight * fid_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if use_fidelity_loss and emotion_predictor is not None:
            total_fid_loss += fid_loss.item()
        num_batches += 1

    if use_fidelity_loss and emotion_predictor is not None:
        return total_loss / num_batches, total_fid_loss / num_batches
    return total_loss / num_batches, 0.0


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print(f"PyTorch 版本: {torch.__version__}")

    # 1. 加载模型
    model, va_encoder, injector = build_model(
        model_name=args.model,
        device=device,
        freeze_backbone=True,
    )

    # 2. 准备数据
    if args.data_mode == "dummy":
        print(f"\n[Dummy 模式] 生成 {args.num_samples} 条随机数据 ...")
        dataset = DummyMusicDataset(
            num_samples=args.num_samples,
            audio_len=args.audio_len,
        )
    elif args.data_mode in ("demo", "mtg"):
        if not args.data_path:
            print(f"\n❌ {args.data_mode} 模式需要 --data_path 指定 JSON 文件")
            return
        print(f"\n[{args.data_mode} 模式] 加载 {args.data_path} ...")
        dataset = RealMusicDataset(
            meta_path=args.data_path,
            model=model,
            device=device,
            audio_len=args.audio_len,
            mode=args.data_mode,
            cache_dir="./cache/audio_tokens",
        )
    else:
        print(f"\n❌ 未知数据模式: {args.data_mode}")
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn,
    )
    print(f"  batch_size={args.batch_size}, {len(dataloader)} batches/epoch, {len(dataset)} 条")

    # 3. 创建情感预测器（用于 fidelity loss）
    emotion_predictor = EmotionPredictor(hidden_size=1024).to(device)

    # 4. 优化器（只更新 adapter + predictor 参数）
    trainable_params = (
        list(va_encoder.parameters())
        + list(injector.parameters())
        + list(emotion_predictor.parameters())
    )
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    # 5. 学习率调度器（缓解训练不稳定）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    print(f"\n开始训练 (epochs={args.epochs}) ...")
    print(f"{'Epoch':>6} | {'Loss':>10} | {'FidLoss':>8} | {'LR':>10} | {'Time':>8}")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        loss, fid_loss = train_one_epoch(
            model, va_encoder, injector, dataloader, optimizer, device,
            use_fidelity_loss=args.use_fidelity,
            fidelity_weight=args.fidelity_weight,
            emotion_predictor=emotion_predictor if args.use_fidelity else None,
        )
        scheduler.step()
        elapsed = time.time() - t0
        current_lr = scheduler.get_last_lr()[0]
        fid_str = f"{fid_loss:.6f}" if args.use_fidelity else "  N/A  "
        print(f"{epoch:>6} | {loss:.6f} | {fid_str:>8} | {current_lr:.2e} | {elapsed:>6.1f}s")

    # 4. 保存 checkpoint
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        torch.save({
            "va_encoder_state_dict": va_encoder.state_dict(),
            "injector_state_dict": injector.state_dict(),
            "emotion_predictor_state_dict": emotion_predictor.state_dict() if args.use_fidelity else None,
            "optimizer_state_dict": optimizer.state_dict(),
            "args": args,
        }, args.save_path)
        print(f"\nCheckpoint 已保存到: {args.save_path}")

    print("\n✅ 训练完成！")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练情感 Adapter")
    parser.add_argument("--model", type=str, default="facebook/musicgen-small",
                        help="MusicGen 模型名称")
    parser.add_argument("--data_mode", type=str, default="dummy",
                        choices=["dummy", "demo", "mtg"],
                        help="数据模式: dummy(随机), demo(合成音频), mtg(无音频标注)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="JSON 数据路径（demo/mtg 模式需要）")
    parser.add_argument("--epochs", type=int, default=5,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="weight decay")
    parser.add_argument("--num_samples", type=int, default=64,
                        help="dummy 模式的数据条数")
    parser.add_argument("--audio_len", type=int, default=50,
                        help="音频 token 序列长度")
    parser.add_argument("--use_fidelity", action="store_true",
                        help="启用 Emotion-Fidelity Alignment Loss")
    parser.add_argument("--fidelity_weight", type=float, default=0.1,
                        help="fidelity loss 权重")
    parser.add_argument("--save_path", type=str, default="checkpoints/adapter.pth",
                        help="checkpoint 保存路径")
    args = parser.parse_args()

    main(args)
