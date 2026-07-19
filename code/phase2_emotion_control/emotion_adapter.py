"""
emotion_adapter.py — 连续情感调节模块（v3: Decoder 层注入）
===========================================================
核心组件：
  1. VAEncoder                — 2D V-A → 1024 情感嵌入
  2. DecoderEmotionInjector   — 生成 24 层层间 bias，注入 decoder 内部
  3. build_model              — 加载 MusicGen，注册层注入 hook
  4. forward_with_emotion     — 带情感条件的训练前向
  5. generate_with_emotion    — 带情感条件的推理生成

注入策略（v3 改进）：
  v1/v2 问题: 修改 encoder 输出 → 扰乱 cross-attention → 音质下降
  v3 方案:  在 decoder 每层输出上加情感 bias，encoder 路径完全不动
            ✅ 不碰 cross-attention 输入
            ✅ 24 层各自学习最佳调制幅度
            ✅ generate() 走原生路径，hook 自动生效

   text → enc → enc_to_dec_proj → decoder(layer0→...→layer23) → audio
                                      ↑           ↑
                              情感bias   情感bias (每层一个)

用法：
  model, va_encoder, injector = build_model()
  loss = forward_with_emotion(model, va_encoder, injector, ...)
  loss.backward()
"""

import torch
import torch.nn as nn
from transformers import MusicgenForConditionalGeneration


# ---------------------------------------------------------------------------
# 1. V-A 坐标编码器
# ---------------------------------------------------------------------------
class VAEncoder(nn.Module):
    """将 2D 连续情感坐标 (Valence, Arousal) 映射为 1024 维情感嵌入。

    架构: 2 → 64 (ReLU) → 128 (ReLU) → 1024
    参数: ~132K
    """

    def __init__(self, hidden_size: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_size),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, valence: torch.Tensor, arousal: torch.Tensor) -> torch.Tensor:
        """输入: valence, arousal: (batch,) 或 (batch, 1), 范围 0~1
           返回: (batch, hidden_size)
        """
        va = torch.stack([valence.view(-1), arousal.view(-1)], dim=1)
        return self.net(va)


# ---------------------------------------------------------------------------
# 2. Decoder 层注入器（v3 核心）
# ---------------------------------------------------------------------------
class DecoderEmotionInjector(nn.Module):
    """从 V-A 生成 24 层的情感 bias，通过 forward hook 注入 decoder 每层输出。

    不修改 encoder 输出，不碰 cross-attention 输入。
    每层独立学习最佳调制幅度。
    """

    def __init__(self, hidden_size: int = 1024, num_layers: int = 24):
        super().__init__()
        self.num_layers = num_layers
        # 每层一个独立的投影：1024 → hidden_size
        self.layer_biases = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Tanh(),  # Tanh 限制幅度，防止扰动过大
            )
            for _ in range(num_layers)
        ])
        # 当前 batch 的情感嵌入（由 set_emotion 设置，hook 中读取）
        self._current_emotion = None

    def set_emotion(self, emotion_emb: torch.Tensor):
        """设置当前情感嵌入，hook 函数会在前向时读取。"""
        self._current_emotion = emotion_emb

    def _make_hook(self, layer_idx: int):
        """为指定 decoder 层创建 forward hook 函数。"""
        scale = 0.01  # 极小扰动，24 层累计不超 0.24
        def hook(module, input, output):
            if self._current_emotion is None or output is None or not isinstance(output, torch.Tensor):
                return output
            if output.dim() < 3:
                return output
            bias = self.layer_biases[layer_idx](self._current_emotion) * scale
            output = output + bias.unsqueeze(1).to(output.device)
            return output
        return hook

    def register_hooks(self, decoder_layers):
        """在 decoder 的每一层注册 forward hook。"""
        self._hooks = []
        for i, layer in enumerate(decoder_layers):
            h = layer.register_forward_hook(self._make_hook(i))
            self._hooks.append(h)

    def remove_hooks(self):
        """移除所有 hook。"""
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def forward(self, emotion_emb: torch.Tensor) -> list[torch.Tensor]:
        """生成所有层的 bias（训练时使用）。"""
        return [self.layer_biases[i](emotion_emb) for i in range(self.num_layers)]


# ---------------------------------------------------------------------------
# 3. 构建完整模型（冻结 backbone + 注册 hook）
# ---------------------------------------------------------------------------
def build_model(
    model_name: str = "facebook/musicgen-small",
    device: str | None = None,
    freeze_backbone: bool = True,
):
    """加载 MusicGen，冻结 backbone，创建 DecoderEmotionInjector 并注册 hook。"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[build_model] 加载 {model_name} 到 {device} ...")
    model = MusicgenForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    model.eval()

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        print("[build_model] backbone 已冻结 ❄️")

    hidden_size = model.config.decoder.hidden_size  # 1024
    # 获取 decoder 层
    decoder_layers = model.decoder.model.decoder.layers
    num_layers = len(decoder_layers)

    va_encoder = VAEncoder(hidden_size=hidden_size)
    injector = DecoderEmotionInjector(hidden_size=hidden_size, num_layers=num_layers)
    injector.register_hooks(decoder_layers)

    va_encoder.to(device)
    injector.to(device)

    # 统计参数量
    va_params = sum(p.numel() for p in va_encoder.parameters())
    inj_params = sum(p.numel() for p in injector.parameters())
    total_bn = sum(p.numel() for p in model.parameters())
    print(f"[build_model] VAEncoder: {va_params:,}  |  Injector: {inj_params:,}")
    print(f"[build_model] 可训练: {va_params + inj_params:,}  /  总计: {total_bn:,}  "
          f"({(va_params + inj_params) / total_bn * 100:.2f}%)")
    print(f"[build_model] Hook 已注册到 {num_layers} 层 decoder")

    return model, va_encoder, injector


# ---------------------------------------------------------------------------
# 4. 带情感条件的前向传播（训练用）
# ---------------------------------------------------------------------------
def forward_with_emotion(
    model: MusicgenForConditionalGeneration,
    va_encoder: VAEncoder,
    injector: DecoderEmotionInjector,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.LongTensor,
    valence: torch.Tensor,
    arousal: torch.Tensor,
    labels: torch.LongTensor | None = None,
):
    """训练前向：设置情感 → hook 自动注入 decoder 每层输出。

    encoder 路径完全走原生代码，不做任何修改。
    """
    # 1. 文本编码（原生路径）
    encoder_outputs = model.text_encoder(
        input_ids=input_ids, attention_mask=attention_mask, return_dict=True,
    )
    encoder_hidden_states = encoder_outputs.last_hidden_state
    encoder_hidden_states = model.enc_to_dec_proj(encoder_hidden_states)

    # 2. 设置情感（hook 会在 decoder 前向时自动调用）
    emotion_emb = va_encoder(valence, arousal)
    injector.set_emotion(emotion_emb)

    # 3. Decoder（原生路径，hook 自动注入情感）
    decoder_outputs = model.decoder(
        input_ids=decoder_input_ids,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=attention_mask,
        labels=labels,
    )
    return decoder_outputs


# ---------------------------------------------------------------------------
# 5. 音频生成（推理用）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_emotion(
    model: MusicgenForConditionalGeneration,
    va_encoder: VAEncoder,
    injector: DecoderEmotionInjector,
    text_prompt: str | list[str],
    valence: float | list[float],
    arousal: float | list[float],
    duration: float = 8.0,
    guidance_scale: float = 3.0,
    device: str | None = None,
    generate_scale: float = 1.0,  # 保留接口兼容性
):
    """推理生成：设置情感 → model.generate()（原生路径，hook 自动生效）。"""
    from transformers import AutoProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

    if isinstance(text_prompt, str):
        text_prompt = [text_prompt]
    if isinstance(valence, (int, float)):
        valence = [valence] * len(text_prompt)
    if isinstance(arousal, (int, float)):
        arousal = [arousal] * len(text_prompt)

    inputs = processor(text=text_prompt, padding=True, return_tensors="pt").to(device)
    val_t = torch.tensor(valence, device=device, dtype=torch.float)
    aro_t = torch.tensor(arousal, device=device, dtype=torch.float)

    # 设置情感（generate 内部调用 decoder 层时 hook 自动生效）
    emotion_emb = va_encoder(val_t, aro_t)
    injector.set_emotion(emotion_emb)

    # 完全走原生 generate，不做任何 encoder 修改
    audio_values = model.generate(
        **inputs,
        do_sample=True,
        guidance_scale=guidance_scale,
        max_new_tokens=int(256 * duration / 5),
    )

    audio = audio_values[0].cpu()
    sampling_rate = model.config.audio_encoder.sampling_rate
    return audio, sampling_rate


# ---------------------------------------------------------------------------
# 6. 快速测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("emotion_adapter.py — 快速测试")
    print("=" * 60)

    model, va_encoder, adapter = build_model()

    batch_size = 2
    device = next(model.parameters()).device

    # 模拟 text input
    dummy_input_ids = torch.randint(0, 32000, (batch_size, 8), device=device)
    dummy_attn_mask = torch.ones_like(dummy_input_ids)

    # 模拟 decoder input (4 codebooks * batch)
    dummy_decoder_ids = torch.randint(0, 2048, (batch_size * 4, 16), device=device)
    dummy_labels = torch.randint(0, 2048, (batch_size, 16, 4), device=device)

    # 模拟 V-A
    val = torch.tensor([0.8, 0.2], device=device, dtype=torch.float)
    aro = torch.tensor([0.7, 0.3], device=device, dtype=torch.float)

    # 测试前向
    outputs = forward_with_emotion(
        model, va_encoder, adapter,
        input_ids=dummy_input_ids,
        attention_mask=dummy_attn_mask,
        decoder_input_ids=dummy_decoder_ids,
        valence=val,
        arousal=aro,
        labels=dummy_labels,
    )

    print(f"\nLoss: {outputs.loss.item():.4f}")
    print(f"Logits 形状: {outputs.logits.shape}")
    print("✅ 前向传播成功！")

    # 测试生成
    print("\n--- 测试 generate ---")
    audio, sr = generate_with_emotion(
        model, va_encoder, adapter,
        text_prompt="calm piano melody",
        valence=0.8,
        arousal=0.2,
        duration=4.0,
    )
    print(f"生成音频形状: {audio.shape}, 采样率: {sr}")
    print("✅ 生成成功！")
