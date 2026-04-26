# OmniVoice 集成状态记录

## 已修复的问题

### 1. decode 路径遗漏 `fc2.bias`
**文件**: `Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift`

之前的 fc2 修复只做了裸矩阵乘法 `matmul(zNLC, fc2.weight.T)`，但 `fc2` 是完整的 `nn.Linear(1024, 256)`，包含 shape 为 `[256]` 的 bias。遗漏 bias 会导致解码特征偏移，输出失真。

**修复**: 在 `OmniVoiceAudioTokenizer.decode()` 中补上 bias：
```swift
let h = (hNLC + (fc2.bias ?? MLXArray.zeros([1]))).transposed(0, 2, 1)
```

### 2. `acousticDecoder` 末尾多余的 `tanh`
**文件**: `Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift`

Python 参考实现在 `_adjust_dac_decoder()` 中明确移除了 decoder 最后一层的 `nn.Tanh`，替换为 `nn.Identity()`。Swift 实现保留了 `MLX.tanh(h)`，导致输出被错误压缩到 `[-1, 1]`，与训练时的分布不一致。

**修复**: 移除了 `OmniVoiceDACAcousticDecoder.callAsFunction()` 末尾的 `tanh`。

---

## 当前限制（Checkpoint 相关）

### `mlx-community/OmniVoice-bf16` 的 tokenizer 是精简版

`audio_tokenizer/model.safetensors` 中**缺少以下权重**：
- `semantic_model.*` — HuBERT 模型（用于提取语义特征）
- `encoder_semantic.*` / `decoder_semantic.*` — 语义编/解码器
- `fc.weight` / `fc.bias` — acoustic + semantic 融合投影层
- `fc1.*`

当前 checkpoint 仅包含：
- `acoustic_encoder.*`
- `acoustic_decoder.*`
- `quantizer.*`
- `fc2.*`

### 影响
- **Auto Voice / Voice Design（无参考音频）**：不受影响，因为只走 `decode` 路径。
- **Voice Cloning（带参考音频）**：**无法完整实现**。`encode()` 需要同时提取 acoustic 和 semantic 特征并拼接，但当前 checkpoint 缺少 semantic 路径的所有权重，encode 出来的 token 与官方实现不一致。

---

## 后续建议

### 如需支持 Voice Cloning
1. **换用完整 Checkpoint**
   - 官方模型：`k2-fsa/OmniVoice`
   - 或 fallback tokenizer：`eustlb/higgs-audio-v2-tokenizer`

2. **在 Swift 中补齐 Semantic 链路**
   需要实现以下模块：
   - **HuBERT 特征提取**：重采样到 16kHz → pad(160,160) → 过 HuBERT → 取所有 hidden_states 的平均值
   - **SemanticEncoder**：Conv1d + 2x SemanticEncoderBlock（ELU + dilated Conv1d + 1x1 Conv 残差）
   - **融合层 `fc`**：`Linear(1024, 1024)`，将 acoustic(256) + semantic(768) 的拼接结果投影到 quantizer 输入维度
   - 修改 `OmniVoiceAudioTokenizer.encode()` 为完整的 `acoustic + semantic → concat → fc → quantizer` 流程

---

## 相关文件
- `Sources/MLXAudioTTS/Models/OmniVoice/OmniVoice.swift` — 模型实现
- `Sources/MLXAudioTTS/Models/OmniVoice/OmniVoiceConfig.swift` — 配置定义
- `OMNIVOICE_FIX.md` — 之前的 fc2 修复记录

*记录时间: 2026-04-15*
