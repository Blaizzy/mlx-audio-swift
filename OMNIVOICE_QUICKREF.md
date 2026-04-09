# OmniVoice 快速参考

## 🚀 快速开始

### 1️⃣ 自动语音（最简单）
```bash
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is OmniVoice text-to-speech." \
    --output output.wav

afplay output.wav
```

### 2️⃣ 语音设计
```bash
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a voice design test." \
    --instruct "male, British accent" \
    --output voice_design.wav

afplay voice_design.wav
```

### 3️⃣ 语音克隆
```bash
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "This is a voice cloning test." \
    --ref_audio ref.wav \
    --ref_text "Reference transcript" \
    --output clone.wav

afplay clone.wav
```

## 🎛️ 重要参数

### 质量控制
```bash
# 快速生成（低质量）
--num_step 16 --guidance_scale 1.5

# 默认（平衡）
--num_step 32 --guidance_scale 2.0

# 高质量
--num_step 64 --guidance_scale 2.5
```

### 语速控制
```bash
# 慢速
--speed 0.8

# 正常
--speed 1.0

# 快速
--speed 1.2
```

### 固定时长
```bash
# 固定 5 秒输出
--duration 5.0
```

## 🧪 测试

### 快速验证
```bash
./validate_omnivoice.sh
```

### 运行测试
```bash
# 基础测试（无需下载模型）
swift test --filter OmniVoiceConfigTests

# 完整测试（需要网络）
MLXAUDIO_ENABLE_NETWORK_TESTS=1 swift test --filter OmniVoiceTests

# 快速测试脚本
./quick_test_omnivoice.sh
```

## 💻 Swift 代码使用

```swift
import MLXAudioTTS
import MLXAudioCore

// 加载模型
let model = try await TTS.loadModel(modelRepo: "mlx-community/OmniVoice-bf16")

// 配置
if let ov = model as? OmniVoiceModel {
    ov.setGenerationConfig(
        numStep: 32,
        guidanceScale: 2.0,
        speed: 1.0
    )
}

// 生成
let audio = try await model.generate(
    text: "Hello world",
    voice: "male, British accent",
    refAudio: nil,
    refText: nil,
    language: "English",
    generationParameters: GenerateParameters(
        maxTokens: 2048,
        temperature: 1.0,
        topP: 0.95
    )
)

// 保存
try AudioUtils.writeWavFile(
    samples: audio.asArray(Float.self),
    sampleRate: model.sampleRate,
    fileURL: URL(fileURLWithPath: "output.wav")
)
```

## 📊 性能参考

| 配置 | num_step | 预期 RTF | 说明 |
|------|----------|----------|------|
| 快速 | 16 | ~0.05 | 20x 实时 |
| 默认 | 32 | ~0.025 | 40x 实时 |
| 高质量 | 64 | ~0.05 | 20x 实时 |

## 🔍 故障排除

### 模型下载慢
```bash
# 检查缓存
ls -la ~/.cache/huggingface/hub/

# 使用环境变量指定仓库
export MLXAUDIO_OMNIVOICE_REPO="k2-fsa/OmniVoice"
```

### 内存不足
```bash
# 减少并发
swift test --parallel-workers 1

# 使用更少的扩散步数
--num_step 16
```

### 测试超时
```bash
swift test --filter OmniVoiceTests --test-timeout 600
```

## 📖 完整文档

- 实现详情: `OMNIVOICE_IMPLEMENTATION.md`
- 测试指南: `TESTING_OMNIVOICE.md`
- 模型文档: `Sources/MLXAudioTTS/Models/OmniVoice/README.md`

## 🎯 语音设计指令示例

```bash
# 男性声音
--instruct "male voice"

# 女性英国口音
--instruct "female, British accent"

# 儿童声音
--instruct "child, cheerful"

# 老年男性
--instruct "elderly male, deep voice"

# 特定情感
--instruct "male, happy and excited"

# 专业语气
--instruct "female, professional and calm"

# 特定口音
--instruct "male, American accent"
--instruct "female, Australian accent"
```

## ✨ 提示

1. **首次使用**会下载约 1.23GB 模型文件
2. **模型缓存**在 `~/.cache/huggingface/hub/`
3. **输出采样率**为 24kHz（高于普通 TTS）
4. **生成速度**约 40x 实时（默认设置）
5. **使用 `--benchmark`** 查看性能指标
