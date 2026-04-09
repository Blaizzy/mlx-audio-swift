# OmniVoice 测试指南

本文档介绍如何测试 `mlx-community/OmniVoice-bf16` 模型。

## 前置条件

1. **依赖项已下载**：确保所有 Swift 包依赖项已完全下载
2. **网络连接**：测试需要从 HuggingFace 下载模型权重（约 1.23 GB）
3. **磁盘空间**：模型缓存需要约 2-3 GB 空间
4. **内存**：建议至少 8 GB RAM

## 测试方式

### 方式 1：快速集成测试（推荐）

运行配置和工厂测试，不需要下载模型：

```bash
swift test --filter OmniVoiceConfigTests
swift test --filter OmniVoiceFactoryTests
```

这些测试验证：
- ✅ JSON 配置解析
- ✅ 生成参数默认值和预设
- ✅ TTS 工厂注册

### 方式 2：完整模型测试

运行需要下载模型的完整测试：

```bash
MLXAUDIO_ENABLE_NETWORK_TESTS=1 swift test --filter OmniVoiceModelTests
```

这将测试：
- ✅ 模型加载
- ✅ 自动语音生成
- ✅ 语音设计生成
- ✅ 语音克隆生成
- ✅ 流式生成

### 方式 3：CLI 工具测试

使用提供的快速测试脚本：

```bash
./quick_test_omnivoice.sh
```

这会测试三种模式并生成 WAV 文件：
- 自动语音模式
- 语音设计模式（使用 "male, British accent" 指令）
- 语音克隆模式（使用测试音频）

### 方式 4：手动 CLI 测试

```bash
# 自动语音
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a test." \
    --output /tmp/test_auto.wav

# 语音设计
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a test." \
    --instruct "male, British accent" \
    --output /tmp/test_design.wav

# 语音克隆
swift run mlx-audio-swift-tts \
    --model mlx-community/OmniVoice-bf16 \
    --text "Hello, this is a test." \
    --ref_audio Tests/media/intention.wav \
    --ref_text "intention" \
    --output /tmp/test_clone.wav

# 播放生成的音频（macOS）
afplay /tmp/test_auto.wav
afplay /tmp/test_design.wav
afplay /tmp/test_clone.wav
```

## 测试输出

### 成功的测试应该显示：

```
◇ Starting test suite OmniVoice Configuration Tests
◇ Test OmniVoiceConfigTests.testConfigDecodesFromJSON started
✔ Test OmniVoiceConfigTests.testConfigDecodesFromJSON passed (0.XXs)
◇ Test OmniVoiceConfigTests.testGenerateParametersDefaults started
✔ Test OmniVoiceConfigTests.testGenerateParametersDefaults passed (0.XXs)
...

◇ Starting test suite OmniVoice Model Tests
Loading OmniVoice model from mlx-community/OmniVoice-bf16...
Model loaded successfully with sample rate: 24000
Generating audio for: Hello, this is a test of OmniVoice text-to-speech.
Generated XXXXX samples in XX.XXs
Audio duration: X.XXs
Saved audio to: /var/folders/.../omnivoice_auto_voice_test.wav
✔ Test OmniVoiceModelTests.testAutoVoiceGeneration passed (XX.XXs)
```

### 生成的文件

测试会在临时目录生成以下文件：
- `omnivoice_auto_voice_test.wav` - 自动语音测试音频
- `omnivoice_voice_design_test.wav` - 语音设计测试音频
- `omnivoice_voice_cloning_test.wav` - 语音克隆测试音频
- `omnivoice_streaming_test.wav` - 流式生成测试音频

## 自定义测试参数

### 环境变量

```bash
# 使用不同的模型仓库
export MLXAUDIO_OMNIVOICE_REPO="k2-fsa/OmniVoice"

# 启用网络测试
export MLXAUDIO_ENABLE_NETWORK_TESTS=1
```

### 测试单个功能

```bash
# 只测试自动语音
swift test --filter "OmniVoiceModelTests/testAutoVoiceGeneration()"

# 只测试语音设计
swift test --filter "OmniVoiceModelTests/testVoiceDesignGeneration()"

# 只测试语音克隆
swift test --filter "OmniVoiceModelTests/testVoiceCloningGeneration()"

# 只测试流式生成
swift test --filter "OmniVoiceModelTests/testStreamingGeneration()"
```

## 故障排除

### 问题 1：依赖项下载失败

```bash
# 清理并重新下载
rm -rf .build
swift package resolve
```

### 问题 2：模型下载失败

```bash
# 检查 HuggingFace 缓存
ls -la ~/.cache/huggingface/hub/

# 手动下载模型
pip install huggingface-hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/OmniVoice-bf16')"
```

### 问题 3：内存不足

```bash
# 减少并发测试
swift test --filter OmniVoiceTests --parallel-workers 1

# 或使用更少的生成步骤
export MLXAUDIO_TEST_NUM_STEP=16
```

### 问题 4：测试超时

```bash
# 增加测试超时时间
swift test --filter OmniVoiceTests --test-timeout 600
```

## 性能基准

在 Apple Silicon (M1/M2/M3) 上的预期性能：

| 模式 | num_step | 预期 RTF | 说明 |
|------|----------|----------|------|
| 快速 | 16 | ~0.05 (20x) | 适合快速原型 |
| 默认 | 32 | ~0.025 (40x) | 质量和速度平衡 |
| 高质量 | 64 | ~0.05 (20x) | 最佳质量 |

RTF = Real Time Factor（实时因子），值越小表示生成越快。

## 验证清单

测试完成后，验证：

- [ ] 配置解析测试通过
- [ ] 生成参数测试通过
- [ ] TTS 工厂注册测试通过
- [ ] 模型成功加载（如果启用网络测试）
- [ ] 生成的音频文件可以播放
- [ ] 音频质量可接受
- [ ] 三种模式都能正常工作
- [ ] 流式生成正常工作（可选）

## 报告问题

如果测试失败，请提供：

1. 完整的测试输出
2. 系统信息（macOS 版本、芯片类型）
3. 模型缓存状态
4. 生成的音频文件（如果有）

```bash
# 收集诊断信息
sw_vers
uname -m
swift --version
ls -la ~/.cache/huggingface/hub/models--mlx-community--OmniVoice-bf16/ 2>/dev/null || echo "Model not cached"
```
