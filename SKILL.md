---
name: faster-whisper-stt
description: Local speech-to-text transcription using Faster-Whisper with CTranslate2 backend. Supports multiple models (tiny/base/small/medium/large-v3) and INT8 quantization for CPU efficiency. Use when user needs to transcribe audio files, voice messages, or any speech-to-text task. Works with mp3, wav, m4a, and other common audio formats.
---

# Faster-Whisper STT

Fast local speech-to-text using Faster-Whisper with CTranslate2, up to 4x faster than original OpenAI Whisper.

## Quick Start

```python
from faster_whisper import WhisperModel

# CPU + INT8 (推荐低资源环境)
model = WhisperModel("small", device="cpu", compute_type="int8")

# 转写
segments, info = model.transcribe("audio.mp3", beam_size=5, language="zh")

print(f"语言: {info.language} ({info.language_probability:.2%})")
for seg in segments:
    print(seg.text)
```

## Model Selection

| 模型 | 参数量 | 精度 | 适用场景 |
|------|--------|------|----------|
| `tiny` | 39M | 快速、低资源 | 实时转写、低精度场景 |
| `base` | 74M | 平衡 | 通用场景 |
| `small` | 244M | 较高精度 | 准确率要求较高 |
| `medium` | 769M | 高精度 | 专业转写 |
| `large-v3` | 1.5B | 最高精度 | 最高质量要求 |

## Compute Types

```python
# CPU + INT8 (最省内存)
model = WhisperModel("small", device="cpu", compute_type="int8")

# GPU + FP16 (最快)
model = WhisperModel("small", device="cuda", compute_type="float16")

# GPU + INT8 (显存优化)
model = WhisperModel("small", device="cuda", compute_type="int8_float16")
```

## Common Options

```python
segments, info = model.transcribe(
    "audio.mp3",
    beam_size=5,              # 束搜索大小，越大越准但越慢
    language="zh",            # 指定语言，None 自动检测
    word_timestamps=True,     # 词级时间戳
    vad_filter=True,          # 静音过滤
)
```

## Script

使用封装好的脚本快速转写：

```bash
uv run scripts/transcribe.py <audio_file> [--model small] [--language zh]
```

## Performance (CPU, small + INT8)

- 模型加载: ~2-3s
- 实时倍率: ~8-10x
- 内存占用: ~300MB

## Installation

本技能自带 `pyproject.toml`，使用 uv 一键安装依赖：

```bash
cd ~/.openclaw/workspace/skills/faster-whisper-stt
uv sync
```

安装完成后即可使用：

```bash
uv run scripts/transcribe.py <audio_file> --language zh
```

### GPU 支持（可选）

如有 NVIDIA GPU，可安装 CUDA 库加速：

```bash
uv add nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
```

然后使用 `--device cuda` 参数：

```bash
uv run scripts/transcribe.py <audio_file> --device cuda --compute-type float16
```
