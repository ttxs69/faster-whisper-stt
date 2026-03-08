# Faster-Whisper STT

基于 Faster-Whisper + CTranslate2 的本地语音转文字技能，比原版 OpenAI Whisper 快 4 倍。

## 特性

- 🚀 快速转写：CPU 模式下可达 3-8 倍实时速度
- 🌍 多语言支持：99+ 语言自动检测
- 💾 低内存占用：INT8 量化，~300MB 即可运行
- 🎯 多种模型：tiny/base/small/medium/large-v3 可选

## 安装

```bash
cd faster-whisper-stt
uv sync
```

## 使用

### 命令行

```bash
# 基本用法
uv run scripts/transcribe.py audio.wav --language zh

# 指定模型
uv run scripts/transcribe.py audio.wav --model small --language zh

# 输出到文件
uv run scripts/transcribe.py audio.wav --language zh --output result.txt

# 启用静音过滤
uv run scripts/transcribe.py audio.wav --language zh --vad
```

### Python API

```python
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.mp3", language="zh")

for seg in segments:
    print(seg.text)
```

## 模型选择

| 模型 | 参数量 | 适用场景 |
|------|--------|----------|
| `tiny` | 39M | 实时转写、低精度 |
| `base` | 74M | 通用场景 |
| `small` | 244M | 准确率要求较高（推荐） |
| `medium` | 769M | 专业转写 |
| `large-v3` | 1.5B | 最高质量 |

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model` / `-m` | 模型大小 | `tiny` |
| `--language` / `-l` | 语言代码（zh, en 等） | 自动检测 |
| `--device` / `-d` | 设备（cpu/cuda） | `cpu` |
| `--compute-type` / `-c` | 计算精度 | `int8` |
| `--beam-size` / `-b` | 束搜索大小 | `5` |
| `--word-timestamps` / `-w` | 词级时间戳 | 关闭 |
| `--vad` | 静音过滤 | 关闭 |
| `--output` / `-o` | 输出文件 | 终端 |

## 性能参考（CPU, small + INT8）

- 模型加载：~2-3s
- 实时倍率：~3-8x
- 内存占用：~300MB

## License

MIT
