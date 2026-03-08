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

## 🇨🇳 国内镜像配置（ModelScope）

如果在中国无法访问 HuggingFace，可以用镜像：

**方式 1：环境变量（推荐）**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**方式 2：安装 modelscope 自动切换**
```bash
pip install modelscope
```

设置后再运行转写脚本，模型会自动从镜像下载。

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

## HTTP API 服务（OpenAI 兼容）

可以将 faster-whisper 包装成 HTTP API，供 QQBot 插件等场景使用：

```bash
# 启动服务
cd ~/.openclaw/workspace/skills/faster-whisper-stt
uv run server.py --port 8080 --model small

# 后台运行
nohup uv run server.py --port 8080 --model small > /tmp/stt-server.log 2>&1 &
```

### API 端点

- `POST /v1/audio/transcriptions` - 转写音频（OpenAI Whisper API 兼容）
- `GET /health` - 健康检查

### 调用示例

```bash
# 上传文件转写
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "file=@audio.mp3" \
  -F "language=zh"

# 使用 URL 转写
curl -X POST http://localhost:8080/v1/audio/transcriptions \
  -F "url=https://example.com/audio.mp3" \
  -F "language=zh"
```

### 配置 QQBot 插件使用本地 STT

**步骤 1：启动 STT API 服务**

```bash
cd ~/.openclaw/workspace/skills/faster-whisper-stt

# 前台运行（调试用）
uv run server.py --port 8080 --model small

# 后台运行（生产用）
nohup uv run server.py --port 8080 --model small > /tmp/stt-server.log 2>&1 &

# 检查服务状态
curl http://localhost:8080/health
# 应返回: {"status":"ok","model":"small"}
```

**步骤 2：修改 OpenClaw 配置**

在 `~/.openclaw/openclaw.json` 中添加 STT 配置：

```json
{
  "channels": {
    "qqbot": {
      "stt": {
        "provider": "local-whisper",
        "model": "whisper-1"
      }
    }
  },
  "models": {
    "providers": {
      "local-whisper": {
        "baseUrl": "http://localhost:8080/v1",
        "apiKey": "not-needed",
        "models": [
          {
            "id": "whisper-1",
            "name": "whisper-1"
          }
        ]
      }
    }
  }
}
```

**⚠️ 重要注意事项：**

1. **必须添加 `models` 数组**：provider 配置必须包含 `models` 数组，否则会导致 gateway 启动失败
2. **`models.id` 要与 `stt.model` 匹配**：上面例子中都是 `whisper-1`
3. **不影响文字聊天**：STT 配置只处理语音消息，文字消息仍然使用 `agents.defaults.model.primary` 指定的模型

**步骤 3：重启 Gateway**

```bash
openclaw gateway restart
```

**步骤 4：验证 STT 是否生效**

发送一条语音消息，查看日志：

```bash
tail -f /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log | grep -i "stt"
```

如果看到 `STT transcript: ...` 说明 STT 工作正常。

### 开机自启（systemd）

推荐使用 systemd 管理服务，实现开机自启和自动重启：

```bash
# 创建服务文件
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/stt-server.service << 'EOF'
[Unit]
Description=Faster-Whisper STT API Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/root/.openclaw/workspace/skills/faster-whisper-stt
ExecStart=/root/.local/bin/uv run server.py --port 8080 --model small
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF

# 启用并启动服务
systemctl --user daemon-reload
systemctl --user enable stt-server
systemctl --user start stt-server

# 检查状态
systemctl --user status stt-server
```

**管理命令：**

```bash
# 查看状态
systemctl --user status stt-server

# 查看日志
journalctl --user -u stt-server -f

# 重启服务
systemctl --user restart stt-server

# 停止服务
systemctl --user stop stt-server
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
