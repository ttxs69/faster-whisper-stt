#!/usr/bin/env python3
"""Faster-Whisper STT HTTP API Server

一个简单的 FastAPI 服务，将 faster-whisper 包装成 HTTP API，
供 OpenClaw QQBot 插件等场景调用。

用法:
    uv run server.py [--host 0.0.0.0] [--port 8080] [--model tiny]

API 端点:
    POST /v1/audio/transcriptions
    - 兼容 OpenAI Whisper API 格式
    - 支持 file 上传或 url 参数
"""

import argparse
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI(title="Faster-Whisper STT API", version="0.1.0")

# 全局模型实例（延迟加载）
_model = None
_model_name = "tiny"
_device = "cpu"
_compute_type = "int8"


def get_model():
    """延迟加载模型"""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print(f"📦 加载模型: {_model_name} ({_device}, {_compute_type})...")
        start = time.time()
        _model = WhisperModel(_model_name, device=_device, compute_type=_compute_type)
        print(f"✅ 模型加载完成 ({time.time() - start:.2f}s)")
    return _model


class TranscriptionResponse(BaseModel):
    """OpenAI 兼容的响应格式"""
    text: str


class TranscriptionResult(BaseModel):
    """详细结果（包含时间戳）"""
    text: str
    language: str
    language_probability: float
    duration: float
    segments: list[dict]


@app.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile | None = File(None),
    url: str | None = Form(None),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
):
    """
    转写音频文件（OpenAI Whisper API 兼容）
    
    - file: 上传的音频文件
    - url: 音频文件 URL（可选，与 file 二选一）
    - model: 模型名（兼容性参数，实际使用服务端配置的模型）
    - language: 语言代码（如 zh, en），不指定则自动检测
    - response_format: 响应格式（json, verbose_json）
    """
    if file is None and url is None:
        raise HTTPException(status_code=400, detail="需要提供 file 或 url 参数")
    
    # 获取音频数据
    if file:
        audio_data = await file.read()
        filename = file.filename or "audio.bin"
    else:
        # 从 URL 下载
        import httpx
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            audio_data = resp.content
            filename = url.split("/")[-1] or "audio.bin"
    
    # 保存到临时文件
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
        tmp.write(audio_data)
        tmp_path = tmp.name
    
    try:
        # 转写
        whisper_model = get_model()
        start = time.time()
        segments, info = whisper_model.transcribe(
            tmp_path,
            beam_size=5,
            language=language,
        )
        
        # 收集结果
        text_parts = []
        segment_list = []
        for seg in segments:
            text_parts.append(seg.text.strip())
            segment_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
            })
        
        full_text = " ".join(text_parts)
        transcribe_time = time.time() - start
        
        print(f"🎤 转写完成: {info.duration:.2f}s 音频, {transcribe_time:.2f}s 处理, {info.duration/transcribe_time:.2f}x")
        
        if response_format == "verbose_json":
            return JSONResponse(content={
                "text": full_text,
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "segments": segment_list,
            })
        else:
            return TranscriptionResponse(text=full_text)
    
    finally:
        # 清理临时文件
        Path(tmp_path).unlink(missing_ok=True)


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "model": _model_name}


def main():
    global _model_name, _device, _compute_type
    
    parser = argparse.ArgumentParser(description="Faster-Whisper STT HTTP API")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8080, help="监听端口")
    parser.add_argument("--model", "-m", default="small",
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en",
                                 "medium", "medium.en", "large-v3", "distil-large-v3"],
                        help="模型大小 (默认: small)")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"],
                        help="设备 (默认: cpu)")
    parser.add_argument("--compute-type", "-c", default="int8",
                        choices=["int8", "float16", "int8_float16"],
                        help="计算精度 (默认: int8)")
    
    args = parser.parse_args()
    
    _model_name = args.model
    _device = args.device
    _compute_type = args.compute_type
    
    import uvicorn
    print(f"🚀 启动 Faster-Whisper STT API 服务")
    print(f"   地址: http://{args.host}:{args.port}")
    print(f"   模型: {args.model} ({args.device}, {args.compute_type})")
    print(f"   端点: POST /v1/audio/transcriptions")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
