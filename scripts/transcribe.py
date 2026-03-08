#!/usr/bin/env python3
"""Faster-Whisper 语音转写脚本

用法:
    uv run transcribe.py <audio_file> [--model tiny] [--language zh] [--output result.txt]
"""

import argparse
import sys
import time
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Faster-Whisper 语音转写")
    parser.add_argument("audio", help="音频文件路径")
    parser.add_argument("--model", "-m", default="tiny", 
                        choices=["tiny", "tiny.en", "base", "base.en", "small", "small.en", 
                                 "medium", "medium.en", "large-v3", "distil-large-v3"],
                        help="模型大小 (默认: tiny)")
    parser.add_argument("--language", "-l", default=None,
                        help="指定语言代码 (如 zh, en)，不指定则自动检测")
    parser.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"],
                        help="设备 (默认: cpu)")
    parser.add_argument("--compute-type", "-c", default="int8",
                        choices=["int8", "float16", "int8_float16"],
                        help="计算精度 (默认: int8)")
    parser.add_argument("--beam-size", "-b", type=int, default=5,
                        help="束搜索大小 (默认: 5)")
    parser.add_argument("--word-timestamps", "-w", action="store_true",
                        help="输出词级时间戳")
    parser.add_argument("--output", "-o", default=None,
                        help="输出文件路径，不指定则打印到终端")
    parser.add_argument("--vad", action="store_true",
                        help="启用静音过滤")
    
    args = parser.parse_args()
    
    if not Path(args.audio).exists():
        print(f"❌ 文件不存在: {args.audio}", file=sys.stderr)
        sys.exit(1)
    
    # 导入 faster_whisper
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("❌ 未安装 faster-whisper，请运行: uv add faster-whisper", file=sys.stderr)
        sys.exit(1)
    
    print(f"📦 加载模型: {args.model} ({args.device}, {args.compute_type})...")
    start = time.time()
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    load_time = time.time() - start
    print(f"✅ 模型加载完成 ({load_time:.2f}s)")
    
    print(f"🎤 转写中: {args.audio}")
    start = time.time()
    segments, info = model.transcribe(
        args.audio,
        beam_size=args.beam_size,
        language=args.language,
        word_timestamps=args.word_timestamps,
        vad_filter=args.vad,
    )
    
    # 收集结果
    results = []
    for seg in segments:
        if args.word_timestamps:
            words_text = " ".join([w.word for w in seg.words])
            results.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {words_text}")
        else:
            results.append(f"[{seg.start:.2f}s -> {seg.end:.2f}s] {seg.text}")
    
    transcribe_time = time.time() - start
    
    # 输出统计
    print(f"\n📊 统计:")
    print(f"   语言: {info.language} ({info.language_probability:.2%})")
    print(f"   时长: {info.duration:.2f}s")
    print(f"   转写耗时: {transcribe_time:.2f}s")
    print(f"   实时倍率: {info.duration/transcribe_time:.2f}x")
    print("-" * 50)
    
    # 输出结果
    output_text = "\n".join(results)
    
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
        print(f"✅ 结果已保存到: {args.output}")
    else:
        print(output_text)

if __name__ == "__main__":
    main()
