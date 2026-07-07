# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import struct
import wave
from pathlib import Path
from urllib.parse import urlencode

import websockets
from stream_client import iter_frames

_ASR_HEADER = struct.Struct(">iii")


def _realtime_url(server: str, session_id: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    query = urlencode({"session_id": session_id})
    return f"{base}/v1/realtime?{query}"


def _read_wav_pcm(path: str) -> bytes:
    with wave.open(path, "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("query WAV must be mono 16-bit PCM")
        return wav.readframes(wav.getnframes())


def _asr_text(payload: dict) -> str:
    response = payload.get("asr_response") or {}
    result = response.get("recognition_result") or {}
    hypotheses = result.get("hypothesis") or []
    if not hypotheses:
        return ""
    return hypotheses[0].get("text") or ""


async def _transcribe_query(asr_url: str, wav_path: str) -> str:
    async with websockets.connect(asr_url, max_size=None) as ws:
        await ws.send(_ASR_HEADER.pack(-1, 0, 0) + _read_wav_pcm(wav_path))
        async for message in ws:
            if not isinstance(message, str):
                continue
            text = _asr_text(json.loads(message))
            if text:
                return text
    return ""


async def _synthesize(tts_url: str, voice: str, text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(tts_url, max_size=None) as ws:
        await ws.send(json.dumps({"config": {"voice": voice}}))
        await ws.send(json.dumps({"type": "input_text.append", "text": text}))
        await ws.send(json.dumps({"type": "input_text.commit"}))
        async for message in ws:
            if isinstance(message, bytes):
                with output_path.open("ab") as out:
                    out.write(message)
                continue
            event = json.loads(message)
            if event.get("type") == "response.done":
                return
            if event.get("type") == "error":
                print("tts error", flush=True)
                return


async def _tts_worker(tts_url: str, voice: str, output_path: Path, queue: asyncio.Queue[str | None]) -> None:
    while True:
        text = await queue.get()
        if text is None:
            return
        await _synthesize(tts_url, voice, text, output_path)


async def _receive(ws, tts_queue: asyncio.Queue[str | None] | None = None) -> None:
    async for message in ws:
        event = json.loads(message)
        etype = event.get("type")
        if etype == "response.delta":
            text = event.get("data", "")
            print(text, flush=True)
            if tts_queue is not None and text:
                await tts_queue.put(text)
        elif etype == "error":
            print(f"error: {event.get('message', '')}", flush=True)


async def run(
    video: str,
    server: str,
    session_id: str,
    query: str | None,
    fps: float,
    *,
    asr_url: str | None = None,
    query_wav: str | None = None,
    tts_url: str | None = None,
    tts_out: str | None = None,
    voice: str = "vivian",
) -> None:
    if query_wav:
        if not asr_url:
            raise ValueError("--query-wav requires --asr-url")
        query = await _transcribe_query(asr_url, query_wav)
        print(f"ASR query: {query}", flush=True)

    tts_queue: asyncio.Queue[str | None] | None = None
    tts_task: asyncio.Task | None = None
    if tts_url:
        if not tts_out:
            raise ValueError("--tts-url requires --tts-out")
        tts_queue = asyncio.Queue()
        tts_task = asyncio.create_task(_tts_worker(tts_url, voice, Path(tts_out), tts_queue))

    try:
        async with websockets.connect(_realtime_url(server, session_id)) as ws:
            recv_task = asyncio.create_task(_receive(ws, tts_queue))
            if query:
                await ws.send(json.dumps({"type": "input.append", "modality": "text", "data": query}))

            delay = 1.0 / fps if fps > 0 else 0.0
            for _, jpeg in iter_frames(video, fps):
                data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
                await ws.send(json.dumps({"type": "input.append", "modality": "video", "data": data_url}))
                if delay:
                    await asyncio.sleep(delay)

            await ws.send(json.dumps({"type": "close"}))
            await recv_task
    finally:
        if tts_queue is not None:
            await tts_queue.put(None)
        if tts_task is not None:
            await tts_task


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL realtime full-duplex websocket demo")
    parser.add_argument("video")
    parser.add_argument("--server", default="http://127.0.0.1:8070")
    parser.add_argument("--query", default=None)
    parser.add_argument("--query-wav", default=None, help="mono 16-bit PCM WAV sent to the ASR bridge as the query")
    parser.add_argument("--asr-url", default=None, help="ASR bridge websocket, e.g. ws://127.0.0.1:8093/v1/asr")
    parser.add_argument("--tts-url", default=None, help="TTS bridge websocket, e.g. ws://127.0.0.1:8092/v1/tts")
    parser.add_argument("--tts-out", default=None, help="append synthesized PCM bytes to this file")
    parser.add_argument("--voice", default="vivian")
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--session-id", default="realtime-cli")
    args = parser.parse_args()

    asyncio.run(
        run(
            args.video,
            args.server,
            args.session_id,
            args.query,
            args.fps,
            asr_url=args.asr_url,
            query_wav=args.query_wav,
            tts_url=args.tts_url,
            tts_out=args.tts_out,
            voice=args.voice,
        )
    )


if __name__ == "__main__":
    main()
