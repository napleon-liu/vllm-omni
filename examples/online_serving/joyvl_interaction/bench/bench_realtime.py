# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import statistics
import time
from dataclasses import dataclass, field
from urllib.parse import urlencode

import websockets

_JPEG_1X1 = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00"
    b"\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b"
    b"\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.' "
    b"\",#\x1c\x1c(7),01444\x1f'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01"
    b"\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00"
    b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10"
    b"\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\xff\xda\x00\x08\x01\x01\x00\x00?\x00T\xdf\xff\xd9"
)


@dataclass
class SessionStats:
    latencies_ms: list[float] = field(default_factory=list)
    errors: int = 0
    completed: int = 0


def _realtime_url(server: str, session_id: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    return f"{base}/v1/realtime?{urlencode({'session_id': session_id})}"


def _image_data_url() -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_JPEG_1X1).decode()


async def _run_session(server: str, session_id: str, frames: int, query: str) -> SessionStats:
    stats = SessionStats()
    image = _image_data_url()
    async with websockets.connect(_realtime_url(server, session_id), max_size=None) as ws:
        await ws.recv()  # session.created
        await ws.send(json.dumps({"type": "input.append", "modality": "text", "data": query}))
        for frame_idx in range(frames):
            started = time.perf_counter()
            await ws.send(json.dumps({"type": "input.append", "modality": "video", "data": image}))
            while True:
                event = json.loads(await ws.recv())
                if event.get("type") == "error":
                    stats.errors += 1
                if event.get("type") == "response.done":
                    stats.completed += 1
                    stats.latencies_ms.append((time.perf_counter() - started) * 1000)
                    break
                if event.get("type") == "response.cancelled":
                    break
        await ws.send(json.dumps({"type": "close"}))
    return stats


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * pct
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    weight = rank - lo
    ordered = sorted(values)
    return ordered[lo] * (1 - weight) + ordered[hi] * weight


async def run(server: str, sessions: int, frames: int, query: str) -> dict:
    started = time.perf_counter()
    results = await asyncio.gather(*[_run_session(server, f"bench-{idx}", frames, query) for idx in range(sessions)])
    wall_time_s = time.perf_counter() - started
    latencies = [latency for result in results for latency in result.latencies_ms]
    completed = sum(result.completed for result in results)
    errors = sum(result.errors for result in results)
    return {
        "sessions": sessions,
        "frames_per_session": frames,
        "completed": completed,
        "errors": errors,
        "wall_time_s": round(wall_time_s, 3),
        "throughput_fps": round(completed / wall_time_s, 3) if wall_time_s else 0.0,
        "latency_ms": {
            "avg": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "p50": round(_percentile(latencies, 0.50), 3),
            "p95": round(_percentile(latencies, 0.95), 3),
            "max": round(max(latencies), 3) if latencies else 0.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark JoyVL realtime websocket serving")
    parser.add_argument("--server", default="http://127.0.0.1:8070")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--query", default="Describe each frame briefly.")
    args = parser.parse_args()

    print(json.dumps(asyncio.run(run(args.server, args.sessions, args.frames, args.query)), indent=2))


if __name__ == "__main__":
    main()
