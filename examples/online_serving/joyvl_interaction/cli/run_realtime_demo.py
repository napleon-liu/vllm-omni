# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import asyncio
import base64
import json
from urllib.parse import urlencode

import websockets
from stream_client import iter_frames


def _realtime_url(server: str, session_id: str) -> str:
    base = server.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://") :]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://") :]
    query = urlencode({"session_id": session_id})
    return f"{base}/v1/realtime?{query}"


async def _receive(ws) -> None:
    async for message in ws:
        event = json.loads(message)
        etype = event.get("type")
        if etype == "response.delta":
            print(event.get("data", ""), flush=True)
        elif etype == "error":
            print(f"error: {event.get('message', '')}", flush=True)


async def run(video: str, server: str, session_id: str, query: str | None, fps: float) -> None:
    async with websockets.connect(_realtime_url(server, session_id)) as ws:
        recv_task = asyncio.create_task(_receive(ws))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="JoyVL realtime full-duplex websocket demo")
    parser.add_argument("video")
    parser.add_argument("--server", default="http://127.0.0.1:8070")
    parser.add_argument("--query", default=None)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--session-id", default="realtime-cli")
    args = parser.parse_args()

    asyncio.run(run(args.video, args.server, args.session_id, args.query, args.fps))


if __name__ == "__main__":
    main()
