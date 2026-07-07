# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json

import pytest
from starlette.testclient import TestClient

from vllm_omni.experimental.fullduplex.core import protocol as ev
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SlowBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict]] = []

    async def generate(self, *args, **kwargs):
        self.calls += 1
        self.messages.append(args[0])
        call = self.calls
        if call == 1:
            await asyncio.sleep(0.2)
            return "</response> stale", None
        return "</response> fresh", None

    async def aclose(self) -> None:
        pass


def test_realtime_websocket_barge_in_cancels_stale_response(monkeypatch):
    from vllm_omni.experimental.fullduplex.joyvl.serving import server as server_mod

    backend = _SlowBackend()
    monkeypatch.setattr(server_mod, "OpenAIBackend", lambda *args, **kwargs: backend)
    app = server_mod.create_app(
        InteractionConfig(enable_memory=False, enable_delegation=False, force_silence_before_query=False)
    )

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?session_id=ws-test") as ws:
            ws.send_json({"type": ev.INPUT_APPEND, "modality": "text", "data": "watch the frame"})
            ws.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,AAA"})
            first_created = ws.receive_json()
            assert first_created["type"] == ev.RESPONSE_CREATED
            assert first_created["response_index"] == 1

            ws.send_json({"type": ev.INPUT_APPEND, "modality": "text", "data": "switch to fresh query"})
            ws.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,BBB"})

            events = [first_created]
            for _ in range(8):
                event = ws.receive_json()
                events.append(event)
                if event["type"] == ev.RESPONSE_DONE:
                    break

            assert [e["response_index"] for e in events if e["type"] == ev.RESPONSE_CREATED] == [1, 2]
            assert [e["data"] for e in events if e["type"] == ev.RESPONSE_DELTA] == ["fresh"]
            assert events[-1]["type"] == ev.RESPONSE_DONE
            assert events[-1]["response_index"] == 2
            assert "switch to fresh query" in json.dumps(backend.messages[-1])
