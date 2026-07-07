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


class _CountingBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def generate(self, *args, **kwargs):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0.02)
            return f"</response> reply {self.calls}", None
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        pass


def _app_with_backend(monkeypatch, backend, **config_kwargs):
    from vllm_omni.experimental.fullduplex.joyvl.serving import server as server_mod

    monkeypatch.setattr(server_mod, "OpenAIBackend", lambda *args, **kwargs: backend)
    config = InteractionConfig(enable_memory=False, enable_delegation=False, **config_kwargs)
    return server_mod.create_app(config)


def test_realtime_websocket_barge_in_cancels_stale_response(monkeypatch):
    backend = _SlowBackend()
    app = _app_with_backend(monkeypatch, backend, force_silence_before_query=False)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?session_id=ws-test") as ws:
            session = ws.receive_json()
            assert session["type"] == ev.SESSION_CREATED
            assert session["session_id"] == "ws-test"
            assert session["input_modalities"] == ["image", "video", "text"]

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


def test_realtime_websocket_rejects_unsupported_modality(monkeypatch):
    app = _app_with_backend(monkeypatch, _CountingBackend(), force_silence_before_query=False)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?session_id=bad-input") as ws:
            assert ws.receive_json()["type"] == ev.SESSION_CREATED
            ws.send_json({"type": ev.INPUT_APPEND, "modality": "audio", "data": "..."})
            event = ws.receive_json()
            assert event["type"] == ev.ERROR
            assert "unsupported input modality: audio" in event["message"]


def test_realtime_websocket_cancel_interrupts_active_response(monkeypatch):
    app = _app_with_backend(monkeypatch, _SlowBackend(), force_silence_before_query=False)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?session_id=cancel-test") as ws:
            assert ws.receive_json()["type"] == ev.SESSION_CREATED
            ws.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": "data:image/jpeg;base64,AAA"})
            created = ws.receive_json()
            assert created["type"] == ev.RESPONSE_CREATED

            ws.send_json({"type": ev.RESPONSE_CANCEL})
            cancelled = ws.receive_json()
            assert cancelled["type"] == ev.RESPONSE_CANCELLED
            assert cancelled["response_index"] == created["response_index"]


def test_realtime_websocket_long_run_completes_repeated_frames(monkeypatch):
    backend = _CountingBackend()
    app = _app_with_backend(monkeypatch, backend, force_silence_before_query=False)

    with TestClient(app) as client:
        with client.websocket_connect("/v1/realtime?session_id=long-run") as ws:
            assert ws.receive_json()["type"] == ev.SESSION_CREATED
            for idx in range(20):
                ws.send_json({"type": ev.INPUT_APPEND, "modality": "video", "data": f"data:image/jpeg;base64,{idx}"})
                events = []
                for _ in range(4):
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] == ev.RESPONSE_DONE:
                        break
                assert [e["type"] for e in events] == [
                    ev.RESPONSE_CREATED,
                    ev.RESPONSE_DELTA,
                    ev.RESPONSE_DONE,
                ]
            assert backend.calls == 20


@pytest.mark.asyncio
async def test_session_manager_allows_concurrent_independent_sessions(monkeypatch):
    from vllm_omni.experimental.fullduplex.joyvl.serving import server as server_mod

    backend = _CountingBackend()
    monkeypatch.setattr(server_mod, "OpenAIBackend", lambda *args, **kwargs: backend)
    manager = server_mod.SessionManager(InteractionConfig(enable_memory=False, enable_delegation=False))
    try:
        await asyncio.gather(
            manager.step("session-a", ["data:image/jpeg;base64,A"], "query"),
            manager.step("session-b", ["data:image/jpeg;base64,B"], "query"),
        )
    finally:
        await manager.aclose()
    assert backend.max_active == 2
