# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from typing import Any

INPUT_APPEND = "input.append"
INPUT_COMMIT = "input.commit"
RESPONSE_CREATE = "response.create"
RESPONSE_CANCEL = "response.cancel"
PLAYBACK_ACK = "playback.ack"
CLOSE = "close"


SESSION_CREATED = "session.created"
RESPONSE_CREATED = "response.created"
RESPONSE_DELTA = "response.delta"
RESPONSE_DONE = "response.done"
RESPONSE_CANCELLED = "response.cancelled"
ERROR = "error"


def session_created(
    session_id: str,
    input_modalities: tuple[str, ...] | list[str],
    output_modalities: tuple[str, ...] | list[str],
) -> dict[str, Any]:
    return {
        "type": SESSION_CREATED,
        "session_id": session_id,
        "input_modalities": list(input_modalities),
        "output_modalities": list(output_modalities),
    }


def created(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_CREATED, "response_index": response_index}


def delta(response_index: int, modality: str, data: Any) -> dict[str, Any]:
    return {"type": RESPONSE_DELTA, "response_index": response_index, "modality": modality, "data": data}


def done(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_DONE, "response_index": response_index}


def cancelled(response_index: int) -> dict[str, Any]:
    return {"type": RESPONSE_CANCELLED, "response_index": response_index}


def error(message: str) -> dict[str, Any]:
    return {"type": ERROR, "message": message}
