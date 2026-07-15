# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MiniCPM-o 4.5 audio placeholder sizing."""

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.transformers_utils.processors.minicpmo import MiniCPMOProcessor

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import (
    MiniCPMO45OmniLLMProcessingInfo,
    MiniCPMOConfig,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeProcessor:
    get_audio_placeholder = MiniCPMOProcessor.get_audio_placeholder

    def __init__(self, pool_step: int):
        self.pool_step = pool_step
        self.feature_extractor = SimpleNamespace(hop_length=160)
        self.tokenizer = SimpleNamespace(audio_start="<audio>", audio_end="</audio>")
        self.image_processor = SimpleNamespace(
            mean=np.array([0.5], dtype=np.float32),
            std=np.array([0.5], dtype=np.float32),
        )


class _FakeContext:
    def __init__(self, audio_pool_step: int | None = 5):
        self.tokenizer = object()
        self.model_config = SimpleNamespace()
        self._hf_config = SimpleNamespace()
        if audio_pool_step is not None:
            self._hf_config.audio_pool_step = audio_pool_step
        self.processor_kwargs: dict[str, object] | None = None

    def get_hf_config(self):
        return self._hf_config

    def get_hf_processor(self, **kwargs: object):
        self.processor_kwargs = kwargs
        return _FakeProcessor(pool_step=int(kwargs.get("pool_step", 2)))


def test_config_defaults_to_minicpmo_4_5_audio_pool_step() -> None:
    assert MiniCPMOConfig().audio_pool_step == 5


def test_processing_info_defaults_to_minicpmo_4_5_audio_pool_step() -> None:
    info = MiniCPMO45OmniLLMProcessingInfo(_FakeContext(audio_pool_step=None))

    assert info.get_default_audio_pool_step() == 5


def test_processor_pool_step_is_forced_to_match_model_config() -> None:
    ctx = _FakeContext(audio_pool_step=5)
    info = MiniCPMO45OmniLLMProcessingInfo(ctx)

    processor = info.get_hf_processor(pool_step=2)

    assert processor.pool_step == 5
    assert ctx.processor_kwargs == {"pool_step": 5}


@pytest.mark.parametrize(
    ("duration_seconds", "expected_audio_tokens"),
    [(5, 50), (30, 300)],
)
def test_audio_placeholder_count_matches_encoder_pooling(
    duration_seconds: int,
    expected_audio_tokens: int,
) -> None:
    info = MiniCPMO45OmniLLMProcessingInfo(_FakeContext(audio_pool_step=5))
    processor = info.get_hf_processor()

    placeholder = processor.get_audio_placeholder(
        duration_seconds * 16_000,
        chunk_input=False,
        chunk_length=1,
    )

    assert placeholder.count("<unk>") == expected_audio_tokens
