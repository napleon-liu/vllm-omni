# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-Omni Thinker fused-MoE LoRA setup."""

import pytest
from transformers import PretrainedConfig

from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerForConditionalGeneration,
    _ensure_thinker_architecture,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_THINKER_ARCHITECTURE = "Qwen3OmniMoeThinkerForConditionalGeneration"


def test_thinker_uses_3d_fused_moe_lora():
    assert Qwen3OmniMoeThinkerForConditionalGeneration.is_3d_moe_weight is True


def test_missing_thinker_architecture_is_populated():
    config = PretrainedConfig()
    assert config.architectures is None

    _ensure_thinker_architecture(config)

    assert config.architectures == [_THINKER_ARCHITECTURE]


def test_existing_thinker_architecture_is_preserved():
    config = PretrainedConfig(architectures=["CustomThinkerArchitecture"])

    _ensure_thinker_architecture(config)

    assert config.architectures == ["CustomThinkerArchitecture"]
