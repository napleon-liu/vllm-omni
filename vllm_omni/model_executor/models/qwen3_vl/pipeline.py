# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-VL single-stage multimodal understanding topology."""

from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

QWEN3_VL_PIPELINE = PipelineConfig(
    model_type="qwen3_vl",
    model_arch="Qwen3VLForConditionalGeneration",
    hf_architectures=("Qwen3VLForConditionalGeneration",),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="qwen3_vl",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            final_output=True,
            final_output_type="text",
            owns_tokenizer=True,
            requires_multimodal_data=True,
            engine_output_type="text",
            model_arch="Qwen3VLForConditionalGeneration",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
