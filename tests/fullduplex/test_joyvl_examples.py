from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "examples" / "online_serving" / "joyvl_interaction" / "scripts"


def _read_script(name: str) -> str:
    return (SCRIPTS_DIR / name).read_text(encoding="utf-8")


def test_joyvl_model_launcher_uses_omni_pipeline():
    script = _read_script("start_model.sh")

    assert "NOT --omni" not in script
    assert 'vllm serve "${MODEL}"' in script
    assert "--omni" in script
    assert "--limit-mm-per-prompt" in script
    assert "IMAGE_LIMIT" in script


def test_joyvl_one_shot_launcher_uses_reproducible_omni_args():
    script = _read_script("start_all.sh")

    assert 'vllm serve "$MODEL" --omni' in script
    assert '--max-model-len "$MAX_MODEL_LEN"' in script
    assert "--limit-mm-per-prompt" in script
    assert "IMAGE_LIMIT" in script
