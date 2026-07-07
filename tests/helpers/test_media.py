# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

from tests.helpers import media


def test_generate_synthetic_video_embed_audio_fallback_has_audio(monkeypatch, tmp_path):
    def raise_tts_unavailable(*args, **kwargs):
        raise RuntimeError("tts unavailable")

    monkeypatch.setattr(media, "generate_synthetic_audio", raise_tts_unavailable)

    result = media.generate_synthetic_video(
        32,
        32,
        4,
        embed_audio=True,
        force_regenerate=True,
        cache_dir=tmp_path,
    )

    assert media._mp4_bytes_have_decodable_audio(Path(result["file_path"]).read_bytes())


def test_generate_synthetic_video_regenerates_audio_missing_cache(monkeypatch, tmp_path):
    def raise_tts_unavailable(*args, **kwargs):
        raise RuntimeError("tts unavailable")

    video_only = media.generate_synthetic_video(
        32,
        32,
        4,
        embed_audio=False,
        force_regenerate=True,
        cache_dir=tmp_path,
    )
    bad_cache_path = tmp_path / "synth_video_w32_h32_nf4_ea1.mp4"
    bad_cache_path.write_bytes(Path(video_only["file_path"]).read_bytes())

    monkeypatch.setattr(media, "generate_synthetic_audio", raise_tts_unavailable)
    result = media.generate_synthetic_video(
        32,
        32,
        4,
        embed_audio=True,
        cache_dir=tmp_path,
    )

    assert Path(result["file_path"]) == bad_cache_path.resolve()
    assert media._mp4_bytes_have_decodable_audio(bad_cache_path.read_bytes())
