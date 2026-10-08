import pytest

from app.core.config import RerankerSettings
from app.core.errors import ProviderConfigurationError
from app.providers.registry import build_reranker
from app.providers.rerankers.fastembed_reranker import (
    KNOWN_ONNX_RERANKERS,
    FastEmbedCrossEncoderReranker,
    register_custom_model,
)

fastembed = pytest.importorskip("fastembed")


def test_bge_v2_m3_is_premapped_to_int8_onnx_export():
    r = build_reranker(RerankerSettings(provider="fastembed", model="BAAI/bge-reranker-v2-m3"))
    assert isinstance(r, FastEmbedCrossEncoderReranker)
    assert r._custom == KNOWN_ONNX_RERANKERS["BAAI/bge-reranker-v2-m3"]
    assert r._custom[1] == "onnx/model_int8.onnx"


def test_settings_override_onnx_source():
    r = build_reranker(
        RerankerSettings(
            provider="fastembed",
            model="BAAI/bge-reranker-v2-m3",
            onnx_repo="onnx-community/bge-reranker-v2-m3-ONNX",
            onnx_file="onnx/model.onnx",
            onnx_additional_files=["onnx/model.onnx_data"],
        )
    )
    assert r._custom == (
        "onnx-community/bge-reranker-v2-m3-ONNX",
        "onnx/model.onnx",
        ("onnx/model.onnx_data",),
    )


def test_unknown_model_without_onnx_source_fails_clearly():
    r = FastEmbedCrossEncoderReranker("someone/unknown-reranker")
    with pytest.raises(ProviderConfigurationError, match="RERANKER__ONNX_REPO"):
        r._load()


def test_register_custom_model_is_idempotent():
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    register_custom_model("test/custom-reranker", "test/custom-reranker-onnx", "onnx/model.onnx")
    register_custom_model("test/custom-reranker", "test/custom-reranker-onnx", "onnx/model.onnx")
    names = [m["model"] for m in TextCrossEncoder.list_supported_models()]
    assert names.count("test/custom-reranker") == 1
