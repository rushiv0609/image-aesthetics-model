from __future__ import annotations

from image_aesthetics.embeddings.mlx_siglip2 import _load_local_hf_model


def test_pinned_huggingface_model_load_is_strictly_local():
    calls = []

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            calls.append((args, kwargs))
            return "local-model"

    assert _load_local_hf_model(FakeAutoModel, "google/siglip2-base-patch16-224", "pinned-revision") == "local-model"
    assert calls == [(("google/siglip2-base-patch16-224",),
                      {"revision": "pinned-revision", "local_files_only": True})]
