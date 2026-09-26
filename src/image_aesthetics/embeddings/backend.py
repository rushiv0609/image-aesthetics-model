"""Embedding backend contract and production SigLIP2 adapter boundary."""

from __future__ import annotations

from typing import Any, Callable


class SigLIP2Embedder:
    """Adapter for the production MLX SigLIP2 Base-224 implementation.

    The cache pipeline owns one instance on its inference thread.  A callable
    ``backend`` is accepted for smoke tests and for wiring the project's MLX
    implementation without importing it at module import time.
    """

    def __init__(self, model_id: str, revision: str, *, backend: Callable[[Any], Any] | None = None):
        self.model_id = model_id
        self.revision = revision
        self._backend = backend

    @classmethod
    def production(cls, model_id: str, revision: str) -> "SigLIP2Embedder":
        """Construct the standalone MLX FP16 backend lazily on Apple Silicon."""
        from .mlx_siglip2 import load_backend
        return cls(model_id, revision, backend=load_backend(model_id, revision))

    def embed(self, images: Any) -> Any:
        if self._backend is None:
            raise RuntimeError(
                "No MLX backend was supplied. Wire the pinned production SigLIP2 MLX extractor "
                "through --backend-factory or the application embedding adapter."
            )
        output = self._backend(images)
        try:
            import numpy as np  # type: ignore
            output = np.asarray(output)
            if output.dtype != np.dtype("float32"):
                raise ValueError(f"SigLIP2 output must be float32, got {output.dtype}")
            if output.ndim != 2 or output.shape[1] != 768:
                raise ValueError(f"SigLIP2 output must have shape (N,768), got {output.shape}")
            if not np.isfinite(output).all():
                raise ValueError("SigLIP2 output contains NaN/Inf")
            norms = np.linalg.norm(output, axis=1)
            if np.any(norms < 1e-6) or not np.allclose(norms, 1.0, atol=5e-3):
                raise ValueError("SigLIP2 output must be nonzero and unit-normalized")
            return output
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("numpy is required for embedding validation") from exc


class CallableEmbedder:
    """Small test adapter that normalizes any deterministic callable output."""

    def __init__(self, function: Callable[[Any], Any]):
        self.function = function

    def embed(self, images: Any) -> Any:
        return SigLIP2Embedder("test", "test", backend=self.function).embed(images)
