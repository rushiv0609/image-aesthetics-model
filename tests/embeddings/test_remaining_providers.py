from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from image_aesthetics.embeddings.feature_contract import validate_features
from image_aesthetics.embeddings.preprocess import load_manifest
from image_aesthetics.embeddings.remaining_providers import (
    REMAINING_FOUR, prepare_remaining_images, remaining_contract,
)
from image_aesthetics.embeddings.ten_signal_runner import RunnerConfig, TenSignalRunner


def test_remaining_contract_uses_fixed_schema_and_marks_nima_inactive():
    contract = remaining_contract()
    validate_features(contract["features"])
    assert set(contract["features"]) == {
        "siglip_embedding", "clip_embedding", "laion_clip", "a1_siglip", "nima",
        "musiq_aesthetic", "musiq_technical", "sharpness", "contrast", "colorfulness",
    }
    assert contract["features"]["nima"]["model_revision"] == "inactive-remaining-four:nima"
    assert contract["features"]["musiq_aesthetic"]["preprocessing"]["resize"].endswith("512")
    assert contract["features"]["musiq_technical"]["checkpoint_sha256"] != contract["features"]["musiq_aesthetic"]["checkpoint_sha256"]


def test_clip_and_musiq_preparations_are_separate_and_preserve_native_aspect(tmp_path: Path):
    from PIL import Image

    path = tmp_path / "wide.png"
    Image.new("RGB", (800, 400), (80, 90, 100)).save(path)
    square, native, decoder = prepare_remaining_images(path)
    assert square.shape == (224, 224, 3)
    assert native.shape == (256, 512, 3)
    assert decoder == "pillow-exif-rgb-v1"
    assert square.dtype == native.dtype == np.uint8


def test_runner_passes_separate_native_images_to_declared_provider(tmp_path: Path):
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(2):
        source = image_root / f"{index}.jpg"
        source.write_bytes(f"fixture-{index}".encode())
        stat = source.stat()
        rows.append({"image_id": f"id-{index}", "source_path": source.name,
                     "source_size": stat.st_size, "source_mtime_ns": stat.st_mtime_ns,
                     "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class NativeProvider:
        names = REMAINING_FOUR
        requires_native_images = True

        def extract(self, images, *, native_images):
            assert images.shape == (len(native_images), 224, 224, 3)
            assert all(image.ndim == 3 and image.shape[2] == 3 for image in native_images)
            result = {}
            for name in self.names:
                dim = 768 if name == "clip_embedding" else 1
                values = np.zeros((len(images), dim), dtype=np.float32)
                if name == "clip_embedding":
                    values[:, 0] = 1.0
                else:
                    values.fill(0.5)
                result[name] = values
            return result

    contract = remaining_contract()
    items = load_manifest(manifest, image_root=image_root)
    result = TenSignalRunner(
        RunnerConfig(tmp_path / "cache", batch_size=2, shard_size=8, max_new_images=1),
        batch_extractor=NativeProvider(),
        native_prepare_fn=lambda path, size: (
            np.zeros((size, size, 3), dtype=np.uint8),
            np.zeros((8 if path.stem == "0" else 7, 6 if path.stem == "0" else 8, 3), dtype=np.uint8),
            "fixture",
        ),
    ).run(items, contract, run_id="native-input-test")
    assert result["state"] == "incomplete"
    assert result["completed_images"] == 1


def test_bounded_run_resumes_in_chunks_without_reprocessing(tmp_path: Path):
    from PIL import Image
    import hashlib

    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(3):
        source = image_root / f"{index}.png"
        Image.new("RGB", (12, 8), (20 + index, 30, 40)).save(source)
        rows.append({"image_id": f"id-{index}", "source_path": source.name,
                     "source_size": source.stat().st_size,
                     "source_mtime_ns": source.stat().st_mtime_ns,
                     "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class TinyProvider:
        names = REMAINING_FOUR
        requires_native_images = True

        def extract(self, images, *, native_images):
            outputs = {}
            for name in self.names:
                width = 768 if name == "clip_embedding" else 1
                values = np.zeros((len(images), width), dtype=np.float32)
                if name == "clip_embedding":
                    values[:, 0] = 1.0
                else:
                    values.fill(0.5)
                outputs[name] = values
            return outputs

    runner = TenSignalRunner(
        RunnerConfig(tmp_path / "cache", batch_size=2, shard_size=1, max_new_images=1),
        batch_extractor=TinyProvider(), native_prepare_fn=prepare_remaining_images,
    )
    items = load_manifest(manifest, image_root=image_root)
    contract = remaining_contract()
    first = runner.run(items, contract, run_id="chunk-1")
    second = runner.run(items, contract, run_id="chunk-2")
    third = runner.run(items, contract, run_id="chunk-3")
    assert first["state"] == "incomplete" and first["completed_images"] == 1
    assert second["state"] == "incomplete" and second["completed_images"] == 2
    assert third["state"] == "completed" and third["completed_images"] == 3
    clean = runner.run(items, contract, run_id="clean-resume")
    assert clean["state"] == "completed"
    assert clean["decoded_images"] == 0 and clean["inference_batches"] == 0
