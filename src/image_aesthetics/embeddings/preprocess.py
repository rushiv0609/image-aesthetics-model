"""AVA image manifest loading and production-compatible pixel preparation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Any


@dataclass(frozen=True)
class ManifestItem:
    image_id: str
    source_path: str
    manifest_position: int
    source_size: int
    source_mtime_ns: int
    source_fingerprint: str | None = None

    def as_ledger_row(self) -> dict[str, Any]:
        return asdict(self)


def source_fingerprint(path: Path, *, content_hash: bool = False) -> str | None:
    if not content_hash:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _item_from_record(record: dict[str, Any], position: int, base: Path | None) -> ManifestItem:
    raw_path = Path(str(record.get("source_path", record.get("path", ""))))
    path = raw_path if raw_path.is_absolute() or base is None else (base / raw_path)
    # A prepared manifest remains usable when a source is later missing.  The
    # decoder will record a permanent failure; re-reading the manifest must not
    # erase the original source identity by calling stat() again.
    if "source_size" in record and "source_mtime_ns" in record:
        size, mtime_ns = int(record["source_size"]), int(record["source_mtime_ns"])
    elif path.exists():
        stat = path.stat(); size, mtime_ns = stat.st_size, stat.st_mtime_ns
    else:
        size, mtime_ns = 0, 0
    image_id = str(record.get("image_id", record.get("id", path.stem)))
    return ManifestItem(image_id, str(path), position, size, mtime_ns,
                        str(record["source_fingerprint"]) if record.get("source_fingerprint") else None)


def load_manifest(path: Path) -> list[ManifestItem]:
    """Load JSON/JSONL/CSV manifests in deterministic position order.

    Parquet manifests are intentionally not read implicitly: requiring an
    explicit conversion/pyarrow dependency avoids silently changing row order.
    """
    path = Path(path)
    base = path.parent
    records: list[dict[str, Any]]
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        records = value["items"] if isinstance(value, dict) and "items" in value else value
    elif path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
    else:
        raise ValueError(f"unsupported manifest format: {path.suffix}; use JSONL, JSON, or CSV")
    items = [_item_from_record(record, idx, base) for idx, record in enumerate(records)]
    if len({item.image_id for item in items}) != len(items):
        raise ValueError("manifest contains duplicate image_id values")
    return items


def preprocessing_contract(image_size: int = 224) -> dict[str, Any]:
    return {
        "image_size": image_size,
        "resize": "aspect_preserving_fit",
        "padding": "black",
        "orientation": "exif_transpose",
        "color": "RGB",
        "interpolation": "lanczos",
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "decoder_preference": ["macos_imageio", "pillow"],
    }


def _prepare_with_macos_imageio(path: Path, image_size: int) -> Any:
    """Decode through ImageIO/CoreGraphics when running on macOS.

    Imports are local so Linux smoke tests and metadata-only commands do not
    require PyObjC.  The Pillow path below is a recorded fallback.
    """
    from Quartz import (  # type: ignore
        CGDataProviderCopyData, CGImageGetBitsPerComponent, CGImageGetBitsPerPixel,
        CGImageGetBytesPerRow, CGImageGetHeight, CGImageGetWidth,
        CGImageSourceCreateThumbnailAtIndex, CGImageSourceCreateWithURL,
        kCGImageSourceCreateThumbnailFromImageAlways, kCGImageSourceCreateThumbnailWithTransform,
        kCGImageSourceThumbnailMaxPixelSize,
    )
    import Foundation  # type: ignore
    import numpy as np  # type: ignore
    options = {
        kCGImageSourceCreateThumbnailFromImageAlways: True,
        kCGImageSourceCreateThumbnailWithTransform: True,
        kCGImageSourceThumbnailMaxPixelSize: image_size,
    }
    source = CGImageSourceCreateWithURL(Foundation.NSURL.fileURLWithPath_(str(path)), None)
    if source is None:
        raise ValueError(f"ImageIO could not open {path}")
    image = CGImageSourceCreateThumbnailAtIndex(source, 0, options)
    if image is None:
        raise ValueError(f"ImageIO could not decode {path}")
    width, height = CGImageGetWidth(image), CGImageGetHeight(image)
    bits = CGImageGetBitsPerPixel(image)
    if bits not in (32, 24) or CGImageGetBitsPerComponent(image) != 8:
        raise ValueError(f"unsupported ImageIO pixel format ({bits} bpp)")
    channels = bits // 8
    stride = CGImageGetBytesPerRow(image)
    raw = bytes(CGDataProviderCopyData(image))
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)[:, :width * channels]
    pixels = pixels.reshape(height, width, channels)
    if channels == 4:
        pixels = pixels[:, :, :3]
    if pixels.shape[:2] != (image_size, image_size):
        canvas = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        scale = min(image_size / width, image_size / height)
        new_w, new_h = max(1, round(width * scale)), max(1, round(height * scale))
        from PIL import Image  # type: ignore
        resized = np.asarray(Image.fromarray(pixels).resize((new_w, new_h), Image.Resampling.LANCZOS), dtype=np.uint8)
        y, x = (image_size - new_h) // 2, (image_size - new_w) // 2
        canvas[y:y + new_h, x:x + new_w] = resized
        pixels = canvas
    return np.ascontiguousarray(pixels, dtype=np.uint8)


def prepare_image(path: Path, image_size: int = 224) -> tuple[Any, str]:
    """Decode, orient, RGB-convert, fit, and pad an image as uint8 HWC."""
    try:
        return _prepare_with_macos_imageio(path, image_size), "macos_imageio"
    except (ImportError, ModuleNotFoundError, AttributeError, TypeError, ValueError, OSError):
        pass
    try:
        # A native ImageIO implementation can be added without changing the
        # contract. Pillow remains the portable fallback and is recorded.
        from PIL import Image, ImageOps  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow is required for image decoding") from exc
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        offset = ((image_size - image.width) // 2, (image_size - image.height) // 2)
        canvas.paste(image, offset)
        # ascontiguousarray is deliberately imported lazily with numpy.
        import numpy as np  # type: ignore
        return np.ascontiguousarray(canvas, dtype=np.uint8), "pillow"
