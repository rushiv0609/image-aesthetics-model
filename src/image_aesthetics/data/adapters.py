"""Metadata adapters for the independent aesthetic/technical datasets.

Adapters intentionally read annotation archives in place.  They do not extract
archives or touch image data; preparation can later resolve ``image_name``
against an externally managed image directory.
"""

from __future__ import annotations

import csv
import io
import math
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator
from xml.etree import ElementTree


@dataclass(frozen=True)
class CanonicalSample:
    """One source annotation in the dataset-neutral manifest contract."""

    dataset_id: str
    image_id: str
    image_name: str
    native_score: float
    native_score_range: tuple[float, float]
    split: str | None = None
    attributes: dict[str, float] = field(default_factory=dict)
    themes: tuple[str, ...] = ()
    exif: dict[str, Any] = field(default_factory=dict)
    source_groups: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetAudit:
    dataset_id: str
    annotation_files: tuple[str, ...]
    accepted: int
    rejected: int
    rejection_reasons: dict[str, int]


class DatasetAdapter:
    dataset_id: str

    def inventory(self) -> DatasetAudit:
        accepted = rejected = 0
        reasons: dict[str, int] = {}
        for _ in self.iter_samples(include_rejections=True):
            accepted += 1
        return DatasetAudit(self.dataset_id, (), accepted, rejected, reasons)

    def grouping_keys(self, sample: CanonicalSample) -> tuple[str, ...]:
        return sample.source_groups or (sample.image_id,)


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


class TAD66KAdapter(DatasetAdapter):
    """Read TAD66K's merged train/test CSVs from ``labels.zip``."""

    dataset_id = "tad66k"
    _files = ("labels/merge/train.csv", "labels/merge/test.csv")

    def __init__(self, labels_archive: Path):
        self.labels_archive = Path(labels_archive)

    def _themes(self, archive: zipfile.ZipFile) -> dict[tuple[str, str], set[str]]:
        memberships: dict[tuple[str, str], set[str]] = {}
        for filename in sorted(archive.namelist()):
            parts = Path(filename).parts
            if len(parts) != 4 or parts[:2] != ("labels", "unmerge") or not filename.endswith(".csv"):
                continue
            split, theme = parts[2], Path(parts[3]).stem
            reader = csv.DictReader(io.StringIO(archive.read(filename).decode("utf-8-sig")))
            for row in reader:
                image = (row.get("image") or row.get("Image") or "").strip()
                if image:
                    memberships.setdefault((split, image), set()).add(theme)
        return memberships

    def _rows(self) -> Iterator[tuple[str, str, dict[str, Any] | None]]:
        with zipfile.ZipFile(self.labels_archive) as archive:
            names = set(archive.namelist())
            for filename in self._files:
                if filename not in names:
                    yield filename, "missing annotation file", None
                    continue
                split = Path(filename).stem
                text = archive.read(filename).decode("utf-8-sig")
                reader = csv.DictReader(io.StringIO(text))
                if reader.fieldnames is None or {x.strip().lower() for x in reader.fieldnames} != {"image", "score"}:
                    yield filename, "invalid columns", None
                    continue
                for row_number, row in enumerate(reader, 2):
                    image = (row.get("image") or row.get("Image") or "").strip()
                    score = _finite(row.get("score") or row.get("Score"))
                    if not image:
                        yield f"{filename}:{row_number}", "missing image name", None
                    elif score is None or not 1.0 <= score <= 10.0:
                        yield f"{filename}:{row_number}", "score outside 1-10", None
                    else:
                        yield image, "", {"image": image, "score": score, "split": split}

    def iter_samples(self, include_rejections: bool = False) -> Iterator[CanonicalSample | tuple[str, str]]:
        seen: set[str] = set()
        with zipfile.ZipFile(self.labels_archive) as archive:
            themes = self._themes(archive)
        for key, reason, row in self._rows():
            if row is None:
                if include_rejections:
                    yield key, reason
                continue
            image = row["image"]
            if image in seen:
                if include_rejections:
                    yield image, "duplicate image name"
                continue
            seen.add(image)
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                image_id=image,
                image_name=image,
                native_score=row["score"],
                native_score_range=(1.0, 10.0),
                split=row["split"],
                themes=tuple(sorted(themes.get((row["split"], image), ()))),
                # TAD does not publish a photographer/source identity. Do not use
                # the theme or native split as a deduplication group: either would
                # incorrectly connect thousands of unrelated images.
                source_groups=(f"tad66k:image:{image}",),
                metadata={"native_label": row["score"]},
            )

    def inventory(self) -> DatasetAudit:
        accepted = rejected = 0
        reasons: dict[str, int] = {}
        for item in self.iter_samples(include_rejections=True):
            if isinstance(item, CanonicalSample):
                accepted += 1
            else:
                rejected += 1
                reasons[item[1]] = reasons.get(item[1], 0) + 1
        return DatasetAudit(self.dataset_id, self._files, accepted, rejected, reasons)


_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def _xlsx_rows(archive: zipfile.ZipFile, filename: str) -> Iterator[list[str]]:
    """Yield worksheet rows using only the Python standard library."""
    shared: list[str] = []
    if "xl/sharedStrings.xml" in archive.namelist():
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in root.findall(f"{{{_XLSX_NS}}}si"):
            shared.append("".join(node.text or "" for node in item.iter(f"{{{_XLSX_NS}}}t")))
    root = ElementTree.fromstring(archive.read(filename))
    for row in root.findall(f".//{{{_XLSX_NS}}}row"):
        values: list[str] = []
        next_column = 0
        for cell in row.findall(f"{{{_XLSX_NS}}}c"):
            reference = cell.get("r", "")
            letters = "".join(character for character in reference if character.isalpha())
            column = 0
            for character in letters:
                column = column * 26 + ord(character.upper()) - ord("A") + 1
            column -= 1
            values.extend([""] * max(0, column - next_column))
            raw = cell.find(f"{{{_XLSX_NS}}}v")
            value = "" if raw is None else (raw.text or "")
            if cell.get("t") == "s" and value:
                value = shared[int(value)]
            elif cell.get("t") == "inlineStr":
                value = "".join(node.text or "" for node in cell.iter(f"{{{_XLSX_NS}}}t"))
            values.append(value)
            next_column = column + 1
        yield values


def _xlsx_table(archive: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    rows = iter(_xlsx_rows(archive, filename))
    header = next(rows, [])
    normalized = [str(value).strip() for value in header]
    return [dict(zip(normalized, row + [""] * (len(normalized) - len(row)))) for row in rows]


class SPAQAdapter(DatasetAdapter):
    """Read SPAQ MOS, image attributes, scenes, and EXIF from its XLSX archive."""

    dataset_id = "spaq"
    _files = (
        "Annotations/MOS and Image attribute scores.xlsx",
        "Annotations/Scene category labels.xlsx",
        "Annotations/EXIF_tags.xlsx",
    )

    def __init__(self, annotations_archive: Path):
        self.annotations_archive = Path(annotations_archive)

    def _tables(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]], dict[str, dict[str, str]], list[str]]:
        with zipfile.ZipFile(self.annotations_archive) as outer:
            names = set(outer.namelist())
            missing = [filename for filename in self._files if filename not in names]
            by_name = []
            for filename in self._files:
                if filename in names:
                    with zipfile.ZipFile(io.BytesIO(outer.read(filename))) as workbook:
                        table = _xlsx_table(workbook, "xl/worksheets/sheet1.xml")
                else:
                    table = []
                by_name.append({(row.get("Image name") or "").strip(): row for row in table if (row.get("Image name") or "").strip()})
        return by_name[0], by_name[1], by_name[2], missing

    def iter_samples(self, include_rejections: bool = False) -> Iterator[CanonicalSample | tuple[str, str]]:
        mos, scenes, exif, missing = self._tables()
        if missing and include_rejections:
            for filename in missing:
                yield filename, "missing annotation file"
        for image in sorted(mos):
            row = mos[image]
            score = _finite(row.get("MOS"))
            if score is None or not 0.0 <= score <= 100.0:
                if include_rejections:
                    yield image, "MOS outside 0-100"
                continue
            attributes: dict[str, float] = {}
            for key in ("Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"):
                value = _finite(row.get(key))
                if value is not None:
                    attributes[key.lower()] = value
            scene = scenes.get(image, {})
            themes = tuple(sorted(key for key, value in scene.items() if key != "Image name" and _finite(value) is not None and float(value) > 0.0))
            exif_values = {key: value for key, value in exif.get(image, {}).items() if key != "Image name" and value != ""}
            yield CanonicalSample(
                dataset_id=self.dataset_id,
                image_id=image,
                image_name=image,
                native_score=score,
                native_score_range=(0.0, 100.0),
                attributes=attributes,
                themes=themes,
                exif=exif_values,
                source_groups=(f"spaq:image:{image}",),
                metadata={"native_label": score, "scene_membership": {key: scene[key] for key in themes}},
            )

    def inventory(self) -> DatasetAudit:
        accepted = rejected = 0
        reasons: dict[str, int] = {}
        for item in self.iter_samples(include_rejections=True):
            if isinstance(item, CanonicalSample):
                accepted += 1
            else:
                rejected += 1
                reasons[item[1]] = reasons.get(item[1], 0) + 1
        return DatasetAudit(self.dataset_id, self._files, accepted, rejected, reasons)
