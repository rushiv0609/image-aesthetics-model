"""Dataset registry loading and licensing-aware validation.

The registry is deliberately file based and side-effect free.  Acquisition is a
later phase; this module only reads small JSON records and produces a plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = (
    "dataset_id", "display_name", "task", "official_url", "download_method",
    "declared_license", "license_class", "commercial_status",
    "redistribution_status", "source_version", "expected_files",
    "known_upstream_overlap", "enabled",
)
VALID_TASKS = {"aesthetic", "technical", "evaluation-only"}
VALID_DOWNLOAD_METHODS = {"direct", "google-drive", "form", "manual"}
VALID_LICENSE_CLASSES = {"broad", "research-only", "academic-only", "unclear"}
VALID_STATUS = {"allowed", "forbidden", "unclear"}
VALID_USE_MODES = {"commercial-safe", "non-commercial-research"}


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    display_name: str
    task: str
    official_url: str
    download_method: str
    declared_license: str
    license_class: str
    commercial_status: str
    redistribution_status: str
    source_version: str
    expected_files: tuple[str, ...]
    known_upstream_overlap: tuple[str, ...]
    enabled: bool
    path: Path | None = None

    @classmethod
    def from_mapping(cls, value: dict[str, Any], path: Path | None = None) -> "DatasetRecord":
        if not isinstance(value, dict):
            raise ValueError("record must be a JSON object")
        missing = [field for field in REQUIRED_FIELDS if field not in value]
        if missing:
            raise ValueError(f"missing required fields: {', '.join(missing)}")
        for field in ("expected_files", "known_upstream_overlap"):
            raw = value[field]
            if not isinstance(raw, list) or not raw or any(not isinstance(item, str) or not item.strip() for item in raw):
                raise ValueError(f"{field} must be a non-empty list of non-empty strings")
        return cls(
            dataset_id=value["dataset_id"], display_name=value["display_name"],
            task=value["task"], official_url=value["official_url"],
            download_method=value["download_method"], declared_license=value["declared_license"],
            license_class=value["license_class"], commercial_status=value["commercial_status"],
            redistribution_status=value["redistribution_status"], source_version=value["source_version"],
            expected_files=tuple(value["expected_files"]),
            known_upstream_overlap=tuple(value["known_upstream_overlap"]),
            enabled=value["enabled"], path=path,
        )

    def errors(self) -> list[str]:
        errors: list[str] = []
        if not self.dataset_id or self.dataset_id != self.dataset_id.strip(): errors.append("dataset_id must be non-empty")
        if not self.display_name: errors.append("display_name must be non-empty")
        if self.task not in VALID_TASKS: errors.append(f"task must be one of {sorted(VALID_TASKS)}")
        if not self.official_url.startswith(("http://", "https://")): errors.append("official_url must be an HTTP(S) URL")
        if self.download_method not in VALID_DOWNLOAD_METHODS: errors.append(f"download_method must be one of {sorted(VALID_DOWNLOAD_METHODS)}")
        if not self.declared_license.strip(): errors.append("declared_license must be non-empty")
        if self.license_class not in VALID_LICENSE_CLASSES: errors.append(f"license_class must be one of {sorted(VALID_LICENSE_CLASSES)}")
        if self.commercial_status not in VALID_STATUS: errors.append(f"commercial_status must be one of {sorted(VALID_STATUS)}")
        if self.redistribution_status not in VALID_STATUS: errors.append(f"redistribution_status must be one of {sorted(VALID_STATUS)}")
        if not self.source_version.strip(): errors.append("source_version must be non-empty")
        if not self.expected_files or any(not isinstance(item, str) or not item for item in self.expected_files): errors.append("expected_files must be non-empty strings")
        if not isinstance(self.known_upstream_overlap, tuple): errors.append("known_upstream_overlap must be a list")
        if not isinstance(self.enabled, bool): errors.append("enabled must be boolean")
        return errors

    def acquisition_decision(self, use_mode: str) -> tuple[bool, str]:
        if use_mode not in VALID_USE_MODES:
            raise ValueError(f"use_mode must be one of {sorted(VALID_USE_MODES)}")
        if not self.enabled:
            return False, "disabled"
        if use_mode == "commercial-safe":
            if self.license_class != "broad" or self.commercial_status != "allowed":
                return False, "not approved for commercial use"
            return True, "approved for commercial-safe local acquisition"
        if self.license_class in {"broad", "research-only", "academic-only"}:
            return True, "approved for local non-commercial research"
        return False, "license terms unclear; explicit source review required"


def load_registry(directory: Path) -> list[DatasetRecord]:
    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            records.append(DatasetRecord.from_mapping(payload, path))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    if not records:
        raise ValueError(f"{directory}: registry is empty")
    return records


def validate(records: list[DatasetRecord]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for record in records:
        if record.dataset_id in seen: errors.append(f"duplicate dataset_id: {record.dataset_id}")
        seen.add(record.dataset_id)
        errors.extend(f"{record.dataset_id}: {error}" for error in record.errors())
    return errors


def acquisition_plan(
    records: list[DatasetRecord], use_mode: str = "commercial-safe"
) -> list[DatasetRecord]:
    """Return enabled records approved for local acquisition in the selected mode."""
    return sorted(
        (record for record in records if record.acquisition_decision(use_mode)[0]),
        key=lambda record: record.dataset_id,
    )
