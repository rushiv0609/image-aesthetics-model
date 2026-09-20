"""Small, safe dataset acquisition primitives (never handles credentials)."""
from __future__ import annotations
import hashlib, json, os, platform, shutil, urllib.request
from datetime import datetime, timezone
from pathlib import Path
from .registry import DatasetRecord

TOOL_VERSION = "acquire-dataset/1.0"

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()

def safe_member_path(root: Path, member: str) -> Path:
    """Resolve an archive member and reject absolute or escaping paths."""
    candidate = (root / member).resolve()
    root = root.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"archive path escapes destination: {member}")
    return candidate

def manifest_path(dataset_root: Path) -> Path: return dataset_root / "download-manifest.json"

def _file_record(dataset_root: Path, path: Path, *, url: str) -> dict[str, object]:
    resolved_root = dataset_root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"manifest file is outside dataset root: {path}")
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    return {
        "filename": str(resolved_path.relative_to(resolved_root)),
        "url": url,
        "size": resolved_path.stat().st_size,
        "sha256": sha256(resolved_path),
    }

def _write_immutable_manifest(dataset_root: Path, payload: dict[str, object]) -> Path:
    target = manifest_path(dataset_root)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") == encoded:
            return target
        raise FileExistsError(f"immutable manifest already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    return target

def write_manifest(dataset_root: Path, *, url: str, source: str, path: Path, user_supplied: bool = False) -> Path:
    payload = {"manifest_version": 1, "dataset_id": dataset_root.name, "filename": str(path.relative_to(dataset_root)), "url": url,
               "retrieved_at": datetime.now(timezone.utc).isoformat(), "size": path.stat().st_size,
               "sha256": sha256(path), "tool_version": TOOL_VERSION, "source": source,
               "user_supplied": user_supplied, "redistribution": "prohibited",
               "platform": platform.platform()}
    return _write_immutable_manifest(dataset_root, payload)

def write_multi_manifest(
    dataset_root: Path,
    *,
    files: list[tuple[Path, str]],
    source: str,
    user_supplied: bool = False,
) -> Path:
    """Record a complete multi-file download as one immutable snapshot."""
    if not files:
        raise ValueError("multi-file manifest requires at least one file")
    records = [_file_record(dataset_root, path, url=url) for path, url in files]
    filenames = [record["filename"] for record in records]
    if len(filenames) != len(set(filenames)):
        raise ValueError("duplicate filename in multi-file manifest")
    records.sort(key=lambda record: str(record["filename"]))
    payload = {
        "manifest_version": 2,
        "dataset_id": dataset_root.name,
        "files": records,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": TOOL_VERSION,
        "source": source,
        "user_supplied": user_supplied,
        "redistribution": "prohibited",
        "platform": platform.platform(),
    }
    return _write_immutable_manifest(dataset_root, payload)

def verify(dataset_root: Path) -> tuple[bool, str]:
    target = manifest_path(dataset_root)
    if not target.exists(): return False, "manifest missing"
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        entries = data.get("files")
        if entries is None:
            entries = [{key: data[key] for key in ("filename", "size", "sha256")}]
        if not isinstance(entries, list) or not entries:
            return False, "invalid manifest"
    except (OSError, ValueError, KeyError): return False, "invalid manifest"
    seen: set[str] = set()
    for entry in entries:
        try:
            filename = entry["filename"]
            if filename in seen:
                return False, "invalid manifest"
            seen.add(filename)
            path = safe_member_path(dataset_root, filename)
            expected_size = entry["size"]
            expected_sha = entry["sha256"]
        except (TypeError, ValueError, KeyError):
            return False, "invalid manifest"
        if not path.is_file(): return False, f"manifest file missing: {filename}"
        if path.stat().st_size != expected_size: return False, f"size mismatch: {filename}"
        if sha256(path) != expected_sha: return False, f"SHA-256 mismatch: {filename}"
    return True, f"verified {len(entries)} file(s)"

def download(url: str, destination: Path) -> Path:
    """Download with HTTP Range resume into a temporary sibling, then atomically rename."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    with urllib.request.urlopen(request) as response:
        status = getattr(response, "status", response.getcode())
        mode = "wb"
        if offset and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {offset}-"):
                raise ValueError(f"invalid resume Content-Range: {content_range!r}")
            mode = "ab"
        with partial.open(mode) as out:
            shutil.copyfileobj(response, out)
    os.replace(partial, destination)
    return destination
