"""Dataset-neutral fixed ten-signal cache contract."""
from __future__ import annotations
import hashlib, json, os, tempfile
from pathlib import Path

FEATURES = ("siglip_embedding", "clip_embedding", "laion_clip", "a1_siglip", "nima", "musiq_aesthetic", "musiq_technical", "sharpness", "contrast", "colorfulness")
DIMENSIONS = {"siglip_embedding": 768, "clip_embedding": 768}
SCHEMA_VERSION = "ten-signal-cache-v1"

def canonical_json(value): return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha256_bytes(value: bytes) -> str: return hashlib.sha256(value).hexdigest()

def validate_features(features: dict) -> None:
    keys = set(features)
    missing = set(FEATURES) - keys
    unknown = keys - set(FEATURES)
    if missing: raise ValueError(f"missing required signals: {sorted(missing)}")
    if unknown: raise ValueError(f"unknown signals: {sorted(unknown)}")
    required = {"dimension", "dtype", "range", "direction", "model_revision", "checkpoint_sha256", "preprocessing", "normalization"}
    for name in FEATURES:
        if not isinstance(features[name], dict) or set(features[name]) != required: raise ValueError(f"invalid signal contract fields: {name}")
        expected = DIMENSIONS.get(name, 1)
        if int(features[name].get("dimension", -1)) != expected: raise ValueError(f"{name} must have dimension {expected}")
        f=features[name]
        if f["dtype"] != "float32" or not isinstance(f["range"], list) or len(f["range"]) != 2 or not all(isinstance(x,(int,float)) for x in f["range"]): raise ValueError(f"invalid dtype/range: {name}")
        if f["range"][0] >= f["range"][1] or f["direction"] not in ("higher_is_better", "lower_is_better") or not isinstance(f["model_revision"], str) or not f["model_revision"] or not isinstance(f["preprocessing"], dict) or not f["preprocessing"] or not isinstance(f["normalization"], str) or not f["normalization"]: raise ValueError(f"invalid signal identity: {name}")
        if len(f["checkpoint_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in f["checkpoint_sha256"].lower()): raise ValueError(f"invalid checkpoint hash: {name}")
    if any("cosine" in key.lower() for key in keys): raise ValueError("query-dependent cosine signal is forbidden")

def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cache_identity(input_manifest_sha256: str, contract: dict,
                   source_fingerprints_sha256: str | None = None) -> str:
    if len(input_manifest_sha256) != 64 or any(c not in "0123456789abcdef" for c in input_manifest_sha256.lower()): raise ValueError("invalid manifest SHA-256")
    validate_features(contract["features"])
    if source_fingerprints_sha256 is not None and (len(source_fingerprints_sha256) != 64 or any(c not in "0123456789abcdef" for c in source_fingerprints_sha256.lower())):
        raise ValueError("invalid source fingerprints SHA-256")
    payload = {"schema_version": SCHEMA_VERSION, "input_manifest_sha256": input_manifest_sha256,
               "source_fingerprints_sha256": source_fingerprints_sha256, "contract": contract}
    return sha256_bytes(canonical_json(payload).encode())

def write_contract(path: Path, *, input_manifest_sha256: str, contract: dict,
                   source_fingerprints_sha256: str | None = None) -> str:
    identity = cache_identity(input_manifest_sha256, contract, source_fingerprints_sha256)
    payload = {"schema_version": SCHEMA_VERSION, "cache_identity": identity,
               "input_manifest_sha256": input_manifest_sha256,
               "source_fingerprints_sha256": source_fingerprints_sha256, "contract": contract}
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode()
    if path.exists():
        if path.read_bytes() != encoded: raise FileExistsError(f"immutable contract differs: {path}")
        return identity
    fd, tmp_name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    tmp = Path(tmp_name)
    with os.fdopen(fd, "wb") as handle:
        handle.write(encoded); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        fd=os.open(path.parent, os.O_RDONLY); os.fsync(fd); os.close(fd)
    except OSError: pass
    return identity

def validate_relative_manifest(rows: list[dict]) -> None:
    ids = set()
    for row in rows:
        if row["image_id"] in ids: raise ValueError("duplicate image_id")
        ids.add(row["image_id"])
        source = str(row.get("source_path", ""))
        parsed = Path(source)
        if not source or parsed.is_absolute() or ".." in parsed.parts or "" in parsed.parts:
            raise ValueError("manifest source_path must be a non-empty safe relative path")

def load_manifest(path: Path, root: Path) -> list[dict]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    validate_relative_manifest(rows)
    base = Path(root).resolve(strict=True)
    if not base.is_dir(): raise ValueError("image root must be a directory")
    for row in rows:
        candidate = base / row["source_path"]
        current = base
        for part in Path(row["source_path"]).parts:
            current = current / part
            if current.is_symlink(): raise ValueError(f"symlink manifest path: {row['source_path']}")
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(base)
        except (OSError, ValueError):
            raise ValueError(f"unsafe or missing manifest path: {row['source_path']}") from None
        if not resolved.is_file():
            raise ValueError(f"unsafe or missing manifest path: {row['source_path']}")
    return rows

def load_contract(path: Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != SCHEMA_VERSION: raise ValueError("unsupported contract schema")
    validate_features(payload["contract"]["features"])
    if payload.get("cache_identity") != cache_identity(payload["input_manifest_sha256"], payload["contract"], payload.get("source_fingerprints_sha256")): raise ValueError("cache identity mismatch")
    return payload
