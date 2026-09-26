from __future__ import annotations
import hashlib, json, os, subprocess, sys, sqlite3
from pathlib import Path
import numpy as np
import pytest
from image_aesthetics.embeddings.ledger import Ledger
from image_aesthetics.embeddings.shards import write_atomic
from image_aesthetics.embeddings.feature_contract import FEATURES, DIMENSIONS, cache_identity

ROOT = Path(__file__).parents[2]

def spec():
    return {"features": {n: {"dimension": DIMENSIONS.get(n, 1), "dtype": "float32", "range": [0, 1], "direction": "higher_is_better", "model_revision": "r1", "checkpoint_sha256": "0" * 64, "preprocessing": {"image_size": 224}, "normalization": "unit"} for n in FEATURES}}

def run(*args):
    env = os.environ.copy(); env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, capture_output=True, text=True)

def cli(command, cache):
    extra = ["--allow-partial-diagnostics"] if command == "verify" else []
    result = run("scripts/embeddings/cache_ten_signal.py", command, "--cache-dir", str(cache), *extra)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)

def seed_cache(tmp_path, *, shard=True):
    cache = tmp_path / "cache"; cache.mkdir()
    ledger = Ledger(cache / "work.sqlite3"); ledger.open()
    source = tmp_path / "a.jpg"; source.write_bytes(b"image")
    ledger.seed_items([{"image_id": "a", "manifest_position": 0, "source_path": str(source), "source_size": source.stat().st_size, "source_mtime_ns": source.stat().st_mtime_ns, "source_fingerprint": "source-a"}])
    contract_value=spec()
    manifest_hash="1"*64
    identity=cache_identity(manifest_hash,contract_value)
    ordered_hash=hashlib.sha256(json.dumps(["a"],separators=(",",":")).encode()).hexdigest()
    fingerprints_hash=hashlib.sha256(json.dumps([["a",0,"source-a"]],separators=(",",":")).encode()).hexdigest()
    ledger.ensure_cache_identity(identity,manifest_hash,hashlib.sha256(json.dumps(contract_value,sort_keys=True,separators=(",",":")).encode()).hexdigest(),1,ordered_hash,fingerprints_hash)
    shard_path = cache / "shard-00000.safetensors"
    if shard:
        values = np.zeros((1, 1544), dtype=np.float32); values[0,0]=1.; values[0,768]=1.
        checksum, _ = write_atomic(shard_path, values, [0], ["a"], metadata_extra={
            "unit_norm":False,"signal_names":list(FEATURES),
            "signal_dimensions":{n:DIMENSIONS.get(n,1) for n in FEATURES},
            "signal_specs":contract_value["features"],"cache_identity":identity,
            "source_fingerprints_sha256":fingerprints_hash,"manifest_sha256":manifest_hash,
            "ordered_ids_sha256":ordered_hash,"item_count":"1"})
        ledger.start_run("run", model_revision="r1", preprocessing_hash="p", dataset_fingerprint="d", cache_schema_version=1, config={})
        ledger.start_writer()
        ledger.emit("commit_shard", wait=True, shard_id=0, path=str(shard_path), row_count=1, sha256=checksum, min_position=0, max_position=0, items=[{"image_id": "a", "position": 0, "checksum": hashlib.sha256(values[0].tobytes()).hexdigest()}])
        ledger.stop()
    return cache, ledger, shard_path

def test_contract_cli_and_unsafe_manifest(tmp_path):
    root = tmp_path / "images"; root.mkdir(); (root / "a.jpg").write_bytes(b"x")
    manifest = tmp_path / "m.jsonl"; manifest.write_text(json.dumps({"image_id": "a", "source_path": "a.jpg"}) + "\n")
    specp = tmp_path / "s.json"; specp.write_text(json.dumps(spec())); out = tmp_path / "c.json"
    result = run("scripts/embeddings/cache_ten_signal.py", "contract", "--manifest", str(manifest), "--image-root", str(root), "--spec", str(specp), "--contract", str(out))
    assert result.returncode == 0 and json.loads(result.stdout)["cache_identity"]

@pytest.mark.parametrize("source_path", ["../a.jpg", "/tmp/a.jpg"])
def test_contract_rejects_traversal_and_absolute_paths(tmp_path, source_path):
    root = tmp_path / "images"; root.mkdir(); manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({"image_id": "a", "source_path": source_path}) + "\n")
    specp = tmp_path / "s.json"; specp.write_text(json.dumps(spec()))
    result = run("scripts/embeddings/cache_ten_signal.py", "contract", "--manifest", str(manifest), "--image-root", str(root), "--spec", str(specp), "--contract", str(tmp_path / "c.json"))
    assert result.returncode != 0

def test_contract_rejects_symlink_manifest_path(tmp_path):
    root = tmp_path / "images"; root.mkdir(); outside = tmp_path / "outside.jpg"; outside.write_bytes(b"x"); (root / "linked.jpg").symlink_to(outside)
    manifest = tmp_path / "m.jsonl"; manifest.write_text(json.dumps({"image_id": "a", "source_path": "linked.jpg"}) + "\n")
    specp = tmp_path / "s.json"; specp.write_text(json.dumps(spec()))
    result = run("scripts/embeddings/cache_ten_signal.py", "contract", "--manifest", str(manifest), "--image-root", str(root), "--spec", str(specp), "--contract", str(tmp_path / "c.json"))
    assert result.returncode != 0

def test_status_and_failures_are_structured(tmp_path):
    cache, ledger, _ = seed_cache(tmp_path)
    ledger.seed_items([{"image_id": "bad", "manifest_position": 1, "source_path": "bad.jpg", "source_size": 1, "source_mtime_ns": 1, "source_fingerprint": None}])
    ledger.start_run("run2", model_revision="r1", preprocessing_hash="p", dataset_fingerprint="d", cache_schema_version=1, config={}); ledger.start_writer()
    ledger.emit("state", wait=True, image_id="bad", state="permanent_error", error_type="decode", error_message="broken"); ledger.stop()
    assert cli("status", cache) == {"completed": 1, "permanent_error": 1}
    assert cli("failures", cache) == [{"error_message": "broken", "error_type": "decode", "image_id": "bad", "state": "permanent_error"}]

def test_verify_valid_completed_row(tmp_path):
    cache, _, _ = seed_cache(tmp_path)
    assert cli("verify", cache) == {"errors": [], "ledger": "ok", "verified_shards": 1}

def test_verify_missing_shard(tmp_path):
    cache, _, shard = seed_cache(tmp_path); shard.unlink()
    assert {e["type"] for e in cli("verify", cache)["errors"]} == {"missing_shard"}

def test_verify_orphan_shard(tmp_path):
    cache, _, _ = seed_cache(tmp_path, shard=False)
    values = np.ones((1, 768), dtype=np.float32); values /= np.linalg.norm(values, axis=1, keepdims=True)
    write_atomic(cache / "shard-00007.safetensors", values, [7], ["orphan"])
    errors=cli("verify", cache)["errors"]
    assert "orphan_shard" in {e["type"] for e in errors}
    assert "incomplete_rows" in {e["type"] for e in errors}

def test_verify_corrupt_shard(tmp_path):
    cache, _, shard = seed_cache(tmp_path); shard.write_bytes(shard.read_bytes() + b"corrupt")
    errors = cli("verify", cache)["errors"]
    assert errors and errors[0]["shard"] == shard.name and "checksum" in errors[0]["error"]

def test_verify_checksum_mismatch_and_out_of_bounds(tmp_path):
    cache, _, _ = seed_cache(tmp_path)
    conn = sqlite3.connect(cache / "work.sqlite3"); conn.execute("UPDATE shards SET sha256=?", ("f" * 64,)); conn.execute("UPDATE items SET shard_row=4"); conn.commit(); conn.close()
    assert {"checksum_mismatch", "row_out_of_range", "unexpected_shard_row"} <= {e["type"] for e in cli("verify", cache)["errors"]}

def test_verify_coordinate_mismatch(tmp_path):
    cache, _, shard = seed_cache(tmp_path)
    metadata = shard.with_suffix(shard.suffix + ".json"); payload = json.loads(metadata.read_text()); payload["image_ids"] = ["different"]; metadata.write_text(json.dumps(payload, sort_keys=True))
    errors = cli("verify", cache)["errors"]
    assert any(e["type"] == "coordinate_mismatch" for e in errors)

def test_verify_errors_exit_nonzero_without_diagnostic_flag(tmp_path):
    cache, _, shard = seed_cache(tmp_path); shard.unlink()
    result = run("scripts/embeddings/cache_ten_signal.py", "verify", "--cache-dir", str(cache))
    assert result.returncode != 0
    assert "missing_shard" in {e["type"] for e in json.loads(result.stdout)["errors"]}
