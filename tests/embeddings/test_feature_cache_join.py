from __future__ import annotations

import hashlib

from scripts.embeddings.join_feature_caches import _identity_mismatches, _rows_sha256, _write_jsonl


def test_join_index_digest_matches_atomically_written_jsonl(tmp_path):
    rows = [{"dataset_id": "fixture", "image_id": "a", "source_sha256": "0" * 64},
            {"dataset_id": "fixture", "image_id": "b", "source_sha256": "1" * 64}]
    path = tmp_path / "join.jsonl"
    _write_jsonl(path, rows)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == _rows_sha256(rows)


def test_join_uses_canonical_manifest_identity_across_cache_generations():
    legacy = {
        "manifest_sha256": "canonical-hash",
        "canonical_items_sha256": "canonical-hash",
        "source_fingerprints_sha256": "source-hash",
        "item_count": "2",
        "ordered_ids_sha256": "ids-hash",
    }
    modern = {
        "manifest_sha256": "raw-bytes-hash",
        "manifest_bytes_sha256": "raw-bytes-hash",
        "canonical_items_sha256": "canonical-hash",
        "source_fingerprints_sha256": "source-hash",
        "item_count": "2",
        "ordered_ids_sha256": "ids-hash",
    }
    assert _identity_mismatches(legacy, modern) == {}


def test_join_rejects_different_canonical_manifest_identity():
    left = {"canonical_items_sha256": "left", "source_fingerprints_sha256": "same",
            "item_count": "2", "ordered_ids_sha256": "ids"}
    right = {**left, "canonical_items_sha256": "right"}
    assert "canonical_items_sha256" in _identity_mismatches(left, right)
