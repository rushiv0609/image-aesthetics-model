from pathlib import Path

from scripts.evaluation.prepare_moments_benchmark import prepare


def test_prepare_uses_opaque_ids_and_groups_by_time(monkeypatch, tmp_path):
    paths = [tmp_path / name for name in ("a.jpg", "b.jpg", "c.mov")]
    for path in paths:
        path.write_bytes(b"x")
    stamps = {"a.jpg": 0.0, "b.jpg": 100.0, "c.mov": 30_000.0}
    monkeypatch.setattr("scripts.evaluation.prepare_moments_benchmark._timestamp",
                        lambda path: stamps[path.name])
    rows = prepare(tmp_path, b"s" * 32, event_gap_hours=1)
    assert len(rows) == 3
    assert rows[0]["event_id"] == rows[1]["event_id"]
    assert rows[1]["event_id"] != rows[2]["event_id"]
    assert all(tmp_path.name not in row["item_id"] for row in rows)
    assert rows[2]["media_type"] == "video"
