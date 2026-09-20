from pathlib import Path

from image_aesthetics.data.registry import acquisition_plan, load_registry, validate


REGISTRY = Path(__file__).parents[2] / "configs/data/datasets"


def test_initial_registry_is_valid_and_complete():
    records = load_registry(REGISTRY)
    assert {record.dataset_id for record in records} == {
        "tad66k", "para", "aadb", "flickr_aes", "spaq", "live_challenge",
        "piq23", "koniq10k", "eva",
    }
    assert validate(records) == []


def test_acquisition_plan_is_deterministic_and_license_gated():
    records = load_registry(REGISTRY)
    assert [record.dataset_id for record in acquisition_plan(records)] == ["live_challenge"]
    assert [record.dataset_id for record in acquisition_plan(records, "non-commercial-research")] == [
        "flickr_aes", "koniq10k", "live_challenge", "para",
    ]
    assert all(record.acquisition_decision("commercial-safe")[0] for record in acquisition_plan(records))
    assert not next(record for record in records if record.dataset_id == "tad66k").acquisition_decision("non-commercial-research")[0]


def test_validation_catches_missing_license_decision(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        '{"dataset_id":"bad","display_name":"Bad","task":"aesthetic",'
        '"official_url":"https://example.com","download_method":"manual",'
        '"declared_license":"","license_class":"unclear","commercial_status":"unclear",'
        '"redistribution_status":"unclear","source_version":"v1",'
        '"expected_files":["images/"],"known_upstream_overlap":["unknown"],"enabled":true}'
    )
    errors = validate(load_registry(tmp_path))
    assert any("declared_license must be non-empty" in error for error in errors)


def test_malformed_lists_and_json_include_filename(tmp_path):
    (tmp_path / "bad.json").write_text('{"expected_files": "images"}')
    try:
        load_registry(tmp_path)
    except ValueError as error:
        assert "bad.json:" in str(error)
        assert "missing required fields" in str(error)
    else:
        raise AssertionError("malformed registry should fail")

    (tmp_path / "bad.json").write_text(
        '{"dataset_id":"bad","display_name":"Bad","task":"aesthetic",'
        '"official_url":"https://example.com","download_method":"manual",'
        '"declared_license":"x","license_class":"unclear","commercial_status":"unclear",'
        '"redistribution_status":"unclear","source_version":"v1",'
        '"expected_files":[],"known_upstream_overlap":["AVA"],"enabled":true}'
    )
    try:
        load_registry(tmp_path)
    except ValueError as error:
        assert "bad.json:" in str(error)
        assert "expected_files must be a non-empty list" in str(error)
    else:
        raise AssertionError("empty expected_files should fail")


def test_empty_registry_is_rejected(tmp_path):
    try:
        load_registry(tmp_path)
    except ValueError as error:
        assert "registry is empty" in str(error)
    else:
        raise AssertionError("empty registry should fail")
