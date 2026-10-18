import ast
from pathlib import Path

from image_aesthetics.data.fusion_holdout import DatasetExclusionView
from image_aesthetics.data.fusion_training_dataset import FusionTrainingDataset
from scripts.training.run_f5_lodo import DATASETS, SEEDS, _make_views, _parser, lodo_specs, plan


def test_lodo_matrix_is_four_by_two_by_two():
    specs = lodo_specs("F5-G-R15")
    assert len(specs) == 16
    assert {spec.held_out_dataset for spec in specs} == set(DATASETS)
    assert {spec.model for spec in specs} == {"F5-G-R15", "F4"}
    assert {spec.seed for spec in specs} == set(SEEDS)


def test_lodo_plan_is_read_only_and_explicit_about_selection_boundary(tmp_path):
    args = _parser().parse_args(["--data-root", str(tmp_path), "--candidate", "F5-G"])
    args.output_root = tmp_path / "fusion/experiments/f5-lodo-v1"
    value = plan(args)
    assert value["state"] == "planned"
    assert value["total"] == 16
    assert value["locked_test_accessed"] is False
    assert value["protocol"]["held_out_used_for_checkpoint_selection"] is False
    assert not args.output_root.exists()


def test_exclusion_view_filters_before_exposing_training_samples(monkeypatch):
    def fake_init(self, *_args, **_kwargs):
        self.partition = "train"; self.validation_fold = 0
        visible = set(_kwargs["dataset_ids"])
        self.samples = [(name, 0, {}, 0.5) for name in DATASETS if name in visible]
        self.datasets = {name: object() for name in visible}
        self.join_dirs = {}; self.split_manifest = Path("split.jsonl")
    monkeypatch.setattr(FusionTrainingDataset, "__init__", fake_init)
    monkeypatch.setattr(FusionTrainingDataset, "normalization_identity",
                        lambda self: {"schema_version": "base"})
    view = DatasetExclusionView(None, {name: Path(name) for name in DATASETS},
                                partition="train", validation_fold=0,
                                excluded_dataset="tad66k")
    assert {sample[0] for sample in view.samples} == set(DATASETS) - {"tad66k"}
    identity = view.normalization_identity()
    assert identity["excluded_dataset"] == "tad66k"
    assert identity["visible_dataset_ids"] == sorted(set(DATASETS) - {"tad66k"})


def test_base_dataset_filters_rows_before_join_lookup(tmp_path, monkeypatch):
    import json
    import image_aesthetics.data.fusion_training_dataset as module

    manifest = tmp_path / "samples.jsonl"
    rows = [
        {"dataset_id": "source", "image_id": "ok", "feature_id": "ok",
         "locked_test": False, "development_fold": 1, "native_score": 0.5,
         "native_score_range": [0, 1], "group_id": "source:1"},
        {"dataset_id": "heldout", "image_id": "must-not-be-joined",
         "feature_id": "must-not-be-joined", "locked_test": False,
         "development_fold": 1, "native_score": 0.5,
         "native_score_range": [0, 1], "group_id": "heldout:1"},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))

    class FakeJoined:
        def __init__(self, _path, **_kwargs):
            self.rows = [{"image_id": "ok"}]

    monkeypatch.setattr(module, "JoinedTenDataset", FakeJoined)
    dataset = FusionTrainingDataset(
        manifest, {"source": tmp_path / "source"}, partition="validation",
        validation_fold=1, dataset_ids={"source"})
    assert [sample[0] for sample in dataset.samples] == ["source"]


def test_source_views_do_not_open_heldout_join_index(monkeypatch, tmp_path):
    import scripts.training.run_f5_lodo as runner

    observed = []

    class FakeView:
        def __init__(self, split, joins, **kwargs):
            observed.append((set(joins), kwargs))
            self.samples = [(name, 0, {}, 0.5) for name in joins]

        def fit_normalizer(self, _path):
            return object()

        def normalized(self, _normalizer):
            return self

    joins = {name: tmp_path / name for name in DATASETS}
    monkeypatch.setattr(runner, "DatasetExclusionView", FakeView)
    monkeypatch.setattr(runner, "_paths", lambda _root: {"split": tmp_path / "split",
                                                            "joins": joins})
    train, validation, _normalizer, paths = _make_views(
        type("Args", (), {"data_root": tmp_path})(), "tad66k", tmp_path / "norm.json")
    assert len(observed) == 2
    assert all("tad66k" not in names for names, _kwargs in observed)
    assert paths["joins"] == {name: path for name, path in joins.items() if name != "tad66k"}
    assert paths["held_out_join"] == joins["tad66k"]


def test_lodo_runner_has_no_locked_test_constructor_or_override():
    source = Path("scripts/training/run_f5_lodo.py").read_text()
    assert "--locked-test" not in source
    tree = ast.parse(source)
    partitions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DatasetExclusionView":
            for keyword in node.keywords:
                if keyword.arg == "partition" and isinstance(keyword.value, ast.Constant):
                    partitions.add(keyword.value.value)
    assert partitions == {"train", "validation"}


def test_heldout_feature_view_is_deferred_until_source_checkpoint_is_frozen():
    source = Path("scripts/training/run_f5_lodo.py").read_text()
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    make_source_views = functions["_make_views"]
    assert not any(
        isinstance(node, ast.keyword) and node.arg == "only_dataset"
        for node in ast.walk(make_source_views)
    )
    make_holdout = functions["_make_holdout_view"]
    assert any(
        isinstance(node, ast.keyword) and node.arg == "only_dataset"
        for node in ast.walk(make_holdout)
    )
    freeze_at = source.index('model.load_state_dict(state["model"])')
    holdout_at = source.index("holdout_data = _make_holdout_view")
    assert holdout_at > freeze_at
