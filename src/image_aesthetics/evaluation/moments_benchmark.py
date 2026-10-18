"""Privacy-safe metrics for the event-grouped Moments benchmark."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping


def evaluate_moments_predictions(
    manifest: Iterable[Mapping], annotations: Iterable[Mapping],
    predictions: Iterable[Mapping], *, top_fractions=(0.01, 0.05),
) -> dict:
    """Evaluate fixed candidate scores against real, blinded human labels.

    Missing human labels are reported as pending rather than synthesized.
    Pairwise tasks must stay within one event. Severe false positives are
    rejected items selected in a candidate's top-k within an event.
    """
    items = {str(row["item_id"]): dict(row) for row in manifest}
    if len(items) == 0:
        raise ValueError("empty benchmark manifest")
    by_model: dict[str, dict[str, float]] = defaultdict(dict)
    for row in predictions:
        item_id, model = str(row["item_id"]), str(row["model"])
        if item_id not in items:
            raise ValueError(f"prediction references unknown item: {item_id}")
        score = float(row["score"])
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score outside [0,1]: {model}:{item_id}")
        if item_id in by_model[model]:
            raise ValueError(f"duplicate prediction: {model}:{item_id}")
        by_model[model][item_id] = score
    if not by_model:
        raise ValueError("no candidate predictions")

    pairs, keep = [], {}
    top_choices: dict[tuple[str, str], set[str]] = {}
    for row in annotations:
        kind = str(row["kind"])
        if kind == "pairwise":
            left, right, winner = map(str, (row["left_id"], row["right_id"], row["winner_id"]))
            if left not in items or right not in items or winner not in {left, right}:
                raise ValueError("invalid pairwise annotation")
            if items[left]["event_id"] != items[right]["event_id"]:
                raise ValueError("pairwise annotation crosses events")
            pairs.append((left, right, winner))
        elif kind == "keep_reject":
            item_id = str(row["item_id"])
            if item_id not in items:
                raise ValueError("keep/reject annotation references unknown item")
            keep[item_id] = bool(row["keep"])
        elif kind == "topk":
            event_id, task_id = str(row["event_id"]), str(row["task_id"])
            selected = {str(value) for value in row["selected_ids"]}
            if any(value not in items or str(items[value]["event_id"]) != event_id
                   for value in selected):
                raise ValueError("top-k annotation crosses events or references unknown item")
            top_choices[(event_id, task_id)] = selected
        else:
            raise ValueError(f"unknown annotation kind: {kind}")

    events: dict[str, list[str]] = defaultdict(list)
    for item_id, row in items.items():
        events[str(row["event_id"])].append(item_id)
    reports = {}
    for model, scores in sorted(by_model.items()):
        missing = set(items) - set(scores)
        if missing:
            raise ValueError(f"{model} lacks {len(missing)} benchmark predictions")
        pair_correct = sum((scores[left] > scores[right]) == (winner == left)
                           for left, right, winner in pairs)
        metrics = {"pairwise_accuracy": (pair_correct / len(pairs) if pairs else None),
                   "pairwise_n": len(pairs), "top_fractions": {}}
        for fraction in top_fractions:
            selected = set()
            for event_items in events.values():
                count = max(1, round(len(event_items) * float(fraction)))
                selected.update(sorted(event_items, key=lambda value: (-scores[value], value))[:count])
            labeled = [value for value in selected if value in keep]
            severe = sum(not keep[value] for value in labeled)
            human_selected = set().union(*top_choices.values()) if top_choices else set()
            metrics["top_fractions"][str(fraction)] = {
                "selected_n": len(selected), "keep_reject_labeled_n": len(labeled),
                "severe_false_positive_rate": (severe / len(labeled) if labeled else None),
                "human_topk_overlap": (len(selected & human_selected) / len(human_selected)
                                        if human_selected else None),
            }
        reports[model] = metrics
    return {"schema_version": "moments-benchmark-evaluation-v1",
            "human_labels_present": bool(pairs or keep or top_choices),
            "annotation_counts": {"pairwise": len(pairs), "keep_reject": len(keep),
                                  "topk_tasks": len(top_choices)},
            "models": reports}
