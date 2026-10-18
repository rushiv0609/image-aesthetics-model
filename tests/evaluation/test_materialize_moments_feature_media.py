import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts.evaluation.materialize_moments_feature_media import materialize


def test_materializes_images_heic_and_video_with_fixed_contract(tmp_path):
    sources = tmp_path / "sources"; sources.mkdir()
    image = sources / "a.jpg"; image.write_bytes(b"jpeg")
    heic = sources / "b.heic"; heic.write_bytes(b"heic")
    video = sources / "c.mov"; video.write_bytes(b"video")
    rows = [
        {"item_id": "a", "media_path": str(image), "media_type": "image"},
        {"item_id": "b", "media_path": str(heic), "media_type": "image"},
        {"item_id": "c", "media_path": str(video), "media_type": "video"},
    ]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    commands = []

    def runner(command):
        commands.append(command)
        if command[0] == "ffprobe":
            return CompletedProcess(command, 0, stdout="12.0\n", stderr="")
        destination = Path(command[-1])
        destination.write_bytes(b"derived")
        return CompletedProcess(command, 0, stdout="", stderr="")

    report = materialize(manifest, tmp_path / "output", runner=runner)
    derived = [json.loads(line) for line in
               (tmp_path / "output/features.jsonl").read_text().splitlines()]
    assert report["items"] == 3 and report["videos"] == 1
    assert {row["materialization_method"] for row in derived} == {
        "hard-link-original", "heic-sips-jpeg95", "video-midpoint-ffmpeg"}
    ffmpeg = next(command for command in commands if command[0] == "ffmpeg")
    assert ffmpeg[ffmpeg.index("-ss") + 1] == "6.000000"
    assert all(row["locked_test"] is False for row in derived)


def test_refuses_to_overwrite_existing_output(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"item_id": "a", "media_path": str(manifest),
                                    "media_type": "image"}) + "\n")
    output = tmp_path / "output"; output.mkdir(); (output / "evidence").write_text("x")
    with pytest.raises(FileExistsError):
        materialize(manifest, output)
