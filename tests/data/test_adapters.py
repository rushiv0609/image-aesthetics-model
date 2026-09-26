from __future__ import annotations

import io
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from image_aesthetics.data.adapters import CanonicalSample, KonIQ10KAdapter, SPAQAdapter, TAD66KAdapter


def test_tad66k_merges_splits_preserves_native_scale_and_rejects_bad_rows(tmp_path: Path):
    archive_path = tmp_path / "labels.zip"
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("labels/merge/train.csv", "image,score\na.jpg,2.5\nb.jpg,10\ninvalid.jpg,11\n")
        archive.writestr("labels/merge/test.csv", "image,score\nc.jpg,1.25\na.jpg,3\n")
        archive.writestr("labels/unmerge/train/nature.csv", "image,score\na.jpg,2.5\n")
        archive.writestr("labels/unmerge/train/sunset.csv", "image,score\na.jpg,2.5\nb.jpg,10\n")
        archive.writestr("labels/unmerge/test/city.csv", "image,score\nc.jpg,1.25\n")

    adapter = TAD66KAdapter(archive_path)
    samples = list(adapter.iter_samples())
    assert [sample.image_id for sample in samples] == ["a.jpg", "b.jpg", "c.jpg"]
    assert all(isinstance(sample, CanonicalSample) for sample in samples)
    assert samples[0].native_score_range == (1.0, 10.0)
    assert samples[0].split == "train"
    assert samples[0].themes == ("nature", "sunset")
    assert samples[0].source_groups == ("tad66k:image:a.jpg",)
    assert samples[2].split == "test"
    audit = adapter.inventory()
    assert (audit.accepted, audit.rejected) == (3, 2)
    assert audit.rejection_reasons == {"score outside 1-10": 1, "duplicate image name": 1}


def _xlsx(rows: list[list[str]]) -> bytes:
    def cell(column: int, value: str) -> str:
        ref = f"{chr(ord('A') + column)}"
        escaped = value.replace("&", "&amp;").replace("<", "&lt;")
        if column == 0 or not value.replace(".", "", 1).isdigit():
            return f'<c r="{ref}1" t="inlineStr"><is><t>{escaped}</t></is></c>'
        return f'<c r="{ref}1"><v>{escaped}</v></c>'

    # The adapter only needs worksheet XML; inline strings avoid shared-string complexity.
    xml_rows = []
    for row_number, row in enumerate(rows, 1):
        cells = []
        for column, value in enumerate(row):
            ref = ""
            n = column + 1
            while n:
                n, remainder = divmod(n - 1, 26)
                ref = chr(65 + remainder) + ref
            escaped = value.replace("&", "&amp;").replace("<", "&lt;")
            if row_number == 1 or not value.replace(".", "", 1).lstrip("-").isdigit():
                cells.append(f'<c r="{ref}{row_number}" t="inlineStr"><is><t>{escaped}</t></is></c>')
            else:
                cells.append(f'<c r="{ref}{row_number}"><v>{escaped}</v></c>')
        xml_rows.append(f'<row r="{row_number}">{"".join(cells)}</row>')
    worksheet = (f'<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>').encode()
    workbook = io.BytesIO()
    with ZipFile(workbook, "w", ZIP_DEFLATED) as archive:
        archive.writestr("xl/worksheets/sheet1.xml", worksheet)
    return workbook.getvalue()


def test_spaq_joins_mos_attributes_scenes_and_exif_without_openpyxl(tmp_path: Path):
    archive_path = tmp_path / "annotations.zip"
    mos = [["Image name", "MOS", "Brightness", "Colorfulness", "Contrast", "Noisiness", "Sharpness"], ["00002.jpg", "62.5", "55.38", "67.13", "53.13", "57.5", "47.5"], ["bad.jpg", "101", "1", "1", "1", "1", "1"]]
    scenes = [["Image name", "Animal", "Landscape", "Others"], ["00002.jpg", "0", "0.5", "0"]]
    exif = [["Image name", "Focal length", "Flash"], ["00002.jpg", "4.04", "0"]]
    with ZipFile(archive_path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("Annotations/MOS and Image attribute scores.xlsx", _xlsx(mos))
        archive.writestr("Annotations/Scene category labels.xlsx", _xlsx(scenes))
        archive.writestr("Annotations/EXIF_tags.xlsx", _xlsx(exif))

    adapter = SPAQAdapter(archive_path)
    samples = list(adapter.iter_samples())
    assert len(samples) == 1
    sample = samples[0]
    assert sample.native_score == 62.5
    assert sample.native_score_range == (0.0, 100.0)
    assert sample.attributes == {"brightness": 55.38, "colorfulness": 67.13, "contrast": 53.13, "noisiness": 57.5, "sharpness": 47.5}
    assert sample.themes == ("Landscape",)
    assert sample.exif == {"Focal length": "4.04", "Flash": "0"}
    audit = adapter.inventory()
    assert audit.accepted == 1
    assert audit.rejection_reasons == {"MOS outside 0-100": 1}


def test_koniq_adapter_keeps_score_rows_and_does_not_infer_indicator_join(tmp_path: Path):
    scores = tmp_path / "koniq10k_scores_and_distributions.csv"
    scores.write_text(
        "image_name,c1,c2,c3,c4,c5,c_total,MOS,SD,MOS_zscore\n"
        "one.jpg,1,2,3,4,5,15,3.5,0.5,72.1\n"
        "bad.jpg,1,2,3,4,5,15,5.5,0.5,72.1\n"
        "one.jpg,1,2,3,4,5,15,3.5,0.5,72.1\n"
    )
    adapter = KonIQ10KAdapter(scores)
    samples = list(adapter.iter_samples())
    assert len(samples) == 1
    assert samples[0].image_id == "one"
    assert samples[0].native_score_range == (1.0, 5.0)
    assert samples[0].attributes == {"rating_standard_deviation": 0.5, "rating_count": 15.0}
    assert samples[0].metadata["rating_distribution"] == {"c1": 1, "c2": 2, "c3": 3, "c4": 4, "c5": 5}
    assert adapter.inventory().rejection_reasons == {"MOS outside 1-5": 1, "duplicate image name": 1}
