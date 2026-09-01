from __future__ import annotations

import json
from pathlib import Path

import pytest

from label_audit import io
from label_audit.errors import DatasetFormatError


def write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class TestCsv:
    def test_minimal_csv_loads(self, tmp_path: Path) -> None:
        data = io.load(write(tmp_path / "d.csv", "id,text,label\na,hello,pos\n"))
        assert len(data) == 1
        assert data.rows[0].item_id == "a"
        assert data.rows[0].text == "hello"
        assert data.rows[0].label == "pos"
        assert data.rows[0].split == "train"
        assert data.rows[0].annotator is None

    def test_split_and_annotator_columns_are_used(self, tmp_path: Path) -> None:
        content = "id,text,label,split,annotator\na,hi,pos,test,priya\n"
        data = io.load(write(tmp_path / "d.csv", content))
        assert data.rows[0].split == "test"
        assert data.rows[0].annotator == "priya"

    def test_extra_columns_become_features(self, tmp_path: Path) -> None:
        content = "id,text,label,source,region\na,hi,pos,web,eu\n"
        data = io.load(write(tmp_path / "d.csv", content))
        assert data.rows[0].features == {"source": "web", "region": "eu"}

    def test_missing_required_column_names_it(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="label"):
            io.load(write(tmp_path / "d.csv", "id,text\na,hi\n"))

    def test_missing_required_value_names_row_and_field(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match=r"d\.csv:3.*label"):
            io.load(write(tmp_path / "d.csv", "id,text,label\na,hi,pos\nb,ho,\n"))

    def test_empty_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="empty"):
            io.load(write(tmp_path / "d.csv", ""))

    def test_header_only_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="no data rows"):
            io.load(write(tmp_path / "d.csv", "id,text,label\n"))

    def test_blank_split_falls_back_to_train(self, tmp_path: Path) -> None:
        content = "id,text,label,split\na,hi,pos,\n"
        data = io.load(write(tmp_path / "d.csv", content))
        assert data.rows[0].split == "train"


class TestJsonl:
    def test_minimal_jsonl_loads(self, tmp_path: Path) -> None:
        line = json.dumps({"id": "a", "text": "hello", "label": "pos"})
        data = io.load(write(tmp_path / "d.jsonl", line + "\n"))
        assert data.rows[0].item_id == "a"

    def test_ndjson_extension_is_accepted(self, tmp_path: Path) -> None:
        line = json.dumps({"id": "a", "text": "hello", "label": "pos"})
        data = io.load(write(tmp_path / "d.ndjson", line + "\n"))
        assert len(data) == 1

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        line = json.dumps({"id": "a", "text": "hi", "label": "pos"})
        data = io.load(write(tmp_path / "d.jsonl", f"{line}\n\n{line}\n"))
        assert len(data) == 2

    def test_numbers_and_booleans_coerce_to_stable_strings(self, tmp_path: Path) -> None:
        obj = {"id": "a", "text": "hi", "label": "pos", "count": 3, "flag": True}
        data = io.load(write(tmp_path / "d.jsonl", json.dumps(obj) + "\n"))
        assert data.rows[0].features == {"count": "3", "flag": "true"}

    def test_null_feature_becomes_empty_string(self, tmp_path: Path) -> None:
        obj = {"id": "a", "text": "hi", "label": "pos", "source": None}
        data = io.load(write(tmp_path / "d.jsonl", json.dumps(obj) + "\n"))
        assert data.rows[0].features == {"source": ""}

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match=r"d\.jsonl:2"):
            io.load(write(tmp_path / "d.jsonl", '{"id":"a","text":"t","label":"x"}\n{oops\n'))

    def test_non_object_line_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="object per line"):
            io.load(write(tmp_path / "d.jsonl", "[1,2]\n"))

    def test_nested_container_feature_is_refused_not_flattened(self, tmp_path: Path) -> None:
        obj = {"id": "a", "text": "hi", "label": "pos", "meta": {"a": 1}}
        with pytest.raises(DatasetFormatError, match="meta.*not a scalar"):
            io.load(write(tmp_path / "d.jsonl", json.dumps(obj) + "\n"))


class TestDispatch:
    def test_unknown_extension_is_refused_not_guessed(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="unsupported input"):
            io.load(write(tmp_path / "d.parquet", "x"))

    def test_missing_file_reports_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetFormatError, match="cannot read"):
            io.load(tmp_path / "absent.csv")
