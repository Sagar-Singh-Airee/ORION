from pathlib import Path

from orion.data.dicom.reader import discover_series


def test_empty_directory_has_no_series(tmp_path: Path):
    assert discover_series(tmp_path) == {}
