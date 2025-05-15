# tests/test_plots.py
from pathlib import Path
import importlib
import visualise


def test_phase_d_outputs(tmp_path: Path, monkeypatch):
    """
    Smoke-test: run visualise.render_all() on a tiny slice of the dataset
    and assert at least one PNG & HTML are created and non-empty.
    """
    # redirect output dirs to tmp_path
    monkeypatch.setattr(visualise, "_DIR_PNG", tmp_path / "png")
    monkeypatch.setattr(visualise, "_DIR_HTML", tmp_path / "html")

    visualise.render_all()

    png_files = list((tmp_path / "png").glob("*.png"))
    html_files = list((tmp_path / "html").glob("*.html"))
    assert png_files and html_files, "No figures generated"

    # check first file is not empty
    assert png_files[0].stat().st_size > 0
    assert b"<html" in html_files[0].read_bytes()[:100].lower()
