import pytest

from research_analyst.ui import gallery


@pytest.fixture(autouse=True)
def _empty_gallery_by_default(monkeypatch, tmp_path):
    """UI tests must not depend on whichever sample reports happen to be bundled; gallery tests opt in."""
    monkeypatch.setattr(gallery, "GALLERY_DIR", tmp_path / "no-gallery")
