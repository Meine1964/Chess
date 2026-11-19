import hashlib
import json
import os
from pathlib import Path

import pytest
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ASSET_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "chess_camera"
    / "assets"
    / "pieces"
    / "cburnett"
)


@pytest.fixture(scope="module")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def test_cburnett_assets_match_pinned_manifest(
    application: QApplication,
) -> None:
    manifest = json.loads(
        (_ASSET_DIRECTORY / "ASSET_MANIFEST.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["revision"] == (
        "5b44d719ec5913dd62aa965f4b8d21e60a00c2c1"
    )
    assert manifest["license"] == "GPL-2.0-or-later"
    assert len(manifest["sha256"]) == 12
    for filename, expected_hash in manifest["sha256"].items():
        asset_path = _ASSET_DIRECTORY / filename
        assert asset_path.is_file()
        assert hashlib.sha256(asset_path.read_bytes()).hexdigest() == (
            expected_hash
        )
        assert QSvgRenderer(str(asset_path)).isValid()


def test_cburnett_assets_include_license_and_attribution() -> None:
    notice = (_ASSET_DIRECTORY / "THIRD_PARTY_NOTICE.md").read_text(
        encoding="utf-8"
    )
    license_text = (
        _ASSET_DIRECTORY / "LICENSE-GPL-2.0-or-later.txt"
    ).read_text(encoding="utf-8")

    assert "Colin M. L. Burnett" in notice
    assert "GPLv2" in notice
    assert "GNU GENERAL PUBLIC LICENSE" in license_text
