import os
import sqlite3
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit,
)
from PySide6.QtCore import QPoint, QRect, QSettings, Qt
from PySide6.QtTest import QTest

from src.chess_camera.vision_data import VisionDataset, VisionSample, demonstration_samples
from src.chess_camera.vision_tools import (
    VisionDataDialog, VisionReviewDialog, install_vision_tools, vision_discontinuity,
    vision_hand_observation, vision_move_accepted, vision_submission,
)


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def vision_settings(tmp_path, monkeypatch):
    class IsolatedSettings(QSettings):
        def __init__(self, *args):
            super().__init__(str(tmp_path / "vision.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr("src.chess_camera.vision_tools.QSettings", IsolatedSettings)
    return IsolatedSettings()


@pytest.fixture
def dataset(tmp_path):
    image = np.full((300, 300, 3), 100, np.uint8)
    grid = np.zeros((9, 9, 2), np.float32)
    grid[:, :, 0] = np.linspace(50, 250, 9)[None, :]
    grid[:, :, 1] = np.linspace(50, 250, 9)[:, None]
    _, samples = demonstration_samples(image, grid)
    result = VisionDataset.create(tmp_path)
    result.save(samples, 20_000_000)
    return result


def test_tools_menu_is_passive_until_explicit_user_action(application):
    window = QMainWindow()
    tools = install_vision_tools(window)
    try:
        assert not tools.collecting
        assert tools.collector is None
        assert tools.dialog is None
        assert not tools.timer.isActive()
        menu = window.menuBar().actions()[0].menu()
        assert [action.text() for action in menu.actions()] == [
            "Vision data...", "Review vision labels...",
        ]
        menu.actions()[1].trigger()
        assert tools.reviewer is not None
        assert tools.reviewer.isVisible()
        assert tools.collector is None
        assert not tools.collecting
    finally:
        window.close()
        if tools.reviewer is not None:
            tools.reviewer.close()


def test_review_corrects_both_kinds_and_persists_original_labels(application, dataset):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(1)
        first = dialog.current()
        assert first is not None and first.kind == "occupancy"
        dialog.set_label(1)
        stored = next(item for item in dataset.records() if item.sample_id == first.sample_id)
        assert stored.label == "occupied"
        assert stored.original_label == "unknown"
        dialog.kind.setCurrentIndex(2)
        hand = dialog.current()
        assert hand is not None and hand.kind == "hand"
        dialog.set_label(1)
        stored = next(item for item in dataset.records() if item.sample_id == hand.sample_id)
        assert stored.label == "crossing"
        assert stored.reviewed
        assert dataset.records(training_only=True) == []
        dialog.discard()
        assert len(dataset.records()) == 7
    finally:
        dialog.close()


def test_review_reopen_next_and_empty_dataset(application, dataset, tmp_path):
    first = VisionReviewDialog(dataset=dataset)
    identifier = first.current().sample_id
    first.navigate(1)
    first.close()
    stored = next(item for item in dataset.records() if item.sample_id == identifier)
    assert not stored.reviewed
    reopened = VisionReviewDialog(dataset=VisionDataset(dataset.path.parent))
    empty = VisionReviewDialog(dataset=VisionDataset.create(tmp_path / "empty"))
    try:
        assert len(reopened.records) == 8
        assert not empty.accept_button.isEnabled()
        assert not empty.previous_button.isEnabled()
        assert not empty.next_button.isEnabled()
        empty.navigate(-1)
        empty.navigate(1)
        assert empty.current() is None
        empty.accept_label()
        empty.discard()
    finally:
        reopened.close()
        empty.close()


def test_disabled_observers_do_not_access_gameplay_state():
    window = SimpleNamespace()
    vision_move_accepted(window, {}, {})
    vision_submission(window, {}, {})
    vision_hand_observation(window, True)
    vision_discontinuity(window, "test")
    assert vars(window) == {}


def test_review_buttons_and_keyboard_do_not_close_the_dialog(application, dataset):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(1)
        dialog.show()
        application.processEvents()
        identifier = dialog.current().sample_id
        dialog.primary_label.setCurrentText("occupied")
        QTest.mouseClick(dialog.accept_button, Qt.MouseButton.LeftButton)
        stored = next(item for item in dataset.records() if item.sample_id == identifier)
        assert stored.label == "occupied"
        next_identifier = dialog.current().sample_id
        QTest.keyClick(dialog, Qt.Key.Key_Return)
        application.processEvents()
        stored = next(item for item in dataset.records() if item.sample_id == next_identifier)
        assert stored.reviewed
        assert dialog.isVisible()
    finally:
        dialog.close()


def test_dropdown_drafts_save_together_and_navigation_discards(application, dataset):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(1)
        identifier = dialog.current().sample_id
        dialog.primary_label.setCurrentText("occupied")
        dialog.piece_colour.setCurrentText("white")
        assert not dataset.records()[0].reviewed
        dialog.navigate(1)
        dialog.navigate(-1)
        assert dialog.primary_label.currentText() == "unknown"
        assert dialog.piece_colour.currentText() == "unknown"
        dialog.primary_label.setCurrentText("occupied")
        dialog.piece_colour.setCurrentText("black")
        dialog.accept_label()
        stored = next(item for item in dataset.records() if item.sample_id == identifier)
        assert stored.label == "occupied"
        assert stored.metadata["piece_colour"] == "black"
        assert stored.metadata["piece_colour_source"] == "reviewer"
        dialog.navigate(-1)
        assert dialog.piece_colour.currentText() == "black"
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", [1, 2])
def test_review_labels_fit_without_scrolling_and_details_are_separate(
    application, dataset, kind,
):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(kind)
        if kind == 1:
            dialog.index = next(
                index for index, item in enumerate(dialog.records)
                if item.metadata["physical_square"] == [0, 0]
            )
            dialog.display()
        dialog.resize(960, 640)
        dialog.show()
        application.processEvents()
        assert dialog.size().width() == 960
        assert dialog.size().height() == 640
        for field in (
            dialog.output_group, dialog.observation_group, dialog.after_caption,
            dialog.primary_label, dialog.primary_label_title,
            dialog.accept_button, dialog.description, dialog.status,
            *((dialog.input_group, dialog.piece_colour, dialog.square_colour, dialog.location)
              if kind == 1 else ()),
            *((dialog.before_caption, dialog.baseline_label) if kind == 2 else ()),
        ):
            bounds = QRect(field.mapTo(dialog, QPoint()), field.size())
            assert field.isVisible()
            assert dialog.rect().contains(bounds)
        assert dialog.piece_colour.isEnabled() == (kind == 1)
        assert dialog.piece_colour.isVisible() == (kind == 1)
        assert dialog.square_colour.isVisible() == (kind == 1)
        assert dialog.location.isVisible() == (kind == 1)
        assert dialog.input_group.isVisible() == (kind == 1)
        assert "Images:" not in [label.text() for label in dialog.findChildren(QLabel)]
        assert dialog.primary_label_title.text() == (
            "Occupancy:" if kind == 1 else "Current obstruction:"
        )
        assert dialog.baseline_label.isVisible() == (kind == 2)
        assert dialog.before_panel.isVisible() == (kind == 2)
        assert dialog.after_caption.text().startswith(
            "Square image" if kind == 1 else "Current observation"
        )
        assert not dialog.findChildren(QPlainTextEdit)
        if kind == 1:
            assert dialog.location.text() == "Corner: top, left"
            assert dialog.square_colour.text() == "Not recorded"
        for image in (dialog.before, dialog.after):
            pixmap = image.pixmap()
            if pixmap is not None and not pixmap.isNull():
                assert pixmap.width() <= image.width()
                assert pixmap.height() <= image.height()
        dialog.show_details()
        application.processEvents()
        details = dialog.findChild(QPlainTextEdit)
        assert details is not None and details.isReadOnly()
        assert details.window() is not dialog
        assert "camera_grid" in details.toPlainText()
        for child in dialog.findChildren(QDialog):
            child.close()
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", [1, 2])
def test_navigation_and_saving_stop_at_filtered_boundaries(application, dataset, kind):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(kind)
        dialog.index = 0
        dialog.display()
        dialog.show()
        application.processEvents()
        first = dialog.current().sample_id
        assert not dialog.previous_button.isEnabled()
        assert dialog.next_button.isEnabled()
        draft = "occupied" if kind == 1 else "crossing"
        dialog.primary_label.setCurrentText(draft)
        dialog.navigate(-1)
        QTest.keyClick(dialog, Qt.Key.Key_Left)
        assert dialog.current().sample_id == first
        assert dialog.primary_label.currentText() == draft
        dialog.navigate(len(dialog.records))
        last = dialog.current().sample_id
        assert dialog.previous_button.isEnabled()
        assert not dialog.next_button.isEnabled()
        assert dialog.accept_button.text() == "Save (Enter)"
        dialog.primary_label.setCurrentText(draft)
        dialog.navigate(1)
        QTest.keyClick(dialog, Qt.Key.Key_Right)
        assert dialog.current().sample_id == last
        assert dialog.primary_label.currentText() == draft
        dialog.accept_label()
        assert dialog.current().sample_id == last
        stored = next(item for item in dataset.records() if item.sample_id == last)
        assert stored.label == draft
        assert stored.reviewed
        dialog.discard()
        assert dialog.index == len(dialog.records) - 1
        assert not dialog.next_button.isEnabled()
    finally:
        dialog.close()


def test_single_example_can_save_without_navigation(application, dataset, tmp_path):
    source = next(item for item in dataset.records() if item.kind == "occupancy")
    single = VisionDataset.create(tmp_path / "single")
    single.save([VisionSample(
        "occupancy", "unknown", dataset.images(source.sample_id), {}, "single", True,
    )], 20_000_000)
    dialog = VisionReviewDialog(dataset=single)
    try:
        identifier = dialog.current().sample_id
        assert not dialog.previous_button.isEnabled()
        assert not dialog.next_button.isEnabled()
        dialog.primary_label.setCurrentText("occupied")
        dialog.accept_label()
        assert dialog.current().sample_id == identifier
        assert single.records()[0].label == "occupied"
        assert not dialog.previous_button.isEnabled()
        assert not dialog.next_button.isEnabled()
    finally:
        dialog.close()


def test_stored_inputs_and_targets_are_grouped_without_exposing_audit_fields(
    application, dataset, tmp_path,
):
    source = next(item for item in dataset.records() if item.kind == "occupancy")
    labelled = VisionDataset.create(tmp_path / "labelled")
    labelled.save([VisionSample(
        "occupancy", "occupied", dataset.images(source.sample_id),
        {"square_colour": "dark", "physical_square": [0, 7],
         "piece_colour": "black", "piece_colour_source": "logical-position",
         "projector_state": "black"},
        "labelled", True,
    )], 20_000_000)
    dialog = VisionReviewDialog(dataset=labelled)
    try:
        assert dialog.input_group.isAncestorOf(dialog.square_colour)
        assert dialog.input_group.isAncestorOf(dialog.location)
        assert not dialog.input_group.findChildren(QComboBox)
        assert dialog.output_group.isAncestorOf(dialog.primary_label)
        assert dialog.output_group.isAncestorOf(dialog.piece_colour)
        assert dialog.primary_label.currentText() == "occupied"
        assert dialog.piece_colour.currentText() == "black"
        assert dialog.square_colour.text() == "dark"
        assert dialog.location.text() == "Corner: top, right"
        main_texts = [label.text() for label in dialog.findChildren(QLabel)]
        assert "Colour source:" not in main_texts
        assert "Projector state:" not in main_texts
        dialog.show_details()
        details = dialog.findChild(QPlainTextEdit).toPlainText()
        assert '"piece_colour_source": "logical-position"' in details
        assert '"projector_state": "black"' in details
        assert not labelled.records()[0].reviewed
        for child in dialog.findChildren(QDialog):
            child.close()
    finally:
        dialog.close()


def test_hand_captions_and_navigation_follow_current_images_and_their_references(
    application, tmp_path,
):
    dataset = VisionDataset.create(tmp_path / "linked")
    samples = []
    mask = np.full((48, 48), 255, np.uint8)
    for baseline_colour, current_colour in ((20, 80), (40, 160), (20, 180)):
        samples.append(VisionSample(
            "hand", "unknown",
            {"baseline": np.full((48, 48, 3), baseline_colour, np.uint8),
             "current": np.full((48, 48, 3), current_colour, np.uint8),
             "baseline_valid": mask, "current_valid": mask},
            {}, "linked", True,
        ))
    identifiers = dataset.save(samples, 1_000_000)
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        for index, (baseline_colour, current_colour) in enumerate(
            ((20, 80), (40, 160), (20, 180))
        ):
            assert dialog.current().sample_id == identifiers[index]
            assert dialog.before.original.toImage().pixelColor(0, 0).red() == baseline_colour
            assert dialog.after.original.toImage().pixelColor(0, 0).red() == current_colour
            assert dialog.before_caption.text().startswith("Linked baseline")
            assert dialog.after_caption.text() == "Current observation"
            assert f"Observation {index + 1} of 3" in dialog.observation_group.title()
            assert f"({2 if index != 1 else 1} observations)" in dialog.before_caption.text()
            dialog.navigate(1)
        assert dialog.current().sample_id == identifiers[-1]
    finally:
        dialog.close()


def test_shared_baseline_review_confirmation_can_cancel_without_saving(
    application, dataset, monkeypatch,
):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(2)
        dialog.index = 0
        dialog.display()
        identifier = dialog.current().sample_id
        before = dataset.records()
        dialog.primary_label.setCurrentText("crossing")
        dialog.baseline_label.setCurrentText("invalid")
        questions = []

        def decline(*args):
            questions.append(args[2])
            return QMessageBox.StandardButton.No

        monkeypatch.setattr(QMessageBox, "question", decline)
        dialog.accept_label()
        assert "2 observations" in questions[0]
        assert dataset.records() == before
        assert dialog.current().sample_id == identifier
        assert dialog.baseline_label.currentText() == "invalid"
        monkeypatch.setattr(
            QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes,
        )
        other = next(
            item for item in dialog.records if item.sample_id != identifier
        )
        dialog.accept_label()
        records = {item.sample_id: item for item in dataset.records()}
        assert records[identifier].label == "crossing"
        assert records[other.sample_id].label == other.label
        assert all(item.baseline_label == "invalid" for item in records.values()
                   if item.kind == "hand")
        assert len(records) == 8
    finally:
        dialog.close()


@pytest.mark.parametrize("corrupt", [False, True])
def test_unavailable_baseline_does_not_prevent_current_label_review(
    application, dataset, tmp_path, corrupt,
):
    source = next(item for item in dataset.records() if item.label == "crossing")
    images = dataset.images(source.sample_id)
    if not corrupt:
        images = {role: image for role, image in images.items() if role.startswith("current")}
    isolated = VisionDataset.create(tmp_path / "unavailable")
    identifier = isolated.save([VisionSample(
        "hand", "unknown", images, {}, "unavailable",
    )], 20_000_000)[0]
    if corrupt:
        with sqlite3.connect(isolated.path) as connection:
            connection.execute(
                "UPDATE assets SET png=? WHERE hash=(SELECT hash FROM sample_assets "
                "WHERE sample=? AND role='baseline')", (b"corrupt", identifier),
            )
    dialog = VisionReviewDialog(dataset=isolated)
    try:
        assert dialog.primary_label.isEnabled()
        assert dialog.accept_button.isEnabled()
        assert not dialog.baseline_label.isEnabled()
        assert "current image can" in dialog.status.text()
        assert "baseline" in dialog.before.text().lower()
        dialog.primary_label.setCurrentText("clear")
        dialog.accept_label()
        stored = isolated.records()[0]
        assert stored.label == "clear"
        assert stored.baseline_label == "unknown"
        assert stored.training_targets["current_obstruction"]
        assert not stored.training_targets["baseline_validity"]
    finally:
        dialog.close()


def test_review_save_failure_keeps_draft_and_current_sample(
    application, dataset, monkeypatch,
):
    dialog = VisionReviewDialog(dataset=dataset)
    try:
        dialog.kind.setCurrentIndex(1)
        identifier = dialog.current().sample_id
        dialog.primary_label.setCurrentText("occupied")
        dialog.piece_colour.setCurrentText("white")

        def fail(*args, **kwargs):
            raise OSError("Drive disconnected")

        monkeypatch.setattr(dataset, "review", fail)
        dialog.accept_label()
        assert dialog.current().sample_id == identifier
        assert dialog.primary_label.currentText() == "occupied"
        assert dialog.piece_colour.currentText() == "white"
        assert "Drive disconnected" in dialog.status.text()
    finally:
        dialog.close()


@pytest.mark.parametrize("missing_edge", [False, True])
def test_hand_coverage_warning_ignores_masked_centre(
    application, dataset, tmp_path, missing_edge,
):
    hand = next(item for item in dataset.records() if item.kind == "hand")
    images = dataset.images(hand.sample_id)
    for role in ("baseline_valid", "current_valid"):
        mask = images[role]
        height, width = mask.shape
        mask[:] = 255
        mask[height // 4:3 * height // 4, width // 4:3 * width // 4] = 0
    if missing_edge:
        images["current_valid"][0, 0] = 0
    isolated = VisionDataset.create(tmp_path / "coverage")
    isolated.save([VisionSample("hand", "clear", images, {}, "coverage")], 20_000_000)
    dialog = VisionReviewDialog(dataset=isolated)
    try:
        assert bool(dialog.coverage_warning.text()) == missing_edge
    finally:
        dialog.close()


def test_review_remembers_dataset_across_instances_and_honours_explicit_dataset(
    application, dataset, tmp_path, monkeypatch,
):
    directories = []

    def choose(_parent, _title, directory):
        directories.append(directory)
        return str(dataset.path.parent)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose)
    first = VisionReviewDialog()
    try:
        first.open_dataset()
        assert first.dataset.path == dataset.path
        first.open_dataset()
        assert directories == ["", str(dataset.path.parent)]
    finally:
        first.close()
    restored = VisionReviewDialog()
    explicit = VisionDataset.create(tmp_path / "explicit")
    supplied = VisionReviewDialog(dataset=explicit)
    try:
        assert restored.dataset.path == dataset.path
        assert len(restored.records) == 8
        assert supplied.dataset.path == explicit.path
        assert supplied.records == []
    finally:
        restored.close()
        supplied.close()


def test_cancelled_and_invalid_dataset_selections_preserve_remembered_directory(
    application, dataset, tmp_path, monkeypatch, vision_settings,
):
    key = "vision/lastDatasetDirectory"
    vision_settings.setValue(key, str(dataset.path.parent))
    dialog = VisionReviewDialog()
    try:
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *args: "")
        dialog.open_dataset()
        assert vision_settings.value(key) == str(dataset.path.parent)
        invalid = tmp_path / "missing"
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *args: str(invalid),
        )
        dialog.open_dataset()
        assert "Could not open dataset" in dialog.status.text()
        assert vision_settings.value(key) == str(dataset.path.parent)
        assert dialog.dataset.path == dataset.path
        assert not invalid.exists()
    finally:
        dialog.close()


def test_unavailable_remembered_dataset_is_reported_without_recreation(
    application, dataset, vision_settings,
):
    key = "vision/lastDatasetDirectory"
    vision_settings.setValue(key, str(dataset.path.parent))
    dataset.path.unlink()
    dialog = VisionReviewDialog()
    try:
        assert dialog.dataset is None
        assert dialog.records == []
        assert "Could not reopen dataset" in dialog.status.text()
        assert str(dataset.path.parent) in dialog.status.text()
        assert not dataset.path.exists()
        assert vision_settings.value(key) == str(dataset.path.parent)
    finally:
        dialog.close()


def test_collection_and_review_share_last_directory_without_starting_collection(
    application, dataset, tmp_path, monkeypatch, vision_settings,
):
    key = "vision/lastDatasetDirectory"
    vision_settings.setValue(key, str(dataset.path.parent))
    selected = VisionDataset.create(tmp_path / "chosen")
    directories = []

    def choose(_parent, _title, directory):
        directories.append(directory)
        return str(selected.path.parent)

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", choose)
    window = QMainWindow()
    tools = install_vision_tools(window)
    dialog = VisionDataDialog(tools)
    try:
        dialog.select_dataset()
        assert directories == [str(dataset.path.parent)]
        assert tools.dataset.path == selected.path
        assert vision_settings.value(key) == str(selected.path.parent)
        tools.show_review()
        assert tools.reviewer.dataset.path == selected.path
        assert tools.collector is None
        assert not tools.collecting
        assert not tools.timer.isActive()
    finally:
        dialog.preview_timer.stop()
        dialog.close()
        if tools.reviewer is not None:
            tools.reviewer.close()
        window.close()


def test_dataset_location_write_failure_is_visible(application, dataset, monkeypatch):
    dialog = VisionReviewDialog()
    try:
        monkeypatch.setattr(
            QFileDialog, "getExistingDirectory", lambda *args: str(dataset.path.parent),
        )
        monkeypatch.setattr(
            dialog.settings, "status", lambda: QSettings.Status.AccessError,
        )
        dialog.open_dataset()
        assert dialog.dataset.path == dataset.path
        assert "location was not saved" in dialog.status.text()
    finally:
        dialog.close()
