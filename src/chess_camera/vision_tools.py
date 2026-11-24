"""Modeless vision tools and optional, one-way gameplay notifications."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING
from uuid import uuid4

import cv2
import numpy as np
from PySide6.QtCore import QObject, QSettings, QStandardPaths, QTimer, Qt
from PySide6.QtGui import QImage, QKeySequence, QPixmap, QResizeEvent, QShortcut
from PySide6.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QLineEdit, QPlainTextEdit,
    QPushButton, QSizePolicy, QSpinBox, QVBoxLayout, QWidget,
)

from .board_detector import FloatArray, ImageArray
from .chess_position import Position
from .vision_collection import (
    COLLECTION_ERRORS, Capture, HandCapture, VisionCollector,
)
from .vision_data import (
    BASELINE_LABELS, LABELS, PIECE_COLOURS, StoredSample, VisionDataset,
    normalize_view, ring_coverage,
)

if TYPE_CHECKING:
    from .camera_app import CameraWindow


_LAST_DATASET_DIRECTORY = "vision/lastDatasetDirectory"


def remember_dataset(settings: QSettings, dataset: VisionDataset) -> None:
    settings.setValue(_LAST_DATASET_DIRECTORY, str(dataset.path.parent))
    settings.sync()
    if settings.status() != QSettings.Status.NoError:
        raise OSError("Could not save the dataset location in application settings.")


def image_pixmap(image: ImageArray) -> QPixmap:
    if image.ndim == 2:
        image = np.asarray(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
    rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(
        QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format.Format_RGB888,
        ).copy()
    )


def show_image(label: QLabel, image: ImageArray) -> None:
    label.setPixmap(image_pixmap(image).scaled(
        600, 500, Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    ))


class ReviewImage(QLabel):
    def __init__(self) -> None:
        super().__init__()
        self.original = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 120)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_image(self, image: ImageArray) -> None:
        self.original = image_pixmap(image)
        self.scale_image()

    def clear(self) -> None:
        self.original = QPixmap()
        super().clear()

    def scale_image(self) -> None:
        if not self.original.isNull():
            self.setPixmap(self.original.scaled(
                self.contentsRect().size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.scale_image()


class VisionReviewDialog(QDialog):
    def __init__(
        self, parent: QWidget | None = None, dataset: VisionDataset | None = None,
        *, settings: QSettings | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review vision labels")
        self.setMinimumSize(960, 640)
        self.resize(1050, 750)
        self.dataset = dataset
        self.settings = (
            settings if settings is not None
            else QSettings("ChessProject", "ChessboardCamera")
        )
        self.records: list[StoredSample] = []
        self.index = 0
        self.shortcuts: list[QShortcut] = []
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        open_button = QPushButton("Open dataset...")
        open_button.clicked.connect(self.open_dataset)
        toolbar.addWidget(open_button)
        self.kind = QComboBox()
        self.kind.addItems(["All examples", "Occupancy", "Hand"])
        self.kind.currentIndexChanged.connect(self.reload)
        toolbar.addWidget(self.kind)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        toolbar.addWidget(refresh)
        export = QPushButton("Export manifest...")
        export.clicked.connect(self.export)
        toolbar.addWidget(export)
        details = QPushButton("Technical details...")
        details.clicked.connect(self.show_details)
        toolbar.addWidget(details)
        layout.addLayout(toolbar)
        self.description = QLabel("Open a dataset; no camera or projector is required.")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.observation_group = QGroupBox("Observation")
        images = QHBoxLayout(self.observation_group)
        self.before_panel = QWidget()
        before_layout = QVBoxLayout(self.before_panel)
        self.before_caption = QLabel("Linked baseline")
        self.before = ReviewImage()
        before_layout.addWidget(self.before_caption)
        before_layout.addWidget(self.before, 1)
        self.after_panel = QWidget()
        after_layout = QVBoxLayout(self.after_panel)
        self.after_caption = QLabel("Current observation")
        self.after = ReviewImage()
        after_layout.addWidget(self.after_caption)
        after_layout.addWidget(self.after, 1)
        images.addWidget(self.before_panel, 1)
        images.addWidget(self.after_panel, 1)
        layout.addWidget(self.observation_group, 1)
        fields = QHBoxLayout()
        self.input_group = QGroupBox("Model inputs (read-only)")
        self.output_group = QGroupBox("Model outputs (training labels)")
        self.input_form = QFormLayout(self.input_group)
        self.output_form = QFormLayout(self.output_group)
        fields.addWidget(self.input_group, 1)
        fields.addWidget(self.output_group, 1)
        layout.addLayout(fields)
        self.square_colour = QLabel()
        self.location = QLabel()
        for title, value in (
            ("Square colour:", self.square_colour), ("Location:", self.location),
        ):
            value.setWordWrap(True)
            self.input_form.addRow(title, value)
        self.primary_label_title = QLabel("Occupancy:")
        self.primary_label = QComboBox()
        self.output_form.addRow(self.primary_label_title, self.primary_label)
        self.piece_colour = QComboBox()
        self.piece_colour.addItems(PIECE_COLOURS)
        self.output_form.addRow("Piece colour:", self.piece_colour)
        self.baseline_label = QComboBox()
        self.baseline_label.addItems(BASELINE_LABELS)
        self.output_form.addRow("Baseline validity:", self.baseline_label)
        self.coverage_warning = QLabel()
        self.coverage_warning.setWordWrap(True)
        layout.addWidget(self.coverage_warning)
        buttons = QHBoxLayout()
        self.previous_button = QPushButton("Previous")
        self.previous_button.clicked.connect(lambda: self.navigate(-1))
        buttons.addWidget(self.previous_button)
        self.accept_button = QPushButton("Save and next (Enter)")
        self.accept_button.clicked.connect(self.accept_label)
        buttons.addWidget(self.accept_button)
        self.next_button = QPushButton("Next (Right)")
        self.next_button.setToolTip("Next observation without saving changes.")
        self.next_button.clicked.connect(lambda: self.navigate(1))
        buttons.addWidget(self.next_button)
        self.discard_button = QPushButton("Discard observation")
        self.discard_button.setToolTip("Delete this observation (Delete).")
        self.discard_button.clicked.connect(self.discard)
        buttons.addWidget(self.discard_button)
        discard_session = QPushButton("Discard session...")
        discard_session.clicked.connect(self.discard_session)
        buttons.addWidget(discard_session)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        for key, action in (
            ("Return", self.accept_label),
            ("Right", lambda: self.navigate(1)),
            ("Left", lambda: self.navigate(-1)),
            ("Delete", self.discard),
            ("1", lambda: self.set_label(0)),
            ("2", lambda: self.set_label(1)),
            ("3", lambda: self.set_label(2)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self.reload()
        if self.dataset is None:
            remembered = str(self.settings.value(_LAST_DATASET_DIRECTORY, ""))
            if remembered:
                try:
                    self.dataset = VisionDataset(Path(remembered))
                    self.reload()
                except COLLECTION_ERRORS as error:
                    self.status.setText(
                        f"Could not reopen dataset at {remembered}: {error}"
                    )

    def open_dataset(self) -> None:
        directory = (
            str(self.dataset.path.parent) if self.dataset is not None
            else str(self.settings.value(_LAST_DATASET_DIRECTORY, ""))
        )
        path = QFileDialog.getExistingDirectory(self, "Open vision dataset", directory)
        if not path:
            return
        try:
            self.dataset = VisionDataset(Path(path))
            self.index = 0
            self.reload()
        except COLLECTION_ERRORS as error:
            self.status.setText(f"Could not open dataset: {error}")
            return
        try:
            remember_dataset(self.settings, self.dataset)
        except OSError as error:
            self.status.setText(f"Dataset opened, but its location was not saved: {error}")

    def reload(self) -> None:
        try:
            self.records = [] if self.dataset is None else self.dataset.records()
            selected = ("", "occupancy", "hand")[self.kind.currentIndex()]
            if selected:
                self.records = [item for item in self.records if item.kind == selected]
            self.index = min(self.index, max(0, len(self.records) - 1))
            self.display()
        except COLLECTION_ERRORS as error:
            self.records = []
            self.display()
            self.status.setText(f"Could not load examples: {error}")

    def display(self) -> None:
        for label in (self.before, self.after):
            label.clear()
        item = self.current()
        self.accept_button.setEnabled(item is not None)
        self.accept_button.setText(
            "Save (Enter)" if item is not None and self.index == len(self.records) - 1
            else "Save and next (Enter)"
        )
        self.previous_button.setEnabled(item is not None and self.index > 0)
        self.next_button.setEnabled(item is not None and self.index < len(self.records) - 1)
        self.discard_button.setEnabled(item is not None)
        self.primary_label.setEnabled(item is not None)
        square_example = item is not None and item.kind == "occupancy"
        hand_example = item is not None and item.kind == "hand"
        self.before_panel.setVisible(hand_example)
        self.input_group.setVisible(square_example)
        self.piece_colour.setEnabled(square_example)
        self.input_form.setRowVisible(self.square_colour, square_example)
        self.input_form.setRowVisible(self.location, square_example)
        self.output_form.setRowVisible(self.piece_colour, square_example)
        self.output_form.setRowVisible(self.baseline_label, hand_example)
        self.baseline_label.setEnabled(False)
        self.baseline_label.setCurrentText(
            item.baseline_label if item is not None and hand_example else "unknown"
        )
        self.before_caption.setText("Linked baseline")
        self.after_caption.setText("Square image" if square_example else "Current observation")
        self.primary_label_title.setText(
            "Occupancy:" if square_example else "Current obstruction:"
        )
        self.primary_label.clear()
        self.piece_colour.setCurrentText("unknown")
        for value in (self.square_colour, self.location):
            value.setText("Not recorded")
        self.coverage_warning.clear()
        if item is None or self.dataset is None:
            self.observation_group.setTitle("Observation")
            self.description.setText("No examples. Open a dataset or generate demonstration examples.")
            self.status.clear()
            return
        prefix = "DEMO ONLY - excluded from training. " if item.demo else ""
        self.observation_group.setTitle(
            f"Observation {self.index + 1} of {len(self.records)} - {item.sample_id[:8]}"
        )
        self.description.setText(
            f"{prefix}{self.index + 1}/{len(self.records)} | {item.kind} | "
            f"label: {item.label} | original: {item.original_label} | "
            f"{'reviewed' if item.reviewed else 'unreviewed'}"
        )
        self.primary_label.addItems(LABELS[item.kind])
        self.primary_label.setCurrentText(item.label)
        try:
            if item.kind == "occupancy":
                colour = item.metadata.get("piece_colour", "unknown")
                if colour not in PIECE_COLOURS:
                    raise ValueError("Invalid recorded piece colour.")
                self.piece_colour.setCurrentText(str(colour))
                self.square_colour.setText(str(
                    item.metadata.get("square_colour", "Not recorded")
                ))
                square = item.metadata.get("physical_square")
                if (
                    isinstance(square, list) and len(square) == 2
                    and all(isinstance(v, int) and 0 <= v < 8 for v in square)
                ):
                    row, column = square
                    edges = [
                        edge for edge, condition in (
                            ("top", row == 0), ("bottom", row == 7),
                            ("left", column == 0), ("right", column == 7),
                        ) if condition
                    ]
                    self.location.setText(
                        ("Corner: " if len(edges) == 2 else "Edge: ")
                        + ", ".join(edges) if edges else "Interior"
                    )
            images = self.dataset.images(
                item.sample_id, roles=(
                    ("crop", "valid") if square_example else ("current", "current_valid")
                ),
            )
            baseline_problem = ""
            if hand_example:
                if item.baseline_id is None:
                    self.before.setText("No linked baseline was recorded.")
                    self.before_caption.setText("Linked baseline (not available)")
                    baseline_problem = (
                        "No baseline recorded. The current image can be labelled independently."
                    )
                else:
                    members = self.dataset.baseline_members(item.sample_id)
                    self.before_caption.setText(
                        f"Linked baseline {item.baseline_id[:8]} "
                        f"({len(members)} observations)"
                    )
                    try:
                        images.update(self.dataset.images(
                            item.sample_id, roles=("baseline", "baseline_valid"),
                        ))
                        self.baseline_label.setEnabled(True)
                    except COLLECTION_ERRORS as error:
                        self.before.setText("Linked baseline could not be read.")
                        baseline_problem = (
                            f"Baseline unavailable: {error}. "
                            "The current image can still be labelled."
                        )
            masks = [image for role, image in images.items()
                     if role in ("valid", "baseline_valid", "current_valid")]
            incomplete = False
            for mask in masks:
                if item.kind == "hand":
                    # The central 6x6 board is excluded, not missing coverage.
                    height, width = mask.shape
                    mask = mask.copy()
                    mask[height // 4:3 * height // 4,
                         width // 4:3 * width // 4] = 255
                incomplete |= bool(np.any(mask == 0))
            if incomplete:
                self.coverage_warning.setText(
                    "Warning: incomplete camera coverage; purple pixels are unavailable."
                )
            if item.kind == "occupancy":
                image = images["crop"].copy()
                valid = images["valid"]
                image[valid == 0] = (120, 0, 120)
                h, w = image.shape[:2]
                cv2.rectangle(image, (w // 4, h // 4), (3 * w // 4, 3 * h // 4),
                              (0, 220, 255), 2)
                self.after.set_image(image)
                self.after_caption.setText("Square image - target outlined in yellow")
            else:
                for label, role, mask in (
                    (self.before, "baseline", "baseline_valid"),
                    (self.after, "current", "current_valid"),
                ):
                    if role not in images:
                        continue
                    image = images[role].copy()
                    image[images[mask] == 0] = (60, 0, 60)
                    label.set_image(image)
            self.status.setText(
                (baseline_problem + "\n" if baseline_problem else "")
                + "Choose labels, then save. Next leaves labels unchanged. "
                "Shortcuts 1/2/3 save the primary label."
            )
        except (KeyError, *COLLECTION_ERRORS) as error:
            self.accept_button.setEnabled(False)
            self.primary_label.setEnabled(False)
            self.piece_colour.setEnabled(False)
            self.baseline_label.setEnabled(False)
            self.status.setText(f"Cannot review corrupt example: {error}")

    def show_details(self) -> None:
        item = self.current()
        if item is None:
            self.status.setText("Open a dataset with examples first.")
            return
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.setWindowTitle("Technical details - current example")
        dialog.resize(700, 500)
        layout = QVBoxLayout(dialog)
        metadata = QPlainTextEdit()
        metadata.setReadOnly(True)
        metadata.setPlainText(json.dumps(item.metadata, indent=2))
        layout.addWidget(metadata)
        dialog.show()

    def current(self) -> StoredSample | None:
        return self.records[self.index] if self.records else None

    def navigate(self, offset: int) -> None:
        if self.records:
            index = max(0, min(self.index + offset, len(self.records) - 1))
            if index != self.index:
                self.index = index
                self.display()

    def accept_label(self) -> None:
        self.save_label(self.primary_label.currentText())

    def set_label(self, index: int) -> None:
        item = self.current()
        if item is not None and self.accept_button.isEnabled():
            self.save_label(LABELS[item.kind][index])

    def save_label(self, label: str) -> None:
        item = self.current()
        if item is None or self.dataset is None or not self.accept_button.isEnabled():
            return
        try:
            baseline_label = (
                self.baseline_label.currentText()
                if item.kind == "hand" and self.baseline_label.isEnabled() else None
            )
            members = None
            if baseline_label is not None and (
                baseline_label != item.baseline_label
                or baseline_label != "unknown" and not item.baseline_reviewed
            ):
                members = self.dataset.baseline_members(item.sample_id)
                if len(members) > 1 and QMessageBox.question(
                    self, "Update shared baseline",
                    f"This baseline is linked to {len(members)} observations.\n"
                    f"Set its validity to '{baseline_label}' for all of them?\n"
                    "Other current-image labels will not change.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                ) != QMessageBox.StandardButton.Yes:
                    return
            self.dataset.review(
                item.sample_id, label,
                piece_colour=(
                    self.piece_colour.currentText() if item.kind == "occupancy" else None
                ),
                baseline_label=baseline_label, confirmed_baseline_members=members,
            )
            self.index += 1
            self.reload()
        except COLLECTION_ERRORS as error:
            self.status.setText(f"Could not save label: {error}")

    def discard(self) -> None:
        item = self.current()
        if item is None or self.dataset is None:
            return
        try:
            self.dataset.discard(sample_id=item.sample_id, reason="reviewer-discard")
            self.reload()
        except COLLECTION_ERRORS as error:
            self.status.setText(f"Could not discard example: {error}")

    def export(self) -> None:
        if self.dataset is None:
            self.status.setText("Open a dataset first.")
            return
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export vision manifest", filter="JSON lines (*.jsonl)"
        )
        if filename:
            try:
                self.dataset.export_manifest(Path(filename))
                self.status.setText("Manifest exported; image assets remain in the dataset.")
            except COLLECTION_ERRORS as error:
                self.status.setText(f"Could not export manifest: {error}")

    def discard_session(self) -> None:
        item = self.current()
        if item is None or self.dataset is None:
            return
        if QMessageBox.question(
            self, "Discard collection session",
            f"Delete all examples in session {item.session_id}? This cannot be undone.",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            self.dataset.discard(session_id=item.session_id, reason="user-session-discard")
            self.reload()
        except COLLECTION_ERRORS as error:
            self.status.setText(f"Could not discard session: {error}")


class VisionDataDialog(QDialog):
    def __init__(self, tools: "VisionTools") -> None:
        super().__init__(tools.window)
        self.tools = tools
        self.setWindowTitle("Vision data - collection is opt-in")
        self.resize(850, 850)
        self.recorded: tuple[ImageArray, FloatArray] | None = None
        layout = QVBoxLayout(self)
        self.status = QLabel("Collection OFF. Previewing never starts collection.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.dataset_label = QLabel("No dataset selected.")
        layout.addWidget(self.dataset_label)
        toolbar = QHBoxLayout()
        select = QPushButton("Select/create dataset...")
        select.clicked.connect(self.select_dataset)
        toolbar.addWidget(select)
        self.quota = QSpinBox()
        self.quota.setRange(1, 800)
        self.quota.setValue(20)
        self.quota.setSuffix(" GB quota")
        toolbar.addWidget(self.quota)
        self.start = QPushButton("Start collection")
        self.start.clicked.connect(self.start_collection)
        toolbar.addWidget(self.start)
        pause = QPushButton("Pause collection")
        pause.clicked.connect(self.tools.pause)
        toolbar.addWidget(pause)
        layout.addLayout(toolbar)
        setup = QHBoxLayout()
        self.phase = QComboBox()
        self.phase.addItems([
            "Phase 1: fixed setup", "Phase 2: placement",
            "Phase 3: lighting", "Phase 4: boards/pieces",
        ])
        setup.addWidget(self.phase)
        self.setup_alias = QLineEdit("setup-01")
        self.setup_alias.setToolTip(
            "Anonymous setup alias; change it when equipment/placement/lighting changes. No personal data."
        )
        setup.addWidget(self.setup_alias)
        layout.addLayout(setup)
        toolbar = QHBoxLayout()
        recorded = QPushButton("Load recorded image + grid...")
        recorded.clicked.connect(self.load_recorded)
        toolbar.addWidget(recorded)
        live = QPushButton("Use live view")
        live.clicked.connect(self.use_live)
        toolbar.addWidget(live)
        generate = QPushButton("Generate demo examples")
        generate.clicked.connect(self.generate_demo)
        toolbar.addWidget(generate)
        review = QPushButton("Review labels...")
        review.clicked.connect(self.tools.show_review)
        toolbar.addWidget(review)
        layout.addLayout(toolbar)
        preview_options = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems([
            "Full normalized view", "Hand-model ring", "Selected-square crop",
        ])
        self.view_mode.currentIndexChanged.connect(self.refresh_preview)
        preview_options.addWidget(self.view_mode)
        self.target = QComboBox()
        self.target.addItems([
            f"Physical row {row}, column {column}"
            for row in range(8) for column in range(8)
        ])
        self.target.currentIndexChanged.connect(self.refresh_preview)
        preview_options.addWidget(self.target)
        layout.addLayout(preview_options)
        self.preview = QLabel("No camera needed: load a recorded camera image and its matching grid JSON.")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.preview, 1)
        self.details = QLabel()
        self.details.setWordWrap(True)
        layout.addWidget(self.details)
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(1000)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_timer.start()

    def use_live(self) -> None:
        self.recorded = None
        self.refresh_preview()

    def select_dataset(self) -> None:
        if self.tools.collector is not None and not self.tools.collector.closed.is_set():
            self.status.setText("Pause collection and wait for pending writes before changing datasets.")
            return
        directory = (
            str(self.tools.dataset.path.parent) if self.tools.dataset is not None
            else str(self.tools.settings.value(_LAST_DATASET_DIRECTORY, ""))
        )
        root = QFileDialog.getExistingDirectory(
            self, "Select vision dataset directory", directory,
        )
        if not root:
            return
        try:
            self.tools.dataset = VisionDataset.create(Path(root))
            self.dataset_label.setText(str(self.tools.dataset.path))
            self.status.setText("Dataset selected. Collection remains OFF.")
        except COLLECTION_ERRORS as error:
            self.status.setText(f"Could not select dataset: {error}")
            return
        try:
            remember_dataset(self.tools.settings, self.tools.dataset)
        except OSError as error:
            self.status.setText(f"Dataset selected, but its location was not saved: {error}")

    def start_collection(self) -> None:
        self.tools.start(self.quota.value() * 1_000_000_000)

    def load_recorded(self) -> None:
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Recorded camera image", filter="Images (*.png *.jpg *.jpeg)"
        )
        if not image_path:
            return
        grid_path, _ = QFileDialog.getOpenFileName(
            self, "Matching calibration/grid JSON", filter="JSON (*.json)"
        )
        if not grid_path:
            return
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Could not read the recorded camera image.")
            image = np.asarray(image, dtype=np.uint8)
            data = json.loads(Path(grid_path).read_text(encoding="utf-8"))
            grid = np.asarray(data["board"]["camera_grid"], dtype=np.float32)
            normalize_view(image, grid)
            self.recorded = (image, grid)
            self.refresh_preview()
            self.status.setText(
                "Recorded preview only. Check that the grid matches this image; "
                "generated examples are demonstration-only."
            )
        except (KeyError, TypeError, *COLLECTION_ERRORS) as error:
            self.status.setText(f"Could not load recorded input: {error}")

    def refresh_preview(self) -> None:
        if not self.isVisible():
            return
        source = self.recorded or self.tools.live_source()
        if source is None:
            self.preview.setText("No verified live grid. Load recorded input to work without hardware.")
            return
        try:
            view = normalize_view(*source)
            image = view.image.copy()
            image[view.valid == 0] = (120, 0, 120)
            p = view.pixels_per_square
            cv2.rectangle(image, (2 * p, 2 * p), (10 * p, 10 * p), (0, 255, 0), 2)
            cv2.rectangle(image, (3 * p, 3 * p), (9 * p, 9 * p), (255, 160, 0), 2)
            for row in range(8):
                for column in range(8):
                    cv2.putText(
                        image, f"{row},{column}",
                        ((column + 2) * p + 4, (row + 2) * p + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1,
                    )
            if self.view_mode.currentIndex() == 1:
                image, valid = view.ring()
                image[valid == 0] = (60, 0, 60)
            elif self.view_mode.currentIndex() == 2:
                row, column = divmod(self.target.currentIndex(), 8)
                image, valid = view.square(row, column)
                image[valid == 0] = (120, 0, 120)
                cv2.rectangle(
                    image, (p // 2, p // 2), (3 * p // 2, 3 * p // 2),
                    (0, 220, 255), 2,
                )
            show_image(self.preview, image)
            self.details.setText(
                "Green: playing board. Blue: central 6x6 region excluded from hand input. "
                "Purple: unavailable pixels. Edge coverage: "
                + ", ".join(f"{key} {value:.0%}" for key, value in ring_coverage(view).items())
            )
        except COLLECTION_ERRORS as error:
            self.details.setText(f"Preview unavailable: {error}")

    def generate_demo(self) -> None:
        if self.tools.dataset is None:
            self.status.setText("Select a dataset first.")
            return
        if self.recorded is None:
            self.status.setText("Load a recorded camera image and its matching grid first.")
            return
        if self.tools.collector is not None and not self.tools.collector.closed.is_set():
            self.status.setText("Pause live collection and wait for it to finish before generating demos.")
            return
        started = self.tools.start_writer(self.quota.value() * 1_000_000_000)
        if started and self.tools.collector is not None:
            self.tools.collector.recorded_demo(*self.recorded)
            self.status.setText("Generating demonstration-only examples; live collection remains OFF.")


class VisionTools(QObject):
    def __init__(self, window: "CameraWindow") -> None:
        super().__init__(window)
        self.window = window
        self.settings = QSettings("ChessProject", "ChessboardCamera")
        self.dataset: VisionDataset | None = None
        self.collector: VisionCollector | None = None
        self.collectors: list[VisionCollector] = []
        self.collecting = False
        self.dialog: VisionDataDialog | None = None
        self.reviewer: VisionReviewDialog | None = None
        self.pending_submission: Capture | None = None
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self.poll)
        self.last_hand: tuple[bool, bool, bool] | None = None
        self.game_end_seen = False

    def message(self, text: str) -> None:
        if self.dialog is not None:
            self.dialog.status.setText(text)
        self.window.statusBar().showMessage(text)

    def show(self) -> None:
        if self.dialog is None:
            self.dialog = VisionDataDialog(self)
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self.dialog.refresh_preview()

    def show_review(self) -> None:
        if self.reviewer is None:
            self.reviewer = VisionReviewDialog(
                self.window, self.dataset, settings=self.settings,
            )
        elif self.dataset is not None:
            self.reviewer.dataset = self.dataset
            self.reviewer.reload()
        self.reviewer.show()
        self.reviewer.raise_()

    def live_source(self) -> tuple[ImageArray, FloatArray] | None:
        window = self.window
        if (
            window._latest_frame is None or window._calibration_result is None
            or not window._camera_geometry_ready or window._calibrator.active
        ):
            return None
        return window._latest_frame, window._calibration_result.mapping.camera_grid

    def start_writer(self, quota: int) -> bool:
        if self.dataset is None:
            self.message("Select a dataset first.")
            return False
        if self.collector is not None and not self.collector.closed.is_set():
            self.message("The previous writer is still running.")
            return False
        control = Path(QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppLocalDataLocation
        )) / "vision-control"
        phase = 1 if self.dialog is None else self.dialog.phase.currentIndex() + 1
        setup_id = "setup-01" if self.dialog is None else self.dialog.setup_alias.text().strip()
        try:
            self.collector = VisionCollector(
                self.dataset, quota, control, phase=phase, setup_id=setup_id,
            )
        except COLLECTION_ERRORS as error:
            self.message(f"Could not start collection: {error}")
            return False
        self.collectors.append(self.collector)
        self.timer.start()
        return True

    def start(self, quota: int) -> None:
        if self.live_source() is None:
            self.message("Live collection requires a verified camera grid; use recorded demos without hardware.")
            return
        if self.collector is not None and not self.collector.closed.is_set():
            self.message("Pause the current writer and wait before restarting.")
            return
        if not self.start_writer(quota):
            return
        if self.collector is not None and self.collector.active:
            self.collecting = True
            self.last_hand = None
            self.message("Collection ON. Gameplay decisions are unchanged.")

    def pause(self) -> None:
        self.collecting = False
        self.pending_submission = None
        if self.collector is not None:
            self.collector.close()
        self.message("Collection OFF. Unconfirmed positions and queued captures are discarded.")

    def poll(self) -> None:
        if self.collector is None:
            return
        for collector in self.collectors:
            while True:
                try:
                    self.message(collector.messages.get_nowait())
                except Empty:
                    break
        if self.collector.failed.is_set():
            self.collecting = False
            self.pending_submission = None
        if self.collector.closed.is_set():
            self.collecting = False
        if all(collector.closed.is_set() for collector in self.collectors):
            self.timer.stop()
        if self.collecting and self.live_source() is None:
            self.pause()
            self.message("Collection paused: camera or verified grid unavailable.")
        if self.collecting:
            remote = self.window._latest_lichess_state
            finished = (
                remote.result != "*" if self.window._lichess_game_controller is not None
                and remote is not None else self.window._legal_game.status.is_game_over
            )
            clock = self.window._offline_clock
            if (
                self.window._lichess_game_controller is None
                and self.window._offline_clock_enabled
            ):
                finished = finished or clock.snapshot().expired_player is not None
            if finished and not self.game_end_seen:
                self.collector.end_game()
                self.pending_submission = None
            self.game_end_seen = finished

    def snapshot(
        self, previous: Position, position: Position, *, finished: bool = False,
    ) -> Capture | None:
        source = self.live_source()
        if not self.collecting or self.collector is None or source is None:
            return None
        if self.window._alignment_monitor.output_mode != "black":
            self.invalidate("Accepted capture is not a black-board observation.")
            return None
        frame, grid = source
        controller = self.window._lichess_game_controller
        if controller is None and self.window._game_mode != "offline-game":
            self.invalidate("Diagnostic-mode observations are not legal gameplay labels.")
            return None
        game_key = (
            hashlib.sha256(controller.game_id.encode()).hexdigest()
            if controller is not None else f"offline-{id(self.window._legal_game)}"
        )
        return Capture(
            frame.copy(), grid.copy(), dict(position), dict(previous),
            self.window._board_quarter_turns, game_key, uuid4().hex, finished,
        )

    def invalidate(self, reason: str) -> None:
        self.pending_submission = None
        self.last_hand = None
        for collector in self.collectors:
            collector.invalidate(reason)
        if self.collectors:
            self.timer.start()


def install_vision_tools(window: "CameraWindow") -> VisionTools:
    tools = VisionTools(window)
    menu = window.menuBar().addMenu("&Tools")
    menu.addAction("Vision data...").triggered.connect(tools.show)
    menu.addAction("Review vision labels...").triggered.connect(tools.show_review)
    return tools


def vision_discontinuity(window: "CameraWindow", reason: str) -> None:
    tools = getattr(window, "_vision_tools", None)
    if tools is not None:
        tools.invalidate(reason)


def vision_submission(
    window: "CameraWindow", previous: Position, position: Position,
) -> None:
    tools = getattr(window, "_vision_tools", None)
    if tools is not None and tools.collecting:
        tools.pending_submission = tools.snapshot(previous, position)


def vision_move_accepted(
    window: "CameraWindow", previous: Position, position: Position,
    *, submission: bool = False, finished: bool = False,
) -> None:
    tools = getattr(window, "_vision_tools", None)
    if tools is None or not tools.collecting or tools.collector is None:
        return
    capture = (
        tools.pending_submission if submission
        else tools.snapshot(previous, position, finished=finished)
    )
    tools.pending_submission = None
    if capture is not None:
        tools.collector.accepted(replace(capture, finished=finished))
        tools.last_hand = None


def vision_hand_observation(
    window: "CameraWindow", crossing: bool, *, baseline: bool = False,
) -> None:
    tools = getattr(window, "_vision_tools", None)
    if tools is None or not tools.collecting or tools.collector is None:
        return
    source = tools.live_source()
    if source is None or window._alignment_monitor.output_mode != "black":
        return
    uncertain = window._board_edge_hand_gate.released
    state = (crossing, baseline, uncertain)
    # Keep one clean baseline and transition frames, not a full camera stream.
    if tools.last_hand == state and not baseline:
        return
    if baseline and tools.last_hand == state:
        return
    tools.last_hand = state
    frame, grid = source
    tools.collector.hand(HandCapture(
        frame.copy(), grid.copy(), crossing, baseline, uncertain,
    ))


def close_vision_tools(window: "CameraWindow") -> None:
    tools = getattr(window, "_vision_tools", None)
    if tools is not None:
        tools.pause()
        for collector in tools.collectors:
            collector.close()
