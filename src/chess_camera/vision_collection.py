"""Passive collection: no detector or game-state decisions are made here."""

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Full, Queue, SimpleQueue
from threading import Event, Lock, Thread
from uuid import uuid4

import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray
from .chess_position import Position, square_name
from .chessboard_widget import rotate_square
from .vision_data import (
    VisionDataset, VisionSample, coverage, demonstration_samples,
    normalize_view, ring_coverage, utc_now,
)

COLLECTION_ERRORS = (OSError, ValueError, sqlite3.Error, cv2.error)


@dataclass(frozen=True)
class Capture:
    frame: ImageArray
    grid: FloatArray
    position: Position
    previous: Position
    turns: int
    game_key: str
    move_id: str
    finished: bool = False


@dataclass(frozen=True)
class HandCapture:
    frame: ImageArray
    grid: FloatArray
    crossing: bool
    baseline: bool = False
    uncertain: bool = False


@dataclass(frozen=True)
class RecordedDemo:
    frame: ImageArray
    grid: FloatArray


@dataclass(frozen=True)
class EndGame:
    pass


def occupancy_samples(
    capture: Capture, session: str, sequence: int, *,
    phase: int = 1, setup_id: str = "setup-01",
) -> list[VisionSample]:
    if capture.turns not in range(4):
        raise ValueError("Invalid board orientation.")
    view = normalize_view(capture.frame, capture.grid)
    changed = {
        square for square in capture.position.keys() | capture.previous.keys()
        if capture.position.get(square) != capture.previous.get(square)
    }
    selected = set(changed)
    squares = [(r, c) for r in range(8) for c in range(8)]
    full_snapshot = sequence % 50 == 0
    if full_snapshot:
        selected.update(squares)
    offset = sequence * 7 % 64
    ordered = squares[offset:] + squares[:offset]
    for occupied in (False, True):
        candidates = [
            square for square in ordered
            if square not in changed
            and (square in capture.position) == occupied
        ]
        selected.update(candidates[:2])
    samples: list[VisionSample] = []
    for row, column in sorted(selected):
        physical = rotate_square(row, column, capture.turns)
        crop, valid = view.square(*physical)
        if coverage(valid) < 0.99:
            raise ValueError("Occupancy crop has less than 99% camera coverage.")
        piece = capture.position.get((row, column))
        samples.append(VisionSample(
            "occupancy", "occupied" if piece else "empty",
            {"crop": crop, "valid": valid},
            {
                "schema_version": 1, "label_source": "implicit-player-acceptance",
                "label_strength": "weak", "game": capture.game_key,
                "move": capture.move_id, "position_sequence": sequence,
                "burst": capture.move_id, "captured_at": utc_now(),
                "physical_square": list(physical),
                "algebraic_square": square_name((row, column)),
                "quarter_turns": capture.turns,
                "physical_edge_flags": [
                    edge for edge, condition in (
                        ("top", physical[0] == 0), ("bottom", physical[0] == 7),
                        ("left", physical[1] == 0), ("right", physical[1] == 7),
                    ) if condition
                ],
                "parity": (physical[0] + physical[1]) % 2,
                "square_colour": "light" if (row + column) % 2 == 0 else "dark",
                "piece_colour": "none" if piece is None else (
                    "white" if piece.isupper() else "black"
                ),
                "piece_colour_source": "logical-position",
                "valid_fraction": coverage(valid),
                "transform": view.transform.tolist(),
                "camera_grid": capture.grid.tolist(),
                "pixels_per_square": view.pixels_per_square,
                "projector_state": "black",
                "capture_settings": "not-available",
                "collection_phase": phase, "setup_id": setup_id,
                "full_position_snapshot": full_snapshot,
            },
            session,
        ))
    return samples


class VisionCollector:
    def __init__(
        self, dataset: VisionDataset, quota_bytes: int, control_root: Path,
        *, queue_size: int = 4, phase: int = 1, setup_id: str = "setup-01",
    ) -> None:
        if quota_bytes <= 0:
            raise ValueError("Quota must be positive.")
        if phase not in range(1, 5) or re.fullmatch(r"[A-Za-z0-9._-]{1,64}", setup_id) is None:
            raise ValueError("Choose phase 1-4 and an anonymous setup alias (letters/digits/._-).")
        self.dataset = dataset
        self.phase, self.setup_id = phase, setup_id
        self.quota_bytes = quota_bytes
        self.control_root = control_root
        self.session = uuid4().hex
        self.jobs: Queue[tuple[int, str, Capture | HandCapture | RecordedDemo | EndGame]] = Queue(
            maxsize=queue_size
        )
        self.generation = 0
        self.controls: SimpleQueue[tuple[str, str]] = SimpleQueue()
        self.messages: SimpleQueue[str] = SimpleQueue()
        self.stopped = Event()
        self.failed = Event()
        self.closed = Event()
        self.lifecycle = Lock()
        self.journal_lock = Lock()
        self.thread = Thread(target=self._run, name="vision-data-writer", daemon=True)
        self.thread.start()

    @property
    def active(self) -> bool:
        return not (self.stopped.is_set() or self.failed.is_set())

    @property
    def journal(self) -> Path:
        return self.control_root / f"{self.dataset.identity}-deletions.json"

    def _submit(
        self, kind: str, value: Capture | HandCapture | RecordedDemo | EndGame,
    ) -> None:
        if not self.active:
            return
        try:
            self.jobs.put_nowait((self.generation, kind, value))
        except Full:
            self.failed.set()
            self.stopped.set()
            self.messages.put(
                f"Collection paused: writer queue full; {self.jobs.qsize() + 1} "
                "queued/new captures and any unconfirmed position are discarded."
            )

    def accepted(self, capture: Capture) -> None:
        self._submit("accepted", capture)

    def hand(self, capture: HandCapture) -> None:
        self._submit("hand", capture)

    def recorded_demo(self, frame: ImageArray, grid: FloatArray) -> None:
        self._submit("demo", RecordedDemo(frame.copy(), grid.copy()))

    def end_game(self) -> None:
        self._submit("finish", EndGame())

    def invalidate(self, reason: str) -> None:
        try:
            self._record_intent(self.session, reason)
        except COLLECTION_ERRORS as error:
            self.failed.set()
            self.messages.put(f"Collection paused: could not record deletion intent: {error}")
        with self.lifecycle:
            self.generation += 1
            self.controls.put(("invalidate", reason))
            if self.closed.is_set():
                self.closed.clear()
                self.thread = Thread(
                    target=self._run, name="vision-data-discard", daemon=True,
                )
                self.thread.start()

    def close(self, *, wait: bool = False) -> None:
        self.stopped.set()
        if wait:
            self.thread.join(timeout=10)
            if self.thread.is_alive():
                raise TimeoutError("Vision writer has not finished closing.")

    def _read_intents(self) -> dict[str, str]:
        if not self.journal.exists():
            return {}
        value = json.loads(self.journal.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or not all(
            isinstance(key, str) and isinstance(reason, str)
            for key, reason in value.items()
        ):
            raise ValueError("Invalid pending-deletion journal.")
        return value

    def _write_intents(self, value: dict[str, str]) -> None:
        self.control_root.mkdir(parents=True, exist_ok=True)
        temporary = self.journal.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(value), encoding="utf-8")
            temporary.replace(self.journal)
        finally:
            temporary.unlink(missing_ok=True)

    def _record_intent(self, session: str, reason: str) -> None:
        with self.journal_lock:
            value = self._read_intents()
            value[session] = reason
            self._write_intents(value)

    def _apply_deletions(self) -> None:
        with self.journal_lock:
            pending = self._read_intents()
        for session, reason in pending.items():
            self.dataset.discard(session_id=session, reason=reason)
        if pending:
            with self.journal_lock:
                remaining = self._read_intents()
                for session, reason in pending.items():
                    if remaining.get(session) == reason:
                        del remaining[session]
                if remaining:
                    self._write_intents(remaining)
                else:
                    self.journal.unlink(missing_ok=True)
            self.messages.put("Pending dataset deletions completed.")

    def _discard_session(self, reason: str) -> None:
        self._record_intent(self.session, reason)
        self._apply_deletions()
        self.messages.put(f"Discarded affected collection session: {reason}")
        self.session = uuid4().hex

    def _run(self) -> None:
        pending: tuple[Capture, list[VisionSample]] | None = None
        baseline: tuple[ImageArray, FloatArray, str, str] | None = None
        last_hand: bool | None = None
        sequence = 0
        try:
            self._apply_deletions()
            while True:
                while not self.controls.empty():
                    _, reason = self.controls.get_nowait()
                    pending, baseline = None, None
                    last_hand = None
                    self._discard_session(reason)
                if self.stopped.is_set():
                    break
                try:
                    generation, kind, value = self.jobs.get(timeout=0.1)
                except Empty:
                    continue
                try:
                    if self.failed.is_set() or generation != self.generation:
                        pending = None
                        continue
                    if kind == "accepted" and isinstance(value, Capture):
                        sequence += 1
                        if pending is not None:
                            previous, samples = pending
                            continuous = (
                                previous.position == value.previous
                                and previous.game_key == value.game_key
                                and previous.turns == value.turns
                                and np.array_equal(previous.grid, value.grid)
                            )
                            if continuous:
                                samples = [
                                    VisionSample(
                                        item.kind, item.label, item.images,
                                        {**item.metadata, "confirmed_by": value.move_id,
                                         "confirmed_at": utc_now()},
                                        item.session_id, item.demo,
                                    ) for item in samples
                                ]
                                self.dataset.save(samples, self.quota_bytes)
                                self.messages.put(
                                    f"Stored {len(samples)} implicitly confirmed occupancy samples. "
                                    + self.dataset.summary()
                                )
                            else:
                                self._discard_session("Position history or geometry changed.")
                                baseline = None
                        pending = (
                            None if value.finished else
                            (value, occupancy_samples(
                                value, self.session, sequence,
                                phase=self.phase, setup_id=self.setup_id,
                            ))
                        )
                    elif kind == "hand" and isinstance(value, HandCapture):
                        captured_at = utc_now()
                        proposed = (
                            value.frame, value.grid, captured_at,
                            "unknown" if value.uncertain else "valid",
                        )
                        if value.baseline and (baseline is None or last_hand is not True):
                            baseline = proposed
                        if baseline is not None and not np.array_equal(
                            baseline[1], value.grid,
                        ):
                            baseline = None
                            last_hand = None
                        if last_hand == value.crossing:
                            if value.baseline:
                                baseline = proposed
                            continue
                        current = normalize_view(value.frame, value.grid)
                        edges = ring_coverage(current)
                        if min(edges.values()) < 0.99:
                            raise ValueError("Hand ring has less than 99% edge coverage.")
                        image, mask = current.ring()
                        images = {"current": image, "current_valid": mask}
                        metadata: dict[str, object] = {
                            "label_source": "deterministic-black-board-gate",
                            "label_strength": "weak", "captured_at": captured_at,
                            "ring_coverage": edges,
                            "transform": current.transform.tolist(),
                            "camera_grid": value.grid.tolist(),
                            "pixels_per_square": current.pixels_per_square,
                            "projector_state": "black",
                            "stillness_release": value.uncertain,
                            "collection_phase": self.phase, "setup_id": self.setup_id,
                            "baseline_compatible": False,
                        }
                        baseline_label = "unknown"
                        if baseline is not None:
                            frame, grid, baseline_at, baseline_label = baseline
                            before = normalize_view(frame, grid)
                            if min(ring_coverage(before).values()) >= 0.99:
                                old_image, old_mask = before.ring()
                                images.update({
                                    "baseline": old_image, "baseline_valid": old_mask,
                                })
                                metadata.update({
                                    "baseline_compatible": True,
                                    "baseline_captured_at": baseline_at,
                                    "baseline_label_source": "deterministic-baseline-proposal",
                                    "baseline_label_strength": "weak",
                                })
                            else:
                                baseline = None
                                baseline_label = "unknown"
                                self.messages.put(
                                    "Baseline excluded: insufficient edge coverage. "
                                    "Retaining the current-only hand observation."
                                )
                        self.dataset.save([VisionSample(
                            "hand", "unknown" if value.uncertain else (
                                "crossing" if value.crossing else "clear"
                            ),
                            images, metadata, self.session, baseline_label=baseline_label,
                        )], self.quota_bytes)
                        last_hand = value.crossing
                        self.messages.put("Stored hand-gate transition for review.")
                        if value.baseline:
                            baseline = proposed
                    elif kind == "demo" and isinstance(value, RecordedDemo):
                        _, samples = demonstration_samples(value.frame, value.grid)
                        self.dataset.save(samples, self.quota_bytes)
                        self.messages.put(
                            f"Stored {len(samples)} demonstration-only samples. "
                            + self.dataset.summary()
                        )
                        self.stopped.set()
                    elif kind == "finish" and isinstance(value, EndGame):
                        pending, baseline = None, None
                        last_hand = None
                    else:
                        raise ValueError("Invalid vision collection job.")
                finally:
                    self.jobs.task_done()
        except COLLECTION_ERRORS as error:
            self.failed.set()
            self.stopped.set()
            self.messages.put(
                f"Collection paused: {error}. Pending samples discarded; "
                "any recorded deletion intent must complete before training."
            )
        finally:
            pending, baseline = None, None
            while True:
                try:
                    self.jobs.get_nowait()
                    self.jobs.task_done()
                except Empty:
                    break
            with self.lifecycle:
                if not self.failed.is_set() and not self.controls.empty():
                    self.thread = Thread(
                        target=self._run, name="vision-data-discard", daemon=True,
                    )
                    self.thread.start()
                else:
                    self.closed.set()
