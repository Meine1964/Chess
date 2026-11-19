from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray
from .chess_position import Position, Square, square_name
from .occupancy import rectify_board

ReferenceTrackerState = Literal[
    "uninitialized",
    "watching",
    "activity",
    "waiting-for-clear",
    "settling",
    "candidate",
    "accepted",
    "ambiguous",
]


@dataclass(frozen=True)
class ReferenceTrackerUpdate:
    state: ReferenceTrackerState
    move: tuple[Square, Square] | None
    message: str
    change_scores: FloatArray | None = None


class ReferenceMoveTracker:
    def __init__(
        self,
        *,
        canonical_size: int = 800,
        settled_observations: int = 2,
        candidate_observations: int = 2,
        minimum_square_change: float = 0.08,
        minimum_dominance: float = 0.025,
    ) -> None:
        if canonical_size < 320 or canonical_size % 8 != 0:
            raise ValueError(
                "Canonical size must be a multiple of eight and at least 320."
            )
        if settled_observations < 1 or candidate_observations < 1:
            raise ValueError("Observation requirements must be positive.")
        if minimum_square_change <= 0 or minimum_dominance <= 0:
            raise ValueError("Change thresholds must be positive.")
        self._canonical_size = canonical_size
        self._settled_observations = settled_observations
        self._candidate_observations = candidate_observations
        self._minimum_square_change = minimum_square_change
        self._minimum_dominance = minimum_dominance
        self._reference: ImageArray | None = None
        self._previous_motion_image: np.ndarray | None = None
        self._settled_count = 0
        self._candidate_move: tuple[Square, Square] | None = None
        self._candidate_count = 0
        self._pending_reference: ImageArray | None = None

    @property
    def initialized(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> ImageArray | None:
        return None if self._reference is None else self._reference.copy()

    def clear(self) -> None:
        self._reference = None
        self.reset_activity()

    def reset_activity(self) -> None:
        self._previous_motion_image = None
        self._settled_count = 0
        self._candidate_move = None
        self._candidate_count = 0
        self._pending_reference = None

    def set_reference(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
    ) -> None:
        self._reference = _canonical_ui_board(
            frame,
            camera_grid,
            ui_quarter_turns,
            self._canonical_size,
        )
        self.reset_activity()

    def set_canonical_reference(self, reference: ImageArray) -> None:
        if (
            reference.ndim != 3
            or reference.shape
            != (self._canonical_size, self._canonical_size, 3)
        ):
            raise ValueError(
                "Canonical reference has an unexpected image shape."
            )
        self._reference = np.asarray(reference, dtype=np.uint8).copy()
        self.reset_activity()

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
        position: Position,
    ) -> ReferenceTrackerUpdate:
        if self._reference is None:
            return ReferenceTrackerUpdate(
                "uninitialized",
                None,
                "Reference move tracking is not initialized.",
            )
        current = _canonical_ui_board(
            frame,
            camera_grid,
            ui_quarter_turns,
            self._canonical_size,
        )
        registered = _register_small_translation(
            self._reference,
            current,
        )
        normalized = _normalize_candidate(self._reference, registered)
        motion_image = cv2.resize(
            cv2.cvtColor(normalized, cv2.COLOR_BGR2GRAY),
            (100, 100),
            interpolation=cv2.INTER_AREA,
        )
        motion_image = cv2.GaussianBlur(motion_image, (5, 5), 0)
        if self._previous_motion_image is not None:
            difference = cv2.absdiff(
                motion_image,
                self._previous_motion_image,
            )
            if (
                float(np.mean(difference)) > 2.5
                and float(np.mean(difference > 12)) > 0.015
            ):
                self._previous_motion_image = motion_image
                self._clear_pending()
                return ReferenceTrackerUpdate(
                    "activity",
                    None,
                    "Board activity detected; waiting for the hand to leave.",
                )
        self._previous_motion_image = motion_image

        if _has_large_obstruction(self._reference, normalized):
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "waiting-for-clear",
                None,
                "A hand or object still obscures multiple board squares.",
            )

        self._settled_count += 1
        if self._settled_count < self._settled_observations:
            return ReferenceTrackerUpdate(
                "settling",
                None,
                "Board is clear; confirming a settled image.",
            )

        scores = square_change_scores(self._reference, normalized)
        if float(np.max(scores)) < self._minimum_square_change:
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "watching",
                None,
                "Board matches the accepted reference.",
                scores,
            )
        move = self._select_move(scores, position)
        if move is None:
            self._candidate_move = None
            self._candidate_count = 0
            return ReferenceTrackerUpdate(
                "ambiguous",
                None,
                _ambiguous_change_message(scores),
                scores,
            )
        if move == self._candidate_move:
            self._candidate_count += 1
        else:
            self._candidate_move = move
            self._candidate_count = 1
        if self._candidate_count < self._candidate_observations:
            return ReferenceTrackerUpdate(
                "candidate",
                None,
                f"Candidate {square_name(move[0])}-{square_name(move[1])} "
                f"({self._candidate_count}/{self._candidate_observations}).",
                scores,
            )

        self._pending_reference = normalized.copy()
        return ReferenceTrackerUpdate(
            "accepted",
            move,
            f"Accepted {square_name(move[0])}-{square_name(move[1])}.",
            scores,
        )

    def commit_accepted(
        self,
        move: tuple[Square, Square],
    ) -> None:
        if move != self._candidate_move or self._pending_reference is None:
            raise ValueError("No matching accepted move is ready to commit.")
        accepted_reference = self._pending_reference
        self.reset_activity()
        self._reference = accepted_reference

    def _select_move(
        self,
        scores: FloatArray,
        position: Position,
    ) -> tuple[Square, Square] | None:
        occupied = set(position)
        if not occupied:
            return None
        empty = {
            (row, column)
            for row in range(8)
            for column in range(8)
            if (row, column) not in occupied
        }
        sources = sorted(
            ((float(scores[square]), square) for square in occupied),
            reverse=True,
        )
        destinations = sorted(
            ((float(scores[square]), square) for square in empty),
            reverse=True,
        )
        source_score, source = sources[0]
        destination_score, destination = destinations[0]
        if (
            source_score < self._minimum_square_change
            or destination_score < self._minimum_square_change
        ):
            return None
        if (
            len(sources) > 1
            and source_score - sources[1][0] < self._minimum_dominance
        ):
            return None
        if (
            len(destinations) > 1
            and destination_score - destinations[1][0]
            < self._minimum_dominance
        ):
            return None
        remaining = scores.copy()
        remaining[source] = 0
        remaining[destination] = 0
        if float(np.max(remaining)) > max(
            self._minimum_square_change,
            min(source_score, destination_score) * 0.75,
        ):
            return None
        return source, destination

    def _clear_pending(self) -> None:
        self._settled_count = 0
        self._candidate_move = None
        self._candidate_count = 0


def _canonical_ui_board(
    frame: ImageArray,
    camera_grid: FloatArray,
    ui_quarter_turns: int,
    size: int,
) -> ImageArray:
    canonical = rectify_board(frame, camera_grid, size)
    turns = (-ui_quarter_turns) % 4
    if turns == 1:
        canonical = cv2.rotate(canonical, cv2.ROTATE_90_CLOCKWISE)
    elif turns == 2:
        canonical = cv2.rotate(canonical, cv2.ROTATE_180)
    elif turns == 3:
        canonical = cv2.rotate(canonical, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return np.asarray(canonical, dtype=np.uint8)


def _normalize_candidate(
    reference: ImageArray,
    candidate: ImageArray,
) -> ImageArray:
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB).astype(
        np.float32
    )
    offset = float(
        np.median(
            reference_lab[:, :, 0].astype(np.float32)
            - candidate_lab[:, :, 0]
        )
    )
    candidate_lab[:, :, 0] = np.clip(
        candidate_lab[:, :, 0] + offset,
        0,
        255,
    )
    return cv2.cvtColor(
        candidate_lab.astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )


def _register_small_translation(
    reference: ImageArray,
    candidate: ImageArray,
) -> ImageArray:
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    candidate_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    (shift_x, shift_y), response = cv2.phaseCorrelate(
        reference_gray,
        candidate_gray,
    )
    if response < 0.2 or max(abs(shift_x), abs(shift_y)) > 6:
        return candidate
    transform = np.float32(
        [
            [1, 0, -shift_x],
            [0, 1, -shift_y],
        ]
    )
    return cv2.warpAffine(
        candidate,
        transform,
        (candidate.shape[1], candidate.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


def _has_large_obstruction(
    reference: ImageArray,
    candidate: ImageArray,
) -> bool:
    difference = cv2.absdiff(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )
    mask = np.asarray(difference > 24, dtype=np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((7, 7), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 9), dtype=np.uint8),
    )
    component_count, _labels, statistics, _centroids = (
        cv2.connectedComponentsWithStats(mask)
    )
    square_size = reference.shape[0] / 8
    for component in range(1, component_count):
        x, y, width, height, area = statistics[component]
        columns = int(np.ceil((x + width) / square_size) - x // square_size)
        rows = int(np.ceil((y + height) / square_size) - y // square_size)
        if area >= reference.shape[0] * reference.shape[1] * 0.0125 and (
            rows * columns >= 3
        ):
            return True
    return False


def square_change_scores(
    reference: ImageArray,
    candidate: ImageArray,
) -> FloatArray:
    if reference.shape != candidate.shape:
        raise ValueError("Reference and candidate images must have equal shape.")
    if (
        reference.ndim != 3
        or reference.shape[2] != 3
        or reference.shape[0] != reference.shape[1]
        or reference.shape[0] % 8 != 0
    ):
        raise ValueError("Reference images must be square BGR board images.")
    size = reference.shape[0] // 8
    margin = round(size * 0.12)
    center_margin = round(size * 0.25)
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB)
    scores = np.zeros((8, 8), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            top = row * size
            left = column * size
            reference_region = reference_lab[
                top + margin : top + size - margin,
                left + margin : left + size - margin,
            ]
            candidate_region = candidate_lab[
                top + margin : top + size - margin,
                left + margin : left + size - margin,
            ]
            full_difference = float(
                np.mean(
                    cv2.absdiff(reference_region, candidate_region)
                )
                / 255
            )
            reference_center = reference_lab[
                top + center_margin : top + size - center_margin,
                left + center_margin : left + size - center_margin,
            ]
            candidate_center = candidate_lab[
                top + center_margin : top + size - center_margin,
                left + center_margin : left + size - center_margin,
            ]
            center_difference = float(
                np.mean(
                    cv2.absdiff(reference_center, candidate_center)
                )
                / 255
            )
            histogram_difference = 0.0
            for channel in range(3):
                reference_histogram = cv2.calcHist(
                    [reference_region],
                    [channel],
                    None,
                    [16],
                    [0, 256],
                )
                candidate_histogram = cv2.calcHist(
                    [candidate_region],
                    [channel],
                    None,
                    [16],
                    [0, 256],
                )
                cv2.normalize(
                    reference_histogram,
                    reference_histogram,
                    norm_type=cv2.NORM_L1,
                )
                cv2.normalize(
                    candidate_histogram,
                    candidate_histogram,
                    norm_type=cv2.NORM_L1,
                )
                histogram_difference += float(
                    cv2.compareHist(
                        reference_histogram,
                        candidate_histogram,
                        cv2.HISTCMP_BHATTACHARYYA,
                    )
                )
            scores[row, column] = (
                full_difference * 0.35
                + center_difference * 0.4
                + (histogram_difference / 3) * 0.25
            )
    return scores


def _ambiguous_change_message(scores: FloatArray) -> str:
    ranked = sorted(
        (
            (float(scores[row, column]), square_name((row, column)))
            for row in range(8)
            for column in range(8)
        ),
        reverse=True,
    )
    details = ", ".join(
        f"{name}={score:.3f}" for score, name in ranked[:4]
    )
    return f"Ambiguous settled board change: {details}."
