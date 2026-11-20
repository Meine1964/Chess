from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

from .board_detector import FloatArray, ImageArray
from .chess_position import Position, Square, square_name
from .legal_game import LegalMoveSignature
from .legal_signature_matcher import (
    find_illegal_removed_piece,
    match_legal_move_signature,
    match_pending_capture,
)
from .piece_color_classifier import classify_piece_colors
from .occupancy import rectify_board

ReferenceTrackerState = Literal[
    "uninitialized",
    "watching",
    "activity",
    "waiting-for-clear",
    "settling",
    "candidate",
    "pending-capture",
    "illegal-removal",
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
        self._camera_grid_remapped = False

    @property
    def initialized(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> ImageArray | None:
        return None if self._reference is None else self._reference.copy()

    def clear(self) -> None:
        self._reference = None
        self._camera_grid_remapped = False
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
        self._camera_grid_remapped = False
        self.reset_activity()

    def canonical_frame(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
    ) -> ImageArray:
        return _canonical_ui_board(
            frame,
            camera_grid,
            ui_quarter_turns,
            self._canonical_size,
        )

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
        self._camera_grid_remapped = False
        self.reset_activity()

    def note_camera_grid_remap(self) -> None:
        self._camera_grid_remapped = True
        self.reset_activity()

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
        position: Position,
        physical_signatures: tuple[LegalMoveSignature, ...] = (),
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

        has_large_obstruction = _has_large_obstruction(
            self._reference,
            normalized,
            allow_boundary_remap_residuals=self._camera_grid_remapped,
        )

        self._settled_count += 1
        if self._settled_count < self._settled_observations:
            if has_large_obstruction:
                return ReferenceTrackerUpdate(
                    "waiting-for-clear",
                    None,
                    "A hand or object may still obscure multiple board squares; "
                    "confirming the settled image.",
                )
            return ReferenceTrackerUpdate(
                "settling",
                None,
                "Board is clear; confirming a settled image.",
            )

        scores = square_change_scores(self._reference, normalized)
        if float(np.max(scores)) < self._minimum_square_change:
            if has_large_obstruction:
                self._clear_pending()
                self._camera_grid_remapped = False
                return ReferenceTrackerUpdate(
                    "waiting-for-clear",
                    None,
                    "A hand or object still obscures multiple board squares.",
                    scores,
                )
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "watching",
                None,
                "Board matches the accepted reference.",
                scores,
            )
        move: tuple[Square, Square] | None = None
        pending_capture_destination: Square | None = None
        pending_capture_sources: tuple[Square, ...] = ()
        illegal_removed_piece: Square | None = None
        if physical_signatures:
            piece_colors = classify_piece_colors(
                self._reference,
                normalized,
                position,
            )
            explained_residuals = {
                signature.uci: tuple(
                    square
                    for square in position
                    if _is_retained_opposite_piece_residual(
                        self._reference,
                        normalized,
                        square,
                        signature.source,
                        position,
                    )
                )
                for signature in physical_signatures
            }
            legal_match = match_legal_move_signature(
                scores,
                physical_signatures,
                explained_residuals=explained_residuals,
                piece_colors=piece_colors,
                minimum_square_change=self._minimum_square_change,
                minimum_dominance=self._minimum_dominance,
            )
            if legal_match.signature is not None:
                move = (
                    legal_match.signature.source,
                    legal_match.signature.destination,
                )
            elif piece_colors is not None:
                pending_capture = match_pending_capture(
                    scores,
                    physical_signatures,
                    piece_colors,
                    minimum_square_change=self._minimum_square_change,
                )
                pending_capture_destination = pending_capture.destination
                pending_capture_sources = pending_capture.sources
                if pending_capture_destination is None:
                    illegal_removed_piece = find_illegal_removed_piece(
                        scores,
                        physical_signatures,
                        piece_colors,
                        position,
                        minimum_square_change=self._minimum_square_change,
                    )
        if (
            move is None
            and pending_capture_destination is not None
            and not has_large_obstruction
        ):
            self._candidate_move = None
            self._candidate_count = 0
            source_names = ", ".join(
                square_name(source) for source in pending_capture_sources
            )
            return ReferenceTrackerUpdate(
                "pending-capture",
                None,
                f"Captured piece removed from "
                f"{square_name(pending_capture_destination)}; complete the "
                f"capture from {source_names}.",
                scores,
            )
        if (
            move is None
            and illegal_removed_piece is not None
            and not has_large_obstruction
        ):
            self._candidate_move = None
            self._candidate_count = 0
            return ReferenceTrackerUpdate(
                "illegal-removal",
                (illegal_removed_piece, illegal_removed_piece),
                f"Illegal removal from {square_name(illegal_removed_piece)}; "
                "restore the piece to the marked square.",
                scores,
            )
        if move is None:
            move = self._select_move(
                scores,
                position,
                self._reference,
                normalized,
                allow_boundary_residuals=self._camera_grid_remapped,
            )
        if move is None:
            self._candidate_move = None
            self._candidate_count = 0
            if has_large_obstruction:
                return ReferenceTrackerUpdate(
                    "waiting-for-clear",
                    None,
                    "A hand or object still obscures multiple board squares.",
                    scores,
                )
            return ReferenceTrackerUpdate(
                "ambiguous",
                None,
                _ambiguous_change_message(scores),
                scores,
            )
        if _has_large_obstruction(
            self._reference,
            normalized,
            ignored_squares=move,
            allow_boundary_remap_residuals=self._camera_grid_remapped,
        ):
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "waiting-for-clear",
                None,
                "A hand or object still obscures multiple board squares.",
                scores,
            )
        if not _has_released_piece_footprint(
            self._reference,
            normalized,
            move,
        ):
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "waiting-for-clear",
                None,
                "A hand or object still covers the moved piece.",
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
        self._camera_grid_remapped = False

    def rejected_move_is_restored(
        self,
        update: ReferenceTrackerUpdate,
        move: tuple[Square, Square],
    ) -> bool:
        if (
            update.state not in ("watching", "ambiguous")
            or update.change_scores is None
        ):
            return False
        source, _destination = move
        scores = np.asarray(update.change_scores, dtype=np.float32).copy()
        if scores.shape != (8, 8):
            raise ValueError("Change scores must describe an 8x8 board.")
        source_score = float(scores[source])
        scores[source] = 0
        return (
            source_score
            < self._minimum_square_change * 1.5
            and float(np.max(scores)) < self._minimum_square_change
        )

    def _select_move(
        self,
        scores: FloatArray,
        position: Position,
        reference: ImageArray,
        candidate: ImageArray,
        *,
        allow_boundary_residuals: bool = False,
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
        remaining_limit = max(
            self._minimum_square_change,
            min(source_score, destination_score) * 0.75,
        )
        for row in range(8):
            for column in range(8):
                square = (row, column)
                if (
                    float(remaining[square]) > remaining_limit
                    and not (
                        allow_boundary_residuals
                        and _is_boundary_square(square)
                    )
                    and not _is_retained_opposite_piece_residual(
                        reference,
                        candidate,
                        square,
                        source,
                        position,
                    )
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
    ignored_squares: tuple[Square, ...] = (),
    *,
    allow_boundary_remap_residuals: bool = False,
) -> bool:
    difference = cv2.absdiff(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )
    mask = np.asarray(difference > 24, dtype=np.uint8) * 255
    square_size = reference.shape[0] // 8
    for row, column in ignored_squares:
        mask[
            row * square_size : (row + 1) * square_size,
            column * square_size : (column + 1) * square_size,
        ] = 0
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
    component_count, labels, statistics, _centroids = (
        cv2.connectedComponentsWithStats(mask)
    )
    minimum_square_pixels = square_size * square_size * 0.15
    board_area = reference.shape[0] * reference.shape[1]
    obstruction_components: list[tuple[set[Square], set[str]]] = []
    residual_sides: set[str] = set()
    for component in range(1, component_count):
        area = statistics[component, cv2.CC_STAT_AREA]
        covered_squares = {
            (row, column)
            for row in range(8)
            for column in range(8)
            if np.count_nonzero(
                labels[
                    row * square_size : (row + 1) * square_size,
                    column * square_size : (column + 1) * square_size,
                ]
                == component
            )
            >= minimum_square_pixels
        }
        sides = _component_boundary_sides(
            statistics[component],
            mask.shape,
        )
        if (
            area >= board_area * 0.0015
            and sides
            and covered_squares
            and all(_is_boundary_square(square) for square in covered_squares)
        ):
            residual_sides.update(sides)
        if (
            area >= board_area * 0.0125
            and len(covered_squares) >= 3
        ) or (
            area >= board_area * 0.02
            and len(covered_squares) >= 2
        ):
            obstruction_components.append((covered_squares, sides))
    if not obstruction_components:
        return False
    if (
        allow_boundary_remap_residuals
        and len(residual_sides) >= 2
        and all(
            sides
            and covered_squares
            and all(
                _is_boundary_square(square)
                for square in covered_squares
            )
            for covered_squares, sides in obstruction_components
        )
    ):
        return False
    return True


def _is_boundary_square(square: Square) -> bool:
    row, column = square
    return row in (0, 7) or column in (0, 7)


def _component_boundary_sides(
    statistics: np.ndarray,
    shape: tuple[int, int],
) -> set[str]:
    x, y, width, height, _area = (int(value) for value in statistics)
    image_height, image_width = shape
    sides: set[str] = set()
    if x == 0:
        sides.add("left")
    if x + width >= image_width:
        sides.add("right")
    if y == 0:
        sides.add("top")
    if y + height >= image_height:
        sides.add("bottom")
    return sides


def _has_released_piece_footprint(
    reference: ImageArray,
    candidate: ImageArray,
    move: tuple[Square, Square],
) -> bool:
    difference = cv2.absdiff(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY),
    )
    mask = np.asarray(difference > 24, dtype=np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )
    square_size = reference.shape[0] // 8
    changed_fractions: list[float] = []
    for row, column in move:
        region = mask[
            row * square_size : (row + 1) * square_size,
            column * square_size : (column + 1) * square_size,
        ]
        changed_fractions.append(float(np.mean(region)))
    source_fraction, destination_fraction = changed_fractions
    if source_fraction <= 0 or destination_fraction <= 0:
        return False
    if max(source_fraction, destination_fraction) > 0.7:
        return False
    ratio = destination_fraction / source_fraction
    return 0.33 <= ratio <= 3.0


def _is_retained_opposite_piece_residual(
    reference: ImageArray,
    candidate: ImageArray,
    square: Square,
    source: Square,
    position: Position,
) -> bool:
    retained_piece = position.get(square)
    moving_piece = position.get(source)
    if (
        retained_piece is None
        or moving_piece is None
        or retained_piece.isupper() == moving_piece.isupper()
        or max(
            abs(square[0] - source[0]),
            abs(square[1] - source[1]),
        )
        > 1
    ):
        return False
    square_size = reference.shape[0] // 8
    row, column = square
    top = row * square_size
    left = column * square_size
    reference_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)
    candidate_lab = cv2.cvtColor(candidate, cv2.COLOR_BGR2LAB)
    reference_base = reference_lab[
        top + round(square_size * 0.55) : top + round(square_size * 0.9),
        left + round(square_size * 0.25) : left + round(square_size * 0.75),
    ]
    candidate_base = candidate_lab[
        top + round(square_size * 0.55) : top + round(square_size * 0.9),
        left + round(square_size * 0.25) : left + round(square_size * 0.75),
    ]
    base_difference = float(
        np.mean(cv2.absdiff(reference_base, candidate_base)) / 255
    )
    return base_difference <= 0.055


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
