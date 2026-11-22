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
    match_pending_castling,
    match_pending_castling_relocation,
    match_pending_capture,
    match_pending_en_passant,
    prefilter_complete_signatures,
)
from .piece_color_classifier import (
    PieceColorClassification,
    PieceColorReferenceModel,
    build_piece_color_reference,
    classify_piece_colors_from_reference,
)
from .occupancy import rectify_board

ReferenceTrackerState = Literal[
    "uninitialized",
    "watching",
    "activity",
    "waiting-for-clear",
    "settling",
    "candidate",
    "pending-capture",
    "pending-castling",
    "pending-en-passant",
    "illegal-removal",
    "illegal-relocation",
    "accepted",
    "compound-accepted",
    "compound-correction",
    "compound-correction-restored",
    "ambiguous",
]


@dataclass(frozen=True)
class ReferenceTrackerUpdate:
    state: ReferenceTrackerState
    move: tuple[Square, Square] | None
    message: str
    change_scores: FloatArray | None = None
    decision_details: dict[str, object] | None = None
    compound_uci: str | None = None
    correction_moves: tuple[tuple[Square, Square], ...] = ()
    correction_squares: tuple[Square, ...] = ()
    opponent_move_proven: bool | None = None


@dataclass(frozen=True)
class _CompoundCorrection:
    opponent_move_proven: bool
    moves: tuple[tuple[Square, Square], ...]
    squares: tuple[Square, ...]


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
        self._piece_color_reference: PieceColorReferenceModel | None = None
        self._piece_color_reference_position: Position | None = None
        self._compound_correction_target: LegalMoveSignature | None = None
        self._compound_correction_position: Position | None = None
        self._compound_correction_moves: tuple[
            tuple[Square, Square], ...
        ] = ()
        self._compound_correction_squares: tuple[Square, ...] = ()

    @property
    def initialized(self) -> bool:
        return self._reference is not None

    @property
    def reference(self) -> ImageArray | None:
        return None if self._reference is None else self._reference.copy()

    def clear(self) -> None:
        self._reference = None
        self._camera_grid_remapped = False
        self._clear_piece_color_reference()
        self._clear_compound_correction()
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
        self._clear_piece_color_reference()
        self._clear_compound_correction()
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

    def stage_accepted_reference(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
        move: tuple[Square, Square],
    ) -> None:
        pending_reference = self.canonical_frame(
            frame,
            camera_grid,
            ui_quarter_turns,
        )
        self.reset_activity()
        self._candidate_move = move
        self._pending_reference = pending_reference

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
        self._clear_piece_color_reference()
        self._clear_compound_correction()
        self.reset_activity()

    def note_camera_grid_remap(self) -> None:
        self._camera_grid_remapped = True
        self._clear_piece_color_reference()
        self._clear_compound_correction()
        self.reset_activity()

    def observe(
        self,
        frame: ImageArray,
        camera_grid: FloatArray,
        ui_quarter_turns: int,
        position: Position,
        physical_signatures: tuple[LegalMoveSignature, ...] = (),
        *,
        required_signature: LegalMoveSignature | None = None,
        compound_signatures: tuple[LegalMoveSignature, ...] = (),
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
            self._ensure_piece_color_reference(position)
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "watching",
                None,
                "Board matches the accepted reference.",
                scores,
            )
        if (
            self._compound_correction_target is not None
            and self._compound_correction_position is not None
        ):
            correction_colors = self._classify_piece_colors(
                normalized,
                self._compound_correction_position,
            )
            correction_signatures = prefilter_complete_signatures(
                scores,
                (self._compound_correction_target,),
                piece_colors=correction_colors,
                minimum_square_change=self._minimum_square_change,
            )
            correction_match = match_legal_move_signature(
                scores,
                correction_signatures,
                piece_colors=correction_colors,
                minimum_square_change=self._minimum_square_change,
                minimum_dominance=self._minimum_dominance,
            )
            if correction_match.signature == self._compound_correction_target:
                self._pending_reference = normalized.copy()
                return ReferenceTrackerUpdate(
                    "compound-correction-restored",
                    (
                        self._compound_correction_target.source,
                        self._compound_correction_target.destination,
                    ),
                    "Incorrect local reply was undone; the opponent-only "
                    "position is verified.",
                    scores,
                    correction_moves=self._compound_correction_moves,
                    correction_squares=self._compound_correction_squares,
                    opponent_move_proven=True,
                )
            return ReferenceTrackerUpdate(
                "compound-correction",
                None,
                "Undo the incorrect local reply to restore the verified "
                "opponent-only position.",
                scores,
                correction_moves=self._compound_correction_moves,
                correction_squares=self._compound_correction_squares,
                opponent_move_proven=True,
            )
        move: tuple[Square, Square] | None = None
        pending_capture_destination: Square | None = None
        pending_capture_sources: tuple[Square, ...] = ()
        pending_castling: LegalMoveSignature | None = None
        pending_castling_rook_source: Square | None = None
        pending_castling_rook_destination: Square | None = None
        pending_en_passant_captured: Square | None = None
        pending_en_passant_destination: Square | None = None
        pending_en_passant_sources: tuple[Square, ...] = ()
        illegal_removed_piece: Square | None = None
        illegal_relocation = False
        matched_signature: LegalMoveSignature | None = None
        compound_ambiguous = False
        decision_details: dict[str, object] | None = None
        complete_candidate_signatures = (
            *physical_signatures,
            *compound_signatures,
        )
        if complete_candidate_signatures:
            piece_colors = self._classify_piece_colors(
                normalized,
                position,
            )
            complete_signatures = prefilter_complete_signatures(
                scores,
                complete_candidate_signatures,
                piece_colors=piece_colors,
                minimum_square_change=self._minimum_square_change,
            )
            explained_residuals = {
                signature.uci: tuple(
                    square
                    for square in (
                        (row, column)
                        for row in range(8)
                        for column in range(8)
                    )
                    if square not in (
                        signature.source,
                        signature.destination,
                    )
                    if _is_retained_opposite_piece_residual(
                        self._reference,
                        normalized,
                        square,
                        signature.source,
                        position,
                    )
                    or _matches_signature_result_state(
                        square,
                        signature,
                        position,
                        piece_colors,
                    )
                )
                for signature in complete_signatures
            }
            legal_match = match_legal_move_signature(
                scores,
                complete_signatures,
                explained_residuals=explained_residuals,
                piece_colors=piece_colors,
                minimum_square_change=self._minimum_square_change,
                minimum_dominance=self._minimum_dominance,
            )
            match_candidates = getattr(legal_match, "candidates", ())
            compound_match_candidates = tuple(
                evidence
                for evidence in match_candidates
                if evidence.signature in compound_signatures
            )
            if len(compound_match_candidates) == 1:
                matched_signature = compound_match_candidates[0].signature
                move = (
                    matched_signature.source,
                    matched_signature.destination,
                )
            elif len(compound_match_candidates) > 1:
                compound_ambiguous = True
            elif len(match_candidates) == 1:
                matched_signature = match_candidates[0].signature
                move = (
                    matched_signature.source,
                    matched_signature.destination,
                )
            elif len(match_candidates) > 1:
                compound_ambiguous = False
            elif legal_match.signature is not None:
                matched_signature = legal_match.signature
                move = (
                    matched_signature.source,
                    matched_signature.destination,
                )
            elif (
                required_signature is not None
                and compound_signatures
                and piece_colors is not None
                and not has_large_obstruction
            ):
                correction = _match_incorrect_compound_attempt(
                    scores,
                    position,
                    required_signature,
                    piece_colors,
                    minimum_square_change=self._minimum_square_change,
                )
                if correction is not None:
                    if correction.opponent_move_proven:
                        self._compound_correction_target = required_signature
                        self._compound_correction_position = dict(position)
                        self._compound_correction_moves = correction.moves
                        self._compound_correction_squares = correction.squares
                    self._clear_pending()
                    return ReferenceTrackerUpdate(
                        "compound-correction",
                        None,
                        (
                            "Opponent move verified; undo the incorrect "
                            "local reply."
                            if correction.opponent_move_proven
                            else "Opponent move is incorrect; restore both "
                            "physical relocations."
                        ),
                        scores,
                        correction_moves=correction.moves,
                        correction_squares=correction.squares,
                        opponent_move_proven=(
                            correction.opponent_move_proven
                        ),
                    )
            elif piece_colors is not None:
                castling_match = match_pending_castling(
                    scores,
                    physical_signatures,
                    piece_colors,
                    minimum_square_change=self._minimum_square_change,
                )
                pending_castling = castling_match.signature
                pending_castling_rook_source = castling_match.rook_source
                pending_castling_rook_destination = (
                    castling_match.rook_destination
                )
                if pending_castling is None:
                    pending_en_passant = match_pending_en_passant(
                        scores,
                        physical_signatures,
                        piece_colors,
                        minimum_square_change=self._minimum_square_change,
                    )
                    pending_en_passant_captured = (
                        pending_en_passant.captured_square
                    )
                    pending_en_passant_destination = (
                        pending_en_passant.destination
                    )
                    pending_en_passant_sources = (
                        pending_en_passant.sources
                    )
                    if pending_en_passant_captured is None:
                        pending_capture = match_pending_capture(
                            scores,
                            physical_signatures,
                            piece_colors,
                            minimum_square_change=self._minimum_square_change,
                        )
                        pending_capture_destination = (
                            pending_capture.destination
                        )
                        pending_capture_sources = pending_capture.sources
                if (
                    pending_castling is None
                    and pending_en_passant_captured is None
                    and pending_capture_destination is None
                ):
                    illegal_removed_piece = find_illegal_removed_piece(
                        scores,
                        physical_signatures,
                        piece_colors,
                        position,
                        minimum_square_change=self._minimum_square_change,
                    )
            if (
                move is None
                and pending_castling is None
                and pending_en_passant_captured is None
                and pending_capture_destination is None
                and piece_colors is not None
            ):
                relocation = self._select_illegal_relocation(
                    scores,
                    position,
                    self._reference,
                    normalized,
                    piece_colors,
                    allow_boundary_residuals=self._camera_grid_remapped,
                )
                if relocation is None:
                    relocation = self._select_move(
                        scores,
                        position,
                        self._reference,
                        normalized,
                        allow_boundary_residuals=self._camera_grid_remapped,
                    )
                if relocation is not None:
                    castling_match = match_pending_castling_relocation(
                        relocation,
                        physical_signatures,
                    )
                    pending_castling = castling_match.signature
                    pending_castling_rook_source = (
                        castling_match.rook_source
                    )
                    pending_castling_rook_destination = (
                        castling_match.rook_destination
                    )
                    if pending_castling is None:
                        move = relocation
                        if (
                            required_signature is not None
                            and required_signature.kind == "ordinary"
                            and relocation
                            == (
                                required_signature.source,
                                required_signature.destination,
                            )
                        ):
                            decision_details = {
                                "verification": (
                                    "required-geometric-fallback"
                                ),
                                "required_uci": required_signature.uci,
                                "source_score": float(
                                    scores[required_signature.source]
                                ),
                                "destination_score": float(
                                    scores[required_signature.destination]
                                ),
                            }
                        else:
                            illegal_relocation = True
        if (
            move is None
            and pending_castling is not None
            and pending_castling_rook_source is not None
            and pending_castling_rook_destination is not None
            and not has_large_obstruction
        ):
            self._candidate_move = None
            self._candidate_count = 0
            return ReferenceTrackerUpdate(
                "pending-castling",
                None,
                f"King moved {square_name(pending_castling.source)}-"
                f"{square_name(pending_castling.destination)}; complete "
                f"castling by moving the rook from "
                f"{square_name(pending_castling_rook_source)} to "
                f"{square_name(pending_castling_rook_destination)}.",
                scores,
            )
        if (
            move is None
            and pending_en_passant_captured is not None
            and pending_en_passant_destination is not None
            and not has_large_obstruction
        ):
            self._candidate_move = None
            self._candidate_count = 0
            source_names = ", ".join(
                square_name(source)
                for source in pending_en_passant_sources
            )
            return ReferenceTrackerUpdate(
                "pending-en-passant",
                None,
                f"Captured pawn removed from "
                f"{square_name(pending_en_passant_captured)}; complete en "
                f"passant from {source_names} to "
                f"{square_name(pending_en_passant_destination)}.",
                scores,
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
        if move is None and compound_ambiguous:
            self._candidate_move = None
            self._candidate_count = 0
            return ReferenceTrackerUpdate(
                "ambiguous",
                None,
                "Multiple compound Lichess replies match the settled board.",
                scores,
            )
        if move is None and not complete_candidate_signatures:
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
            ignored_squares=(
                move
                if matched_signature is None
                else tuple(
                    change.square for change in matched_signature.changes
                )
            ),
            allow_boundary_remap_residuals=self._camera_grid_remapped,
        ):
            self._clear_pending()
            return ReferenceTrackerUpdate(
                "waiting-for-clear",
                None,
                "A hand or object still obscures multiple board squares.",
                scores,
            )
        release_moves = (move,)
        if matched_signature in compound_signatures:
            release_moves = tuple(
                _uci_move(component)
                for component in matched_signature.uci.split("+")
            )
        if (
            not illegal_relocation
            and any(
                not _has_released_piece_footprint(
                    self._reference,
                    normalized,
                    release_move,
                )
                for release_move in release_moves
            )
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

        if illegal_relocation:
            return ReferenceTrackerUpdate(
                "illegal-relocation",
                move,
                f"Illegal relocation {square_name(move[0])}-"
                f"{square_name(move[1])}.",
                scores,
            )
        self._pending_reference = normalized.copy()
        compound_uci = (
            None
            if matched_signature not in compound_signatures
            else matched_signature.uci.split("+", maxsplit=1)[1]
        )
        return ReferenceTrackerUpdate(
            "accepted" if compound_uci is None else "compound-accepted",
            move,
            (
                f"Accepted {square_name(move[0])}-{square_name(move[1])}."
                if compound_uci is None
                else f"Compound Lichess reply {compound_uci} accepted."
            ),
            scores,
            decision_details,
            compound_uci,
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
        self._clear_piece_color_reference()
        self._clear_compound_correction()

    def commit_compound_correction_reference(self) -> None:
        if (
            self._compound_correction_target is None
            or self._pending_reference is None
        ):
            raise ValueError(
                "No verified compound correction reference is ready."
            )
        accepted_reference = self._pending_reference
        self.reset_activity()
        self._reference = accepted_reference
        self._camera_grid_remapped = False
        self._clear_piece_color_reference()
        self._clear_compound_correction()

    def _classify_piece_colors(
        self,
        candidate: ImageArray,
        position: Position,
    ) -> dict[Square, PieceColorClassification]:
        if self._reference is None:
            raise RuntimeError(
                "Piece colors cannot be classified without a reference."
            )
        if (
            self._piece_color_reference is None
            or self._piece_color_reference_position != position
        ):
            self._ensure_piece_color_reference(position)
        if self._piece_color_reference is None:
            raise RuntimeError("Piece-color reference cache was not built.")
        return classify_piece_colors_from_reference(
            self._piece_color_reference,
            candidate,
        )

    def _ensure_piece_color_reference(
        self,
        position: Position,
    ) -> None:
        if self._reference is None:
            raise RuntimeError(
                "Piece-color reference cannot be built without a reference."
            )
        if (
            self._piece_color_reference is not None
            and self._piece_color_reference_position == position
        ):
            return
        self._piece_color_reference = build_piece_color_reference(
            self._reference,
            position,
        )
        self._piece_color_reference_position = dict(position)

    def _clear_piece_color_reference(self) -> None:
        self._piece_color_reference = None
        self._piece_color_reference_position = None

    def _clear_compound_correction(self) -> None:
        self._compound_correction_target = None
        self._compound_correction_position = None
        self._compound_correction_moves = ()
        self._compound_correction_squares = ()

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

    def _select_illegal_relocation(
        self,
        scores: FloatArray,
        position: Position,
        reference: ImageArray,
        candidate: ImageArray,
        piece_colors: dict[Square, PieceColorClassification],
        *,
        allow_boundary_residuals: bool = False,
    ) -> tuple[Square, Square] | None:
        source_candidates = {
            square
            for square in position
            if (
                classification := piece_colors.get(square)
            ) is not None
            and classification.label == "empty"
        }
        if not source_candidates:
            return None
        sources = sorted(
            (
                (float(scores[square]), square)
                for square in source_candidates
            ),
            reverse=True,
        )
        source_score, source = sources[0]
        source_piece = position[source]
        source_color = "white" if source_piece.isupper() else "black"
        destination_candidates = {
            square
            for square, classification in piece_colors.items()
            if classification.label == source_color
            and (
                square not in position
                or position[square].isupper() != source_piece.isupper()
            )
        }
        if not destination_candidates:
            return None
        destinations = sorted(
            (
                (float(scores[square]), square)
                for square in destination_candidates
            ),
            reverse=True,
        )
        destination_score, destination = destinations[0]
        if (
            source_score < self._minimum_square_change * 0.75
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
                    and (
                        classification := piece_colors.get(square)
                    ) is not None
                    and (
                        square in position
                        and classification.label == "empty"
                        or square not in position
                        and classification.label != "empty"
                    )
                ):
                    return None
        return source, destination

    def _clear_pending(self) -> None:
        self._settled_count = 0
        self._candidate_move = None
        self._candidate_count = 0


def _match_incorrect_compound_attempt(
    scores: FloatArray,
    position: Position,
    required_signature: LegalMoveSignature,
    piece_colors: dict[Square, PieceColorClassification],
    *,
    minimum_square_change: float,
) -> _CompoundCorrection | None:
    required_piece = position.get(required_signature.source)
    if required_piece is None:
        return None
    opponent_is_white = required_piece.isupper()
    intermediate = _apply_signature_changes(position, required_signature)
    required_squares = {
        change.square for change in required_signature.changes
    }
    local_relocations = _observed_relocations(
        scores,
        intermediate,
        piece_colors,
        mover_is_white=not opponent_is_white,
        minimum_square_change=minimum_square_change,
    )
    proven_candidates = [
        relocation
        for relocation in local_relocations
        if required_squares.isdisjoint(relocation)
        and _relocation_sequence_matches(
            scores,
            position,
            (
                (
                    required_signature.source,
                    required_signature.destination,
                ),
                relocation,
            ),
            piece_colors,
            minimum_square_change=minimum_square_change,
            required_signature=required_signature,
        )
    ]
    if len(proven_candidates) == 1:
        local_move = proven_candidates[0]
        moves = (
            (
                required_signature.source,
                required_signature.destination,
            ),
            local_move,
        )
        return _CompoundCorrection(
            True,
            moves,
            tuple(dict.fromkeys(local_move)),
        )

    first_relocations = _observed_relocations(
        scores,
        position,
        piece_colors,
        mover_is_white=opponent_is_white,
        minimum_square_change=minimum_square_change,
    )
    incorrect_pairs: list[
        tuple[tuple[Square, Square], tuple[Square, Square]]
    ] = []
    required_move = (
        required_signature.source,
        required_signature.destination,
    )
    for first_move in first_relocations:
        if first_move == required_move:
            continue
        after_first = _apply_relocation(position, first_move)
        for second_move in _observed_relocations(
            scores,
            after_first,
            piece_colors,
            mover_is_white=not opponent_is_white,
            minimum_square_change=minimum_square_change,
        ):
            moves = (first_move, second_move)
            if _relocation_sequence_matches(
                scores,
                position,
                moves,
                piece_colors,
                minimum_square_change=minimum_square_change,
            ):
                incorrect_pairs.append(moves)
    unique_pairs = tuple(dict.fromkeys(incorrect_pairs))
    if len(unique_pairs) != 1:
        return None
    moves = unique_pairs[0]
    return _CompoundCorrection(
        False,
        moves,
        tuple(
            dict.fromkeys(
                square for relocation in moves for square in relocation
            )
        ),
    )


def _observed_relocations(
    scores: FloatArray,
    position: Position,
    piece_colors: dict[Square, PieceColorClassification],
    *,
    mover_is_white: bool,
    minimum_square_change: float,
) -> tuple[tuple[Square, Square], ...]:
    source_threshold = minimum_square_change * 0.7
    mover_label = "white" if mover_is_white else "black"
    sources = tuple(
        square
        for square, piece in position.items()
        if piece.isupper() == mover_is_white
        and float(scores[square]) >= source_threshold
        and (
            classification := piece_colors.get(square)
        ) is not None
        and classification.label == "empty"
    )
    destinations = tuple(
        (row, column)
        for row in range(8)
        for column in range(8)
        if float(scores[row, column]) >= source_threshold
        and (
            classification := piece_colors.get((row, column))
        ) is not None
        and classification.label == mover_label
    )
    return tuple(
        (source, destination)
        for source in sources
        for destination in destinations
        if source != destination
        and (
            destination not in position
            or position[destination].isupper() != mover_is_white
        )
    )


def _relocation_sequence_matches(
    scores: FloatArray,
    position: Position,
    moves: tuple[tuple[Square, Square], ...],
    piece_colors: dict[Square, PieceColorClassification],
    *,
    minimum_square_change: float,
    required_signature: LegalMoveSignature | None = None,
) -> bool:
    result = dict(position)
    touched: list[Square] = []
    start_index = 0
    if required_signature is not None:
        result = _apply_signature_changes(result, required_signature)
        touched.extend(
            change.square for change in required_signature.changes
        )
        start_index = 1
    for move in moves[start_index:]:
        source, destination = move
        piece = result.pop(source, None)
        if piece is None:
            return False
        result[destination] = piece
        touched.extend(move)
    touched_squares = tuple(dict.fromkeys(touched))
    if not touched_squares:
        return False
    for square in touched_squares:
        piece = result.get(square)
        expected = (
            "empty"
            if piece is None
            else "white"
            if piece.isupper()
            else "black"
        )
        classification = piece_colors.get(square)
        if classification is None or classification.label != expected:
            return False
    changed_scores = [float(scores[square]) for square in touched_squares]
    if any(
        score < minimum_square_change * 0.7
        for score in changed_scores
    ):
        return False
    unexpected_limit = max(
        minimum_square_change,
        min(changed_scores) * 0.75,
    )
    return all(
        square in touched_squares
        or float(scores[square]) <= unexpected_limit
        for square in (
            (row, column)
            for row in range(8)
            for column in range(8)
        )
    )


def _apply_signature_changes(
    position: Position,
    signature: LegalMoveSignature,
) -> Position:
    result = dict(position)
    for change in signature.changes:
        if change.after is None:
            result.pop(change.square, None)
        else:
            result[change.square] = change.after
    return result


def _apply_relocation(
    position: Position,
    move: tuple[Square, Square],
) -> Position:
    source, destination = move
    result = dict(position)
    piece = result.pop(source)
    result[destination] = piece
    return result


def _uci_move(uci: str) -> tuple[Square, Square]:
    if len(uci) not in (4, 5):
        raise ValueError("Compound verification requires a UCI move.")

    def square(value: str) -> Square:
        file_name, rank_name = value
        if file_name not in "abcdefgh" or rank_name not in "12345678":
            raise ValueError("Compound verification contains invalid UCI.")
        return 8 - int(rank_name), ord(file_name) - ord("a")

    return square(uci[:2]), square(uci[2:4])


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


def _matches_signature_result_state(
    square: Square,
    signature: LegalMoveSignature,
    position: Position,
    piece_colors: dict[Square, PieceColorClassification],
) -> bool:
    piece = position.get(square)
    for change in signature.changes:
        if change.square == square:
            piece = change.after
            break
    expected = (
        "empty"
        if piece is None
        else "white"
        if piece.isupper()
        else "black"
    )
    classification = piece_colors.get(square)
    return classification is not None and classification.label == expected


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
