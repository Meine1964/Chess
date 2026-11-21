from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .board_detector import FloatArray
from .chess_position import Position, Square
from .legal_game import LegalMoveSignature
from .piece_color_classifier import PieceColorClassification

LEGAL_SOURCE_CHANGE_RATIO = 0.7
COLOR_CONFIRMED_CAPTURE_CHANGE_RATIO = 0.3


@dataclass(frozen=True)
class SignatureEvidence:
    signature: LegalMoveSignature
    minimum_expected_change: float
    mean_expected_change: float
    strongest_unexpected_change: float
    score: float


@dataclass(frozen=True)
class LegalSignatureMatch:
    signature: LegalMoveSignature | None
    candidates: tuple[SignatureEvidence, ...]
    message: str


@dataclass(frozen=True)
class PendingCaptureMatch:
    destination: Square | None
    sources: tuple[Square, ...]
    signatures: tuple[LegalMoveSignature, ...]
    message: str


@dataclass(frozen=True)
class PendingCastlingMatch:
    signature: LegalMoveSignature | None
    rook_source: Square | None
    rook_destination: Square | None
    message: str


@dataclass(frozen=True)
class PendingEnPassantMatch:
    captured_square: Square | None
    destination: Square | None
    sources: tuple[Square, ...]
    signatures: tuple[LegalMoveSignature, ...]
    message: str


def prefilter_complete_signatures(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    *,
    piece_colors: Mapping[Square, PieceColorClassification] | None = None,
    minimum_square_change: float = 0.08,
) -> tuple[LegalMoveSignature, ...]:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0:
        raise ValueError("The movement threshold must be positive.")
    relaxed_threshold = minimum_square_change * LEGAL_SOURCE_CHANGE_RATIO
    capture_threshold = (
        minimum_square_change * COLOR_CONFIRMED_CAPTURE_CHANGE_RATIO
    )
    return tuple(
        signature
        for signature in signatures
        if float(scores[signature.source]) >= relaxed_threshold
        and (
            float(scores[signature.destination]) >= relaxed_threshold
            or (
                piece_colors is not None
                and signature.kind in ("capture", "capture-promotion")
                and float(scores[signature.destination]) >= capture_threshold
                and _capture_colors_match(signature, piece_colors)
            )
        )
    )


def match_legal_move_signature(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    *,
    explained_residuals: Mapping[str, Collection[Square]] | None = None,
    piece_colors: Mapping[Square, PieceColorClassification] | None = None,
    minimum_square_change: float = 0.08,
    minimum_dominance: float = 0.025,
) -> LegalSignatureMatch:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0 or minimum_dominance <= 0:
        raise ValueError("Signature thresholds must be positive.")
    explained = {} if explained_residuals is None else explained_residuals
    evidence: list[SignatureEvidence] = []
    for signature in signatures:
        if (
            piece_colors is not None
            and signature.kind in ("capture", "capture-promotion")
            and not _capture_colors_match(signature, piece_colors)
        ):
            continue
        if (
            piece_colors is not None
            and signature.kind in ("castling", "en-passant", "promotion")
            and not _changed_square_colors_match(signature, piece_colors)
        ):
            continue
        expected = {change.square for change in signature.changes}
        expected_scores = {
            square: float(scores[square]) for square in expected
        }
        if not expected_scores:
            continue
        source_score = expected_scores.get(signature.source)
        capture_colors_confirmed = (
            signature.kind in ("capture", "capture-promotion")
            and piece_colors is not None
            and _capture_colors_match(signature, piece_colors)
        )
        ordinary_colors_confirmed = (
            signature.kind == "ordinary"
            and piece_colors is not None
            and _changed_square_colors_match(signature, piece_colors)
        )
        required_change = (
            minimum_square_change * COLOR_CONFIRMED_CAPTURE_CHANGE_RATIO
            if capture_colors_confirmed
            else (
                minimum_square_change * LEGAL_SOURCE_CHANGE_RATIO
                if (
                    signature.kind in ("castling", "en-passant", "promotion")
                    or ordinary_colors_confirmed
                )
                else minimum_square_change
            )
        )
        if (
            source_score is None
            or source_score
            < minimum_square_change * LEGAL_SOURCE_CHANGE_RATIO
            or any(
                score < required_change
                for square, score in expected_scores.items()
                if square != signature.source
            )
        ):
            continue
        minimum_expected = min(expected_scores.values())
        ignored = expected | set(explained.get(signature.uci, ()))
        unexpected_scores = [
            float(scores[row, column])
            for row in range(8)
            for column in range(8)
            if (row, column) not in ignored
        ]
        strongest_unexpected = max(unexpected_scores, default=0.0)
        unexpected_limit = max(
            minimum_square_change,
            minimum_expected * 0.75,
        )
        if strongest_unexpected > unexpected_limit:
            continue
        mean_expected = float(np.mean(tuple(expected_scores.values())))
        evidence.append(
            SignatureEvidence(
                signature,
                minimum_expected,
                mean_expected,
                strongest_unexpected,
                mean_expected - strongest_unexpected,
            )
        )
    ranked = tuple(
        sorted(
            evidence,
            key=lambda item: (-item.score, item.signature.uci),
        )
    )
    if not ranked:
        return LegalSignatureMatch(
            None,
            (),
            "No legal physical signature matches the settled board.",
        )
    if (
        len(ranked) > 1
        and ranked[0].score - ranked[1].score < minimum_dominance
    ):
        return LegalSignatureMatch(
            None,
            ranked,
            "Multiple legal physical signatures match the settled board.",
        )
    return LegalSignatureMatch(
        ranked[0].signature,
        ranked,
        f"Legal physical signature {ranked[0].signature.uci} matched.",
    )


def match_pending_capture(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    piece_colors: Mapping[Square, PieceColorClassification],
    *,
    minimum_square_change: float = 0.08,
) -> PendingCaptureMatch:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0:
        raise ValueError("The movement threshold must be positive.")
    candidates: list[LegalMoveSignature] = []
    for signature in signatures:
        if signature.kind not in ("capture", "capture-promotion"):
            continue
        source_color = piece_colors.get(signature.source)
        destination_color = piece_colors.get(signature.destination)
        moving_piece = next(
            (
                change.before
                for change in signature.changes
                if change.square == signature.source
            ),
            None,
        )
        expected_source_color = (
            None
            if moving_piece is None
            else "white"
            if moving_piece.isupper()
            else "black"
        )
        source_score = float(scores[signature.source])
        destination_score = float(scores[signature.destination])
        if (
            expected_source_color is None
            or source_color is None
            or source_color.label != expected_source_color
            or destination_color is None
            or destination_color.label != "empty"
            or source_score >= minimum_square_change
            or destination_score < minimum_square_change
        ):
            continue
        unexplained = scores.copy()
        unexplained[signature.destination] = 0
        unexpected_limit = max(
            minimum_square_change,
            destination_score * 0.75,
        )
        if float(np.max(unexplained)) > unexpected_limit:
            continue
        candidates.append(signature)
    destinations = {signature.destination for signature in candidates}
    if len(destinations) != 1:
        return PendingCaptureMatch(
            None,
            (),
            tuple(candidates),
            "No unique pending capture destination matches the settled board.",
        )
    destination = destinations.pop()
    sources = tuple(
        sorted({signature.source for signature in candidates})
    )
    return PendingCaptureMatch(
        destination,
        sources,
        tuple(candidates),
        "Captured piece removal matches a pending physical capture.",
    )


def match_pending_castling(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    piece_colors: Mapping[Square, PieceColorClassification],
    *,
    minimum_square_change: float = 0.08,
) -> PendingCastlingMatch:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0:
        raise ValueError("The movement threshold must be positive.")
    candidates: list[tuple[LegalMoveSignature, Square, Square]] = []
    for signature in signatures:
        if signature.kind != "castling":
            continue
        rook_source = next(
            (
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
                and change.before in ("R", "r")
                and change.after is None
            ),
            None,
        )
        rook_destination = next(
            (
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
                and change.before is None
                and change.after in ("R", "r")
            ),
            None,
        )
        moving_piece = next(
            (
                change.before
                for change in signature.changes
                if change.square == signature.source
            ),
            None,
        )
        expected_color = (
            None
            if moving_piece is None
            else "white"
            if moving_piece.isupper()
            else "black"
        )
        if (
            rook_source is None
            or rook_destination is None
            or expected_color is None
            or float(scores[signature.source])
            < minimum_square_change * LEGAL_SOURCE_CHANGE_RATIO
            or float(scores[signature.destination]) < minimum_square_change
            or float(scores[rook_source]) >= minimum_square_change
            or float(scores[rook_destination]) >= minimum_square_change
            or piece_colors.get(signature.source) is None
            or piece_colors[signature.source].label != "empty"
            or piece_colors.get(signature.destination) is None
            or piece_colors[signature.destination].label != expected_color
            or piece_colors.get(rook_source) is None
            or piece_colors[rook_source].label != expected_color
            or piece_colors.get(rook_destination) is None
            or piece_colors[rook_destination].label != "empty"
        ):
            continue
        unexplained = scores.copy()
        unexplained[signature.source] = 0
        unexplained[signature.destination] = 0
        unexpected_limit = max(
            minimum_square_change,
            min(
                float(scores[signature.source]),
                float(scores[signature.destination]),
            )
            * 0.75,
        )
        if float(np.max(unexplained)) > unexpected_limit:
            continue
        candidates.append((signature, rook_source, rook_destination))
    if len(candidates) != 1:
        return PendingCastlingMatch(
            None,
            None,
            None,
            "No unique king-first castling signature matches the board.",
        )
    signature, rook_source, rook_destination = candidates[0]
    return PendingCastlingMatch(
        signature,
        rook_source,
        rook_destination,
        "The king move matches pending physical castling.",
    )


def match_pending_castling_relocation(
    move: tuple[Square, Square],
    signatures: Sequence[LegalMoveSignature],
) -> PendingCastlingMatch:
    candidates: list[tuple[LegalMoveSignature, Square, Square]] = []
    for signature in signatures:
        if (
            signature.kind != "castling"
            or (signature.source, signature.destination) != move
        ):
            continue
        rook_source = next(
            (
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
                and change.before in ("R", "r")
                and change.after is None
            ),
            None,
        )
        rook_destination = next(
            (
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
                and change.before is None
                and change.after in ("R", "r")
            ),
            None,
        )
        if rook_source is not None and rook_destination is not None:
            candidates.append((signature, rook_source, rook_destination))
    if len(candidates) != 1:
        return PendingCastlingMatch(
            None,
            None,
            None,
            "No unique castling signature matches the king relocation.",
        )
    signature, rook_source, rook_destination = candidates[0]
    return PendingCastlingMatch(
        signature,
        rook_source,
        rook_destination,
        "The detected king relocation matches pending physical castling.",
    )


def match_pending_en_passant(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    piece_colors: Mapping[Square, PieceColorClassification],
    *,
    minimum_square_change: float = 0.08,
) -> PendingEnPassantMatch:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0:
        raise ValueError("The movement threshold must be positive.")
    candidates: list[tuple[LegalMoveSignature, Square]] = []
    for signature in signatures:
        if signature.kind != "en-passant":
            continue
        captured_square = next(
            (
                change.square
                for change in signature.changes
                if change.square
                not in (signature.source, signature.destination)
                and change.before in ("P", "p")
                and change.after is None
            ),
            None,
        )
        moving_piece = next(
            (
                change.before
                for change in signature.changes
                if change.square == signature.source
            ),
            None,
        )
        expected_color = (
            None
            if moving_piece is None
            else "white"
            if moving_piece.isupper()
            else "black"
        )
        if (
            captured_square is None
            or expected_color is None
            or float(scores[signature.source]) >= minimum_square_change
            or float(scores[signature.destination]) >= minimum_square_change
            or float(scores[captured_square])
            < minimum_square_change * LEGAL_SOURCE_CHANGE_RATIO
            or piece_colors.get(signature.source) is None
            or piece_colors[signature.source].label != expected_color
            or piece_colors.get(signature.destination) is None
            or piece_colors[signature.destination].label != "empty"
            or piece_colors.get(captured_square) is None
            or piece_colors[captured_square].label != "empty"
        ):
            continue
        unexplained = scores.copy()
        unexplained[captured_square] = 0
        unexpected_limit = max(
            minimum_square_change,
            float(scores[captured_square]) * 0.75,
        )
        if float(np.max(unexplained)) > unexpected_limit:
            continue
        candidates.append((signature, captured_square))
    target_pairs = {
        (captured_square, signature.destination)
        for signature, captured_square in candidates
    }
    if len(target_pairs) != 1:
        return PendingEnPassantMatch(
            None,
            None,
            (),
            tuple(signature for signature, _square in candidates),
            "No unique captured-pawn-first en passant signature matches "
            "the board.",
        )
    captured_square, destination = target_pairs.pop()
    return PendingEnPassantMatch(
        captured_square,
        destination,
        tuple(sorted({signature.source for signature, _square in candidates})),
        tuple(signature for signature, _square in candidates),
        "The removed pawn matches pending physical en passant.",
    )


def find_illegal_removed_piece(
    change_scores: FloatArray,
    signatures: Sequence[LegalMoveSignature],
    piece_colors: Mapping[Square, PieceColorClassification],
    position: Position,
    *,
    minimum_square_change: float = 0.08,
) -> Square | None:
    scores = np.asarray(change_scores, dtype=np.float32)
    if scores.shape != (8, 8):
        raise ValueError("Change scores must describe an 8x8 board.")
    if minimum_square_change <= 0:
        raise ValueError("The movement threshold must be positive.")
    removed_pieces = [
        square
        for square in position
        if float(scores[square]) >= minimum_square_change
        and (
            classification := piece_colors.get(square)
        ) is not None
        and classification.label == "empty"
    ]
    if len(removed_pieces) != 1:
        return None
    removed = removed_pieces[0]
    if any(
        (
            signature.kind in ("capture", "capture-promotion")
            and signature.destination == removed
        )
        or (
            signature.kind == "en-passant"
            and any(
                change.square == removed
                and change.square
                not in (signature.source, signature.destination)
                and change.before is not None
                and change.after is None
                for change in signature.changes
            )
        )
        for signature in signatures
    ):
        return None
    unexplained = scores.copy()
    removed_score = float(unexplained[removed])
    unexplained[removed] = 0
    unexpected_limit = max(
        minimum_square_change,
        removed_score * 0.75,
    )
    if float(np.max(unexplained)) > unexpected_limit:
        return None
    return removed


def _capture_colors_match(
    signature: LegalMoveSignature,
    piece_colors: Mapping[Square, PieceColorClassification],
) -> bool:
    for change in signature.changes:
        if change.square == signature.source:
            expected = "empty"
        elif change.square == signature.destination and change.after is not None:
            expected = "white" if change.after.isupper() else "black"
        else:
            continue
        classification = piece_colors.get(change.square)
        if classification is None or classification.label != expected:
            return False
    return True


def _changed_square_colors_match(
    signature: LegalMoveSignature,
    piece_colors: Mapping[Square, PieceColorClassification],
) -> bool:
    for change in signature.changes:
        expected = (
            "empty"
            if change.after is None
            else "white"
            if change.after.isupper()
            else "black"
        )
        classification = piece_colors.get(change.square)
        if classification is None or classification.label != expected:
            return False
    return True
