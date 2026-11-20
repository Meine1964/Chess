from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .board_detector import FloatArray
from .chess_position import Square
from .legal_game import LegalMoveSignature
from .piece_color_classifier import PieceColorClassification


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
        expected = {change.square for change in signature.changes}
        expected_scores = [float(scores[square]) for square in expected]
        if not expected_scores:
            continue
        minimum_expected = min(expected_scores)
        if minimum_expected < minimum_square_change:
            continue
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
        mean_expected = float(np.mean(expected_scores))
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
