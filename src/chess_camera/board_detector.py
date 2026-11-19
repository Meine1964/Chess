from dataclasses import dataclass
from typing import Literal, TypeAlias

import cv2
import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float32]
ImageArray: TypeAlias = NDArray[np.uint8]


@dataclass(frozen=True)
class BoardDetection:
    """Detected board geometry in camera pixel coordinates."""

    grid_points: FloatArray
    confidence: float
    reprojection_error: float | None
    method: str
    grid_alignment: float

    @property
    def outer_corners(self) -> FloatArray:
        return self.grid_points[[0, 0, 8, 8], [0, 8, 8, 0]]


TrackingState = Literal[
    "searching",
    "candidate_accumulating",
    "stable_board",
    "motion_or_occlusion",
    "insufficient_evidence",
]


class StableBoardTracker:
    """Keep confirmed board geometry stable through motion and occlusion."""

    def __init__(
        self,
        detector: "BoardDetector",
        *,
        stable_frame_count: int = 3,
        motion_pixel_threshold: int = 25,
        motion_fraction_threshold: float = 0.03,
        unchanged_corner_threshold: float = 2.0,
        candidate_corner_threshold: float = 6.0,
    ) -> None:
        if stable_frame_count < 1:
            raise ValueError("Stable frame count must be at least one.")
        if motion_pixel_threshold < 1:
            raise ValueError("Motion pixel threshold must be positive.")
        if not 0 < motion_fraction_threshold < 1:
            raise ValueError("Motion fraction threshold must be between 0 and 1.")
        if unchanged_corner_threshold <= 0 or candidate_corner_threshold <= 0:
            raise ValueError("Corner thresholds must be positive.")

        self._detector = detector
        self._stable_frame_count = stable_frame_count
        self._motion_pixel_threshold = motion_pixel_threshold
        self._motion_fraction_threshold = motion_fraction_threshold
        self._unchanged_corner_threshold = unchanged_corner_threshold
        self._candidate_corner_threshold = candidate_corner_threshold
        self._confirmed: BoardDetection | None = None
        self._candidate: BoardDetection | None = None
        self._candidate_count = 0
        self._previous_gray: ImageArray | None = None
        self._state: TrackingState = "searching"

    @property
    def state(self) -> TrackingState:
        return self._state

    @property
    def confirmed(self) -> BoardDetection | None:
        return self._confirmed

    def reset(self, confirmed: BoardDetection | None = None) -> None:
        self._confirmed = confirmed
        self._candidate = None
        self._candidate_count = 0
        self._previous_gray = None
        self._state = "stable_board" if confirmed is not None else "searching"

    def detect(self, frame: ImageArray) -> BoardDetection | None:
        gray = self._gray(frame)
        if self._has_motion(gray):
            self._candidate = None
            self._candidate_count = 0
            self._previous_gray = gray.copy()
            self._state = "motion_or_occlusion"
            return self._confirmed
        self._previous_gray = gray.copy()

        detection = self._detector.detect(frame)
        if detection is None:
            self._candidate = None
            self._candidate_count = 0
            self._state = (
                "motion_or_occlusion"
                if self._confirmed is not None
                else "insufficient_evidence"
            )
            return self._confirmed

        if self._confirmed is not None:
            confirmed_displacement = self._corner_displacement(
                detection,
                self._confirmed,
            )
            if confirmed_displacement <= self._unchanged_corner_threshold:
                self._candidate = None
                self._candidate_count = 0
                self._state = "stable_board"
                return self._confirmed

        if (
            self._candidate is not None
            and self._corner_displacement(detection, self._candidate)
            <= self._candidate_corner_threshold
        ):
            self._candidate_count += 1
            self._candidate = detection
        else:
            self._candidate = detection
            self._candidate_count = 1

        if self._candidate_count >= self._stable_frame_count:
            self._confirmed = self._candidate
            self._candidate = None
            self._candidate_count = 0
            self._state = "stable_board"
            return self._confirmed

        self._state = "candidate_accumulating"
        return self._confirmed

    def _has_motion(self, gray: ImageArray) -> bool:
        if self._previous_gray is None or self._previous_gray.shape != gray.shape:
            return False
        difference = cv2.absdiff(gray, self._previous_gray)
        moving = difference >= self._motion_pixel_threshold
        if self._confirmed is None:
            return float(np.mean(moving)) >= self._motion_fraction_threshold

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.fillConvexPoly(
            mask,
            np.rint(self._confirmed.outer_corners).astype(np.int32),
            1,
        )
        board_pixels = int(np.count_nonzero(mask))
        if board_pixels == 0:
            return False
        moving_board_pixels = int(np.count_nonzero(np.logical_and(moving, mask)))
        return (
            moving_board_pixels / board_pixels
            >= self._motion_fraction_threshold
        )

    @staticmethod
    def _corner_displacement(
        first: BoardDetection,
        second: BoardDetection,
    ) -> float:
        return float(
            np.max(
                np.linalg.norm(
                    first.outer_corners - second.outer_corners,
                    axis=1,
                )
            )
        )

    @staticmethod
    def _gray(frame: ImageArray) -> ImageArray:
        if frame.ndim == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.ndim == 2:
            return frame
        raise ValueError("Expected a grayscale or BGR camera frame.")


class BoardDetector:
    """Detect an 8x8 playing plane and reconstruct its equal-square grid."""

    _INNER_PATTERN = (7, 7)
    _DETECTION_FLAGS = (
        cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    _FALLBACK_FLAGS = cv2.CALIB_CB_NORMALIZE_IMAGE

    def detect(self, frame: ImageArray) -> BoardDetection | None:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Expected a grayscale or BGR camera frame.")

        detections: list[BoardDetection] = []
        found, corners = self._find_inner_corners(gray)
        if found and corners is not None:
            internal_detection = self._detection_from_inner_corners(
                corners,
                gray.shape,
            )
            if internal_detection is not None:
                detections.append(internal_detection)

        quadrilateral_detections = sorted(
            self._detect_by_quadrilateral(frame, gray),
            key=lambda detection: (
                detection.grid_alignment,
                detection.confidence,
            ),
            reverse=True,
        )
        for detection in quadrilateral_detections:
            if not any(
                self._detections_overlap(detection, existing)
                for existing in detections
            ):
                detections.append(detection)

        if not detections:
            return None
        image_height, image_width = gray.shape
        image_center = np.array(
            [image_width / 2, image_height / 2],
            dtype=np.float32,
        )
        return min(
            detections,
            key=lambda detection: float(
                np.linalg.norm(detection.grid_points[4, 4] - image_center)
            ),
        )

    def _detection_from_inner_corners(
        self,
        corners: cv2.typing.MatLike,
        image_shape: tuple[int, int],
    ) -> BoardDetection | None:
        inner_points = corners.reshape(7, 7, 2).astype(np.float32)
        model_points = np.array(
            [(column, row) for row in range(1, 8) for column in range(1, 8)],
            dtype=np.float32,
        )
        image_points = inner_points.reshape(-1, 2)
        homography, inlier_mask = cv2.findHomography(
            model_points,
            image_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=2.0,
        )
        if homography is None or inlier_mask is None:
            return None

        grid_model = np.array(
            [(column, row) for row in range(9) for column in range(9)],
            dtype=np.float32,
        )
        grid_points = np.asarray(
            cv2.perspectiveTransform(
                grid_model.reshape(-1, 1, 2),
                homography,
            ),
            dtype=np.float32,
        ).reshape(9, 9, 2)
        typed_inlier_mask = np.asarray(inlier_mask, dtype=np.uint8)

        if not self._is_valid_grid(grid_points, image_shape):
            return None

        projected_inner = cv2.perspectiveTransform(
            model_points.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        residuals = np.linalg.norm(projected_inner - image_points, axis=1)
        reprojection_error = float(np.sqrt(np.mean(np.square(residuals))))
        confidence = self._calculate_confidence(
            grid_points,
            typed_inlier_mask,
            reprojection_error,
            image_shape,
        )

        return BoardDetection(
            grid_points=grid_points.astype(np.float32),
            confidence=confidence,
            reprojection_error=reprojection_error,
            method="internal corners",
            grid_alignment=1.0,
        )

    def _find_inner_corners(
        self,
        gray: ImageArray,
    ) -> tuple[bool, cv2.typing.MatLike | None]:
        found, corners = cv2.findChessboardCornersSB(
            gray,
            self._INNER_PATTERN,
            flags=self._DETECTION_FLAGS,
        )
        if found:
            return found, corners

        normalized = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)
        return cv2.findChessboardCornersSB(
            normalized,
            self._INNER_PATTERN,
            flags=self._FALLBACK_FLAGS,
        )

    def _detect_by_quadrilateral(
        self,
        frame: ImageArray,
        gray: ImageArray,
    ) -> list[BoardDetection]:
        color_frame = (
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if frame.ndim == 2
            else frame
        )
        image_height, image_width = gray.shape
        image_area = image_height * image_width
        normalized = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        ).apply(gray)

        detections: list[BoardDetection] = []
        kernel = np.ones((3, 3), dtype=np.uint8)
        for source in (gray, normalized):
            for low_threshold, high_threshold in ((30, 90), (50, 150), (80, 200)):
                edges = cv2.Canny(source, low_threshold, high_threshold)
                edges = cv2.dilate(edges, kernel)
                contours, _hierarchy = cv2.findContours(
                    edges,
                    cv2.RETR_LIST,
                    cv2.CHAIN_APPROX_SIMPLE,
                )
                for contour in contours:
                    contour_area = abs(float(cv2.contourArea(contour)))
                    if not image_area * 0.04 <= contour_area <= image_area * 0.8:
                        continue

                    perimeter = cv2.arcLength(contour, True)
                    for approximation_ratio in (0.02, 0.04, 0.06):
                        candidate = cv2.approxPolyDP(
                            contour,
                            approximation_ratio * perimeter,
                            True,
                        )
                        if (
                            len(candidate) != 4
                            or not cv2.isContourConvex(candidate)
                        ):
                            continue

                        corners = self._order_corners(
                            np.asarray(
                                candidate,
                                dtype=np.float32,
                            ).reshape(4, 2)
                        )
                        corners, grid_alignment = self._refine_grid_corners(
                            color_frame,
                            corners,
                        )
                        (
                            corners,
                            score,
                            separation,
                            edge_consistency,
                        ) = self._best_grid_hypothesis(
                            color_frame,
                            corners,
                        )
                        detection = self._detection_from_quadrilateral(
                            corners,
                            score,
                            separation,
                            edge_consistency,
                            grid_alignment,
                            gray.shape,
                        )
                        if detection is not None:
                            detections.append(detection)
                        break
        return detections

    def _detection_from_quadrilateral(
        self,
        corners: FloatArray,
        score: float,
        separation: float,
        edge_consistency: float,
        grid_alignment: float,
        image_shape: tuple[int, int],
    ) -> BoardDetection | None:
        if (
            score < 1.5
            or separation < 15.0
            or edge_consistency < 0.85
        ):
            return None

        model_corners = np.array(
            [[0, 0], [8, 0], [8, 8], [0, 8]],
            dtype=np.float32,
        )
        homography = cv2.getPerspectiveTransform(model_corners, corners)
        grid_model = np.array(
            [(column, row) for row in range(9) for column in range(9)],
            dtype=np.float32,
        )
        grid_points = np.asarray(
            cv2.perspectiveTransform(
                grid_model.reshape(-1, 1, 2),
                homography,
            ),
            dtype=np.float32,
        ).reshape(9, 9, 2)
        if not self._is_valid_grid(grid_points, image_shape):
            return None

        contrast_confidence = min(1.0, separation / 30.0)
        pattern_confidence = min(1.0, (score - 1.5) / 2.0)
        confidence = round(
            0.55 + 0.30 * contrast_confidence + 0.15 * pattern_confidence,
            3,
        )
        return BoardDetection(
            grid_points=grid_points,
            confidence=min(confidence, 0.95),
            reprojection_error=None,
            method="alternating-color quadrilateral",
            grid_alignment=grid_alignment,
        )

    @staticmethod
    def _refine_grid_corners(
        frame: ImageArray,
        corners: FloatArray,
    ) -> tuple[FloatArray, float]:
        target_size = 512
        target_corners = np.array(
            [
                [0, 0],
                [target_size - 1, 0],
                [target_size - 1, target_size - 1],
                [0, target_size - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners, target_corners)
        normalized_board = cv2.warpPerspective(
            frame,
            transform,
            (target_size, target_size),
        )
        gray = cv2.cvtColor(normalized_board, cv2.COLOR_BGR2GRAY)
        vertical_gradient = np.abs(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        )
        horizontal_gradient = np.abs(
            cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        )
        vertical_profile = np.median(
            vertical_gradient[20:-20, :],
            axis=0,
        )
        horizontal_profile = np.median(
            horizontal_gradient[:, 20:-20],
            axis=1,
        )

        def find_grid_lines(
            profile: FloatArray,
        ) -> tuple[list[int], list[float]]:
            def scan(sequential: bool) -> tuple[list[int], list[float]]:
                positions: list[int] = []
                ratios: list[float] = []
                spacing = (target_size - 1) / 8
                search_radius = 24 if sequential else 20
                for grid_index in range(1, 8):
                    expected = round(
                        positions[-1] + spacing
                        if sequential and positions
                        else grid_index * (target_size - 1) / 8
                    )
                    if not 0 <= expected < target_size:
                        return [], []
                    search_start = max(0, expected - search_radius)
                    search_end = min(
                        target_size,
                        expected + search_radius + 1,
                    )
                    window = profile[search_start:search_end]
                    if window.size == 0:
                        return [], []
                    position = search_start + int(np.argmax(window))
                    positions.append(position)
                    ratios.append(
                        float(
                            profile[
                                max(0, expected - 2) : expected + 3
                            ].max()
                            / (window.max() + 1.0)
                        )
                    )
                    if sequential and len(positions) >= 2:
                        spacing = float(
                            np.median(np.diff(np.asarray(positions)))
                        )
                        if not 32.0 <= spacing <= 96.0:
                            return [], []
                return positions, ratios

            def fit_residual(positions: list[int]) -> float:
                indices = np.arange(1, 8, dtype=np.float32)
                slope, offset = np.polyfit(indices, positions, 1)
                return float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.asarray(positions)
                                - (slope * indices + offset)
                            )
                        )
                    )
                )

            nominal = scan(False)
            sequential = scan(True)
            if not sequential[0]:
                return nominal
            return min(
                (nominal, sequential),
                key=lambda candidate: fit_residual(candidate[0]),
            )

        vertical_positions, vertical_ratios = find_grid_lines(
            vertical_profile
        )
        horizontal_positions, horizontal_ratios = find_grid_lines(
            horizontal_profile
        )
        alignment_ratios = vertical_ratios + horizontal_ratios

        grid_indices = np.arange(1, 8, dtype=np.float32)
        x_slope, x_offset = np.polyfit(
            grid_indices,
            vertical_positions,
            1,
        )
        y_slope, y_offset = np.polyfit(
            grid_indices,
            horizontal_positions,
            1,
        )
        x_residual = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(vertical_positions)
                        - (x_slope * grid_indices + x_offset)
                    )
                )
            )
        )
        y_residual = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.asarray(horizontal_positions)
                        - (y_slope * grid_indices + y_offset)
                    )
                )
            )
        )
        alignment = float(np.mean(alignment_ratios))
        if (
            not 44.0 <= x_slope <= 84.0
            or not 44.0 <= y_slope <= 84.0
            or x_residual > 8.0
            or y_residual > 8.0
            or alignment < 0.25
        ):
            return corners, alignment

        x_start = float(x_offset)
        x_end = float(x_offset + 8 * x_slope)
        if abs(x_start) > 16.0:
            x_start = 0.0
            x_end = float(target_size - 1)
        y_start = float(y_offset)
        y_end = float(y_offset + 8 * y_slope)
        if abs(y_start) > 16.0:
            y_start = 0.0
            y_end = float(target_size - 1)

        normalized_corners = np.array(
            [
                [x_start, y_start],
                [x_end, y_start],
                [x_end, y_end],
                [x_start, y_end],
            ],
            dtype=np.float32,
        )
        if (
            np.min(normalized_corners) < -40
            or np.max(normalized_corners) > target_size + 40
        ):
            return corners, alignment

        inverse_transform = np.linalg.inv(transform).astype(np.float32)
        refined_corners = np.asarray(
            cv2.perspectiveTransform(
                normalized_corners.reshape(-1, 1, 2),
                inverse_transform,
            ),
            dtype=np.float32,
        ).reshape(4, 2)
        return refined_corners, alignment

    @staticmethod
    def _detections_overlap(
        first: BoardDetection,
        second: BoardDetection,
    ) -> bool:
        first_corners = np.ascontiguousarray(
            first.outer_corners,
            dtype=np.float32,
        )
        second_corners = np.ascontiguousarray(
            second.outer_corners,
            dtype=np.float32,
        )
        first_area = abs(float(cv2.contourArea(first_corners)))
        second_area = abs(float(cv2.contourArea(second_corners)))
        intersection_area, _intersection = cv2.intersectConvexConvex(
            first_corners,
            second_corners,
        )
        return float(intersection_area) / min(first_area, second_area) > 0.5

    @staticmethod
    def _order_corners(corners: FloatArray) -> FloatArray:
        sorted_by_y = corners[np.argsort(corners[:, 1])]
        top = sorted_by_y[:2]
        bottom = sorted_by_y[2:]
        top = top[np.argsort(top[:, 0])]
        bottom = bottom[np.argsort(bottom[:, 0])]
        return np.array(
            [top[0], top[1], bottom[1], bottom[0]],
            dtype=np.float32,
        )

    @staticmethod
    def _alternating_color_score(
        frame: ImageArray,
        corners: FloatArray,
    ) -> tuple[float, float, float]:
        target_size = 512
        target_corners = np.array(
            [
                [0, 0],
                [target_size - 1, 0],
                [target_size - 1, target_size - 1],
                [0, target_size - 1],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(corners, target_corners)
        normalized_board = cv2.warpPerspective(
            frame,
            transform,
            (target_size, target_size),
        )
        lab_board = cv2.cvtColor(
            normalized_board,
            cv2.COLOR_BGR2LAB,
        ).astype(np.float32)

        cell_means: list[FloatArray] = []
        parities: list[int] = []
        cell_size = target_size / 8
        for row in range(8):
            for column in range(8):
                patches: list[FloatArray] = []
                for y_fraction in (0.12, 0.70):
                    for x_fraction in (0.12, 0.70):
                        y_start = int(
                            row * cell_size + y_fraction * cell_size
                        )
                        y_end = int(
                            row * cell_size
                            + (y_fraction + 0.18) * cell_size
                        )
                        x_start = int(
                            column * cell_size + x_fraction * cell_size
                        )
                        x_end = int(
                            column * cell_size
                            + (x_fraction + 0.18) * cell_size
                        )
                        patches.append(
                            lab_board[y_start:y_end, x_start:x_end].reshape(
                                -1,
                                3,
                            )
                        )
                cell_means.append(
                    np.median(
                        np.concatenate(patches, axis=0),
                        axis=0,
                    )
                )
                parities.append((row + column) % 2)

        means = np.asarray(cell_means, dtype=np.float32)
        parity = np.asarray(parities)
        first_class = means[parity == 0]
        second_class = means[parity == 1]
        first_center = np.median(first_class, axis=0)
        second_center = np.median(second_class, axis=0)
        separation = float(np.linalg.norm(first_center - second_center))
        within_class_spread = float(
            (
                np.median(np.linalg.norm(first_class - first_center, axis=1))
                + np.median(
                    np.linalg.norm(second_class - second_center, axis=1)
                )
            )
            / 2
        )
        distances = np.stack(
            [
                np.linalg.norm(means - first_center, axis=1),
                np.linalg.norm(means - second_center, axis=1),
            ],
            axis=1,
        )
        parity_matches = (
            np.argmin(distances, axis=1) == parity
        ).reshape(8, 8)
        edge_matches = np.concatenate(
            [
                parity_matches[0, :],
                parity_matches[-1, :],
                parity_matches[1:-1, 0],
                parity_matches[1:-1, -1],
            ]
        )
        edge_consistency = float(np.mean(edge_matches))
        return (
            separation / (within_class_spread + 1.0),
            separation,
            edge_consistency,
        )

    @classmethod
    def _best_grid_hypothesis(
        cls,
        frame: ImageArray,
        corners: FloatArray,
    ) -> tuple[FloatArray, float, float, float]:
        target_size = 512
        target_corners = np.asarray(
            [
                [0, 0],
                [target_size - 1, 0],
                [target_size - 1, target_size - 1],
                [0, target_size - 1],
            ],
            dtype=np.float32,
        )
        inverse_transform = np.linalg.inv(
            cv2.getPerspectiveTransform(corners, target_corners)
        ).astype(np.float32)
        score, separation, edge_consistency = cls._alternating_color_score(
            frame,
            corners,
        )
        base_hypothesis = (
            corners,
            score,
            separation,
            edge_consistency,
        )
        if edge_consistency >= 0.9:
            return base_hypothesis

        cell_size = target_size / 8
        hypotheses = [base_hypothesis]
        for row_shift in (-1, 0, 1):
            for column_shift in (-1, 0, 1):
                if row_shift == 0 and column_shift == 0:
                    continue
                left = column_shift * cell_size
                top = row_shift * cell_size
                shifted_corners = np.asarray(
                    [
                        [left, top],
                        [left + target_size - 1, top],
                        [
                            left + target_size - 1,
                            top + target_size - 1,
                        ],
                        [left, top + target_size - 1],
                    ],
                    dtype=np.float32,
                )
                camera_corners = np.asarray(
                    cv2.perspectiveTransform(
                        shifted_corners.reshape(-1, 1, 2),
                        inverse_transform,
                    ),
                    dtype=np.float32,
                ).reshape(4, 2)
                score, separation, edge_consistency = (
                    cls._alternating_color_score(frame, camera_corners)
                )
                hypotheses.append(
                    (
                        camera_corners,
                        score,
                        separation,
                        edge_consistency,
                    )
                )
        return max(
            hypotheses,
            key=lambda hypothesis: (
                hypothesis[3],
                hypothesis[1],
                hypothesis[2],
            ),
        )

    @staticmethod
    def draw_overlay(frame: ImageArray, detection: BoardDetection) -> ImageArray:
        overlay = frame.copy()
        grid = np.rint(detection.grid_points).astype(np.int32)

        for row in range(9):
            row_points = np.ascontiguousarray(
                grid[row].reshape(-1, 1, 2),
                dtype=np.int32,
            )
            cv2.polylines(
                overlay,
                [row_points],
                False,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )
        for column in range(9):
            column_points = np.ascontiguousarray(
                grid[:, column].reshape(-1, 1, 2),
                dtype=np.int32,
            )
            cv2.polylines(
                overlay,
                [column_points],
                False,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        outer_corners = np.ascontiguousarray(
            np.rint(detection.outer_corners).reshape(-1, 1, 2),
            dtype=np.int32,
        )
        cv2.polylines(
            overlay,
            [outer_corners],
            True,
            (0, 255, 255),
            4,
            cv2.LINE_AA,
        )
        return overlay

    @staticmethod
    def _is_valid_grid(
        grid_points: FloatArray,
        image_shape: tuple[int, int],
    ) -> bool:
        height, width = image_shape
        margin = 0.02 * max(width, height)
        x_coordinates = grid_points[:, :, 0]
        y_coordinates = grid_points[:, :, 1]
        if (
            np.min(x_coordinates) < -margin
            or np.max(x_coordinates) > width - 1 + margin
            or np.min(y_coordinates) < -margin
            or np.max(y_coordinates) > height - 1 + margin
        ):
            return False

        outer = grid_points[[0, 0, 8, 8], [0, 8, 8, 0]]
        board_area = abs(float(cv2.contourArea(outer)))
        if board_area < width * height * 0.05:
            return False

        horizontal_steps = np.linalg.norm(np.diff(grid_points, axis=1), axis=2)
        vertical_steps = np.linalg.norm(np.diff(grid_points, axis=0), axis=2)
        all_steps = np.concatenate(
            [horizontal_steps.reshape(-1), vertical_steps.reshape(-1)]
        )
        median_step = float(np.median(all_steps))
        if median_step < 4.0:
            return False

        return bool(
            np.min(all_steps) > median_step * 0.35
            and np.max(all_steps) < median_step * 2.5
        )

    @staticmethod
    def _calculate_confidence(
        grid_points: FloatArray,
        inlier_mask: NDArray[np.uint8],
        reprojection_error: float,
        image_shape: tuple[int, int],
    ) -> float:
        horizontal_steps = np.linalg.norm(np.diff(grid_points, axis=1), axis=2)
        vertical_steps = np.linalg.norm(np.diff(grid_points, axis=0), axis=2)
        square_size = float(
            np.median(
                np.concatenate(
                    [horizontal_steps.reshape(-1), vertical_steps.reshape(-1)]
                )
            )
        )
        accuracy = max(0.0, 1.0 - reprojection_error / max(square_size * 0.2, 1.0))
        inlier_ratio = float(np.mean(inlier_mask))

        height, width = image_shape
        outer = grid_points[[0, 0, 8, 8], [0, 8, 8, 0]]
        area_ratio = abs(float(cv2.contourArea(outer))) / (width * height)
        coverage = min(1.0, area_ratio / 0.2)

        return round(0.55 * accuracy + 0.35 * inlier_ratio + 0.10 * coverage, 3)
