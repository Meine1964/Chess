from dataclasses import dataclass
from typing import TypeAlias

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


class BoardDetector:
    """Detect an empty 8x8 chessboard and reconstruct its full 9x9 grid."""

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
                    candidate = cv2.approxPolyDP(
                        contour,
                        0.02 * perimeter,
                        True,
                    )
                    if len(candidate) != 4 or not cv2.isContourConvex(candidate):
                        continue

                    corners = self._order_corners(
                        np.asarray(candidate, dtype=np.float32).reshape(4, 2)
                    )
                    corners, grid_alignment = self._refine_grid_corners(
                        color_frame,
                        corners,
                    )
                    score, separation = self._alternating_color_score(
                        color_frame,
                        corners,
                    )
                    detection = self._detection_from_quadrilateral(
                        corners,
                        score,
                        separation,
                        grid_alignment,
                        gray.shape,
                    )
                    if detection is not None:
                        detections.append(detection)
        return detections

    def _detection_from_quadrilateral(
        self,
        corners: FloatArray,
        score: float,
        separation: float,
        grid_alignment: float,
        image_shape: tuple[int, int],
    ) -> BoardDetection | None:
        if score < 1.5 or separation < 15.0:
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

        vertical_positions: list[int] = []
        horizontal_positions: list[int] = []
        alignment_ratios: list[float] = []
        for grid_index in range(1, 8):
            expected = round(grid_index * (target_size - 1) / 8)
            search_start = expected - 20
            search_end = expected + 21
            vertical_window = vertical_profile[search_start:search_end]
            horizontal_window = horizontal_profile[search_start:search_end]
            vertical_position = search_start + int(
                np.argmax(vertical_window)
            )
            horizontal_position = search_start + int(
                np.argmax(horizontal_window)
            )
            vertical_positions.append(vertical_position)
            horizontal_positions.append(horizontal_position)
            alignment_ratios.extend(
                [
                    float(
                        vertical_profile[
                            max(0, expected - 2) : expected + 3
                        ].max()
                        / (vertical_window.max() + 1.0)
                    ),
                    float(
                        horizontal_profile[
                            max(0, expected - 2) : expected + 3
                        ].max()
                        / (horizontal_window.max() + 1.0)
                    ),
                ]
            )

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

        normalized_corners = np.array(
            [
                [x_offset, y_offset],
                [x_offset + 8 * x_slope, y_offset],
                [x_offset + 8 * x_slope, y_offset + 8 * y_slope],
                [x_offset, y_offset + 8 * y_slope],
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
    ) -> tuple[float, float]:
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
        for row in range(8):
            for column in range(8):
                y_start = int((row + 0.25) * target_size / 8)
                y_end = int((row + 0.75) * target_size / 8)
                x_start = int((column + 0.25) * target_size / 8)
                x_end = int((column + 0.75) * target_size / 8)
                cell_means.append(
                    lab_board[y_start:y_end, x_start:x_end].mean(
                        axis=(0, 1)
                    )
                )
                parities.append((row + column) % 2)

        means = np.asarray(cell_means, dtype=np.float32)
        parity = np.asarray(parities)
        first_class = means[parity == 0]
        second_class = means[parity == 1]
        first_center = first_class.mean(axis=0)
        second_center = second_class.mean(axis=0)
        separation = float(np.linalg.norm(first_center - second_center))
        within_class_spread = float(
            (
                np.linalg.norm(first_class - first_center, axis=1).mean()
                + np.linalg.norm(second_class - second_center, axis=1).mean()
            )
            / 2
        )
        return separation / (within_class_spread + 1.0), separation

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
