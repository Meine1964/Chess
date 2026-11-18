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
    reprojection_error: float

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

        found, corners = self._find_inner_corners(gray)
        if not found or corners is None:
            return None

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

        if not self._is_valid_grid(grid_points, gray.shape):
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
            gray.shape,
        )

        return BoardDetection(
            grid_points=grid_points.astype(np.float32),
            confidence=confidence,
            reprojection_error=reprojection_error,
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
