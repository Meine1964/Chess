from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtGui import (
    QColor,
    QImage,
    QPaintEvent,
    QPainter,
    QPen,
    QPolygonF,
    QScreen,
)
from PySide6.QtWidgets import QWidget

from .projector_calibration import square_polygons

DisplayMode = Literal[
    "black",
    "target",
    "pattern",
    "grid",
    "squares",
    "selected_square",
    "move",
]
PROJECTED_ON_LIGHT_SQUARE = QColor("#007fff")
PROJECTED_ON_DARK_SQUARE = QColor("#00ffff")


def focus_pattern(
    width: int,
    height: int,
    *,
    cell_size: int = 16,
) -> NDArray[np.uint8]:
    if width <= 0 or height <= 0:
        raise ValueError("Focus-pattern dimensions must be positive.")
    if cell_size < 2:
        raise ValueError("Focus-pattern cell size must be at least two pixels.")

    y_coordinates, x_coordinates = np.indices((height, width))
    checkerboard = (
        (x_coordinates // cell_size + y_coordinates // cell_size) % 2
    )
    pattern = (checkerboard * 255).astype(np.uint8)

    center_x = width // 2
    center_y = height // 2
    pattern[:, max(0, center_x - 4) : min(width, center_x + 5)] = 0
    pattern[:, max(0, center_x - 1) : min(width, center_x + 2)] = 255
    pattern[max(0, center_y - 4) : min(height, center_y + 5), :] = 0
    pattern[max(0, center_y - 1) : min(height, center_y + 2), :] = 255

    border = min(3, width, height)
    pattern[:border, :] = 255
    pattern[-border:, :] = 255
    pattern[:, :border] = 255
    pattern[:, -border:] = 255
    return pattern


def target_rectangle(
    width: int,
    height: int,
    center_x_percent: int,
    center_y_percent: int,
    size_percent: int,
) -> QRect:
    if width <= 0 or height <= 0:
        raise ValueError("Projector dimensions must be positive.")
    if not 0 <= center_x_percent <= 100:
        raise ValueError("center_x_percent must be between 0 and 100.")
    if not 0 <= center_y_percent <= 100:
        raise ValueError("center_y_percent must be between 0 and 100.")
    if not 1 <= size_percent <= 100:
        raise ValueError("size_percent must be between 1 and 100.")

    side = max(1, round(min(width, height) * size_percent / 100))
    center_x = round(width * center_x_percent / 100)
    center_y = round(height * center_y_percent / 100)
    left = min(max(center_x - side // 2, 0), width - side)
    top = min(max(center_y - side // 2, 0), height - side)
    return QRect(left, top, side, side)


class ProjectorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.FramelessWindowHint)
        self._display_mode: DisplayMode = "black"
        self._center_x_percent = 50
        self._center_y_percent = 50
        self._size_percent = 10
        self._target_color = QColor(255, 255, 255)
        self._pattern: QImage | None = None
        self._projector_grid: NDArray[np.float32] | None = None
        self._selected_square: tuple[int, int] | None = None
        self._selected_square_is_light: bool | None = None
        self._move_squares: tuple[
            tuple[int, int, bool],
            tuple[int, int, bool],
        ] | None = None
        self.setWindowTitle("Chess Projector Output")
        self.setCursor(Qt.CursorShape.BlankCursor)

    @property
    def output_mode(self) -> DisplayMode:
        return self._display_mode

    def show_on_screen(self, screen: QScreen) -> None:
        self.winId()
        window_handle = self.windowHandle()
        if window_handle is None:
            raise RuntimeError("Could not create the projector window.")
        window_handle.setScreen(screen)
        self.setGeometry(screen.geometry())
        self.show()
        self.setGeometry(screen.geometry())
        actual_screen = self.screen()
        if actual_screen is not screen:
            self.close()
            raise RuntimeError(
                f'Projector window opened on "{actual_screen.name()}" instead '
                f'of selected display "{screen.name()}".'
            )
        self.raise_()
        self.activateWindow()

    def set_target(
        self,
        center_x_percent: int,
        center_y_percent: int,
        size_percent: int,
    ) -> None:
        self._center_x_percent = center_x_percent
        self._center_y_percent = center_y_percent
        self._size_percent = size_percent
        self.update()

    def show_target(self) -> None:
        self._display_mode = "target"
        self.update()

    def show_black(self) -> None:
        self._display_mode = "black"
        self.update()

    def show_pattern(self, pattern: NDArray[np.uint8]) -> None:
        if pattern.ndim != 2:
            raise ValueError("Projector patterns must be grayscale images.")
        contiguous_pattern = np.ascontiguousarray(pattern, dtype=np.uint8)
        height, width = contiguous_pattern.shape
        self._pattern = QImage(
            contiguous_pattern.data,
            width,
            height,
            contiguous_pattern.strides[0],
            QImage.Format.Format_Grayscale8,
        ).copy()
        self._display_mode = "pattern"
        self.update()

    def set_projector_grid(self, grid: NDArray[np.float32]) -> None:
        if grid.shape != (9, 9, 2):
            raise ValueError("The projector grid must have shape (9, 9, 2).")
        self._projector_grid = np.asarray(grid, dtype=np.float32).copy()

    def clear_projector_grid(self) -> None:
        self._projector_grid = None
        self._selected_square = None
        self.show_black()

    def show_grid(self) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        self._display_mode = "grid"
        self.update()

    def show_all_squares(self) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        self._display_mode = "squares"
        self.update()

    def show_square(
        self,
        row: int,
        column: int,
        *,
        light_square: bool | None = None,
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        if not 0 <= row < 8 or not 0 <= column < 8:
            raise ValueError("Projected square must be inside the 8x8 board.")
        self._selected_square = (row, column)
        self._selected_square_is_light = light_square
        self._display_mode = "selected_square"
        self.update()

    def show_move(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
        *,
        source_is_light: bool,
        destination_is_light: bool,
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        for row, column in (source, destination):
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Projected move squares must be inside the 8x8 board."
                )
        self._move_squares = (
            (*source, source_is_light),
            (*destination, destination_is_light),
        )
        self._display_mode = "move"
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._display_mode == "target":
            painter.fillRect(
                target_rectangle(
                    self.width(),
                    self.height(),
                    self._center_x_percent,
                    self._center_y_percent,
                    self._size_percent,
                ),
                self._target_color,
            )
        elif self._display_mode == "pattern" and self._pattern is not None:
            painter.drawImage(self.rect(), self._pattern)
        elif self._display_mode == "grid" and self._projector_grid is not None:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            for row in range(9):
                painter.drawPolyline(self._polygon(self._projector_grid[row]))
            for column in range(9):
                painter.drawPolyline(
                    self._polygon(self._projector_grid[:, column])
                )
        elif (
            self._display_mode == "squares"
            and self._projector_grid is not None
        ):
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(255, 0, 0))
            for polygon in square_polygons(self._projector_grid):
                painter.drawPolygon(self._polygon(polygon))
        elif (
            self._display_mode == "selected_square"
            and self._projector_grid is not None
            and self._selected_square is not None
        ):
            row, column = self._selected_square
            polygon = square_polygons(self._projector_grid)[row * 8 + column]
            color = (
                PROJECTED_ON_LIGHT_SQUARE
                if (
                    self._selected_square_is_light
                    if self._selected_square_is_light is not None
                    else (row + column) % 2 == 0
                )
                else PROJECTED_ON_DARK_SQUARE
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawPolygon(self._polygon(polygon))
        elif (
            self._display_mode == "move"
            and self._projector_grid is not None
            and self._move_squares is not None
        ):
            polygons = square_polygons(self._projector_grid)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            for row, column, is_light in self._move_squares:
                painter.setBrush(
                    PROJECTED_ON_LIGHT_SQUARE
                    if is_light
                    else PROJECTED_ON_DARK_SQUARE
                )
                painter.drawPolygon(
                    self._polygon(polygons[row * 8 + column])
                )
        painter.end()
        event.accept()

    @staticmethod
    def _polygon(points: NDArray[np.float32]) -> QPolygonF:
        return QPolygonF(
            [
                QPointF(float(point[0]), float(point[1]))
                for point in points
            ]
        )
