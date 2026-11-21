from typing import Literal

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QPaintEvent,
    QPainter,
    QPen,
    QPolygonF,
    QScreen,
    QTransform,
)
from PySide6.QtWidgets import QWidget

from .projector_calibration import square_polygons
from .legal_game import PromotionPiece
from .promotion_selection import (
    PROMOTION_CHOICES,
    PromotionEdge,
    promotion_choice_polygons,
    validate_promotion_polygons,
)

DisplayMode = Literal[
    "black",
    "pattern",
    "grid",
    "squares",
    "selected_square",
    "move",
    "illegal",
    "correction",
    "reconciliation",
    "promotion_choices",
    "promotion_confirmation",
]
PROJECTED_ON_LIGHT_SQUARE = QColor("#007fff")
PROJECTED_ON_DARK_SQUARE = QColor("#00ffff")
PROJECTED_LEGAL_MOVE = QColor("#ffffff")
PROJECTED_ILLEGAL_MOVE = QColor("#ff0000")
PROMOTION_GLYPHS: dict[PromotionPiece, str] = {
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
}


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


class ProjectorWindow(QWidget):
    def __init__(self) -> None:
        super().__init__(
            None,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._display_mode: DisplayMode = "black"
        self._pattern: QImage | None = None
        self._projector_grid: NDArray[np.float32] | None = None
        self._selected_square: tuple[int, int] | None = None
        self._selected_square_is_light: bool | None = None
        self._illegal_square: tuple[int, int] | None = None
        self._correction_squares: tuple[
            tuple[int, int],
            tuple[int, int],
        ] | None = None
        self._reconciliation_squares: tuple[tuple[int, int], ...] = ()
        self._move_squares: tuple[tuple[int, int, bool], ...] = ()
        self._promotion_edge: PromotionEdge | None = None
        self._promotion_viewer_edge: PromotionEdge | None = None
        self._promotion_piece: PromotionPiece | None = None
        self._promotion_destination: tuple[int, int] | None = None
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
        self.show_move_squares(
            (
                (*source, source_is_light),
                (*destination, destination_is_light),
            )
        )

    def show_move_squares(
        self,
        squares: tuple[tuple[int, int, bool], ...],
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        if not squares:
            raise ValueError("At least one projected move square is required.")
        for row, column, _is_light in squares:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Projected move squares must be inside the 8x8 board."
                )
        self._move_squares = tuple(dict.fromkeys(squares))
        self._display_mode = "move"
        self.update()

    def show_illegal_square(
        self,
        row: int,
        column: int,
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        if not 0 <= row < 8 or not 0 <= column < 8:
            raise ValueError("Projected square must be inside the 8x8 board.")
        self._illegal_square = (row, column)
        self._display_mode = "illegal"
        self.update()

    def show_correction(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        for row, column in (source, destination):
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Projected correction squares must be inside the board."
                )
        self._correction_squares = (source, destination)
        self._display_mode = "correction"
        self.update()

    def show_reconciliation(
        self,
        squares: tuple[tuple[int, int], ...],
    ) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        if not squares:
            raise ValueError("At least one reconciliation square is required.")
        for row, column in squares:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Projected reconciliation squares must be inside the board."
                )
        self._reconciliation_squares = tuple(dict.fromkeys(squares))
        self._display_mode = "reconciliation"
        self.update()

    def show_promotion_choices(
        self,
        edge: PromotionEdge,
        viewer_edge: PromotionEdge,
    ) -> None:
        polygons = self._promotion_polygons(edge)
        validate_promotion_polygons(
            polygons,
            max(1, self.width()),
            max(1, self.height()),
        )
        self._promotion_edge = edge
        self._promotion_viewer_edge = viewer_edge
        self._promotion_piece = None
        self._promotion_destination = None
        self._display_mode = "promotion_choices"
        self.update()

    def show_promotion_confirmation(
        self,
        edge: PromotionEdge,
        viewer_edge: PromotionEdge,
        piece: PromotionPiece,
        destination: tuple[int, int],
    ) -> None:
        polygons = self._promotion_polygons(edge)
        validate_promotion_polygons(
            polygons,
            max(1, self.width()),
            max(1, self.height()),
        )
        row, column = destination
        if not 0 <= row < 8 or not 0 <= column < 8:
            raise ValueError("Promotion destination must be on the board.")
        self._promotion_edge = edge
        self._promotion_viewer_edge = viewer_edge
        self._promotion_piece = piece
        self._promotion_destination = destination
        self._display_mode = "promotion_confirmation"
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._display_mode == "pattern" and self._pattern is not None:
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
            and self._move_squares
        ):
            polygons = square_polygons(self._projector_grid)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            for move_square in self._move_squares:
                row, column = move_square[:2]
                painter.setBrush(PROJECTED_LEGAL_MOVE)
                painter.drawPolygon(
                    self._polygon(polygons[row * 8 + column])
                )
        elif (
            self._display_mode == "illegal"
            and self._projector_grid is not None
            and self._illegal_square is not None
        ):
            row, column = self._illegal_square
            polygon = square_polygons(self._projector_grid)[row * 8 + column]
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PROJECTED_ILLEGAL_MOVE)
            painter.drawPolygon(self._polygon(polygon))
        elif (
            self._display_mode == "correction"
            and self._projector_grid is not None
            and self._correction_squares is not None
        ):
            source, destination = self._correction_squares
            polygons = square_polygons(self._projector_grid)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PROJECTED_ILLEGAL_MOVE)
            for row, column in (source, destination):
                painter.drawPolygon(
                    self._polygon(polygons[row * 8 + column])
                )
        elif (
            self._display_mode == "reconciliation"
            and self._projector_grid is not None
            and self._reconciliation_squares
        ):
            polygons = square_polygons(self._projector_grid)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PROJECTED_ILLEGAL_MOVE)
            for row, column in self._reconciliation_squares:
                painter.drawPolygon(
                    self._polygon(polygons[row * 8 + column])
                )
        elif (
            self._display_mode
            in ("promotion_choices", "promotion_confirmation")
            and self._projector_grid is not None
            and self._promotion_edge is not None
            and self._promotion_viewer_edge is not None
        ):
            choice_polygons = promotion_choice_polygons(
                self._projector_grid,
                self._promotion_edge,
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            if self._display_mode == "promotion_choices":
                for piece, polygon in zip(
                    PROMOTION_CHOICES,
                    choice_polygons,
                    strict=True,
                ):
                    self._draw_promotion_glyph(
                        painter,
                        polygon,
                        piece,
                        self._promotion_edge,
                        self._promotion_viewer_edge,
                    )
            elif (
                self._promotion_piece is not None
                and self._promotion_destination is not None
            ):
                selected_index = PROMOTION_CHOICES.index(
                    self._promotion_piece
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(PROJECTED_LEGAL_MOVE)
                painter.drawPolygon(
                    self._polygon(choice_polygons[selected_index])
                )
                row, column = self._promotion_destination
                destination = square_polygons(self._projector_grid)[
                    row * 8 + column
                ]
                self._draw_promotion_glyph(
                    painter,
                    destination,
                    self._promotion_piece,
                    self._promotion_edge,
                    self._promotion_viewer_edge,
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

    def _promotion_polygons(
        self,
        edge: PromotionEdge,
    ) -> NDArray[np.float32]:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        return promotion_choice_polygons(self._projector_grid, edge)

    @staticmethod
    def _promotion_rotation(viewer_edge: PromotionEdge) -> float:
        return {
            "top": 180.0,
            "right": -90.0,
            "bottom": 0.0,
            "left": 90.0,
        }[viewer_edge]

    @staticmethod
    def _draw_promotion_glyph(
        painter: QPainter,
        polygon: NDArray[np.float32],
        piece: PromotionPiece,
        edge: PromotionEdge,
        viewer_edge: PromotionEdge,
    ) -> None:
        canonical_size = 256.0
        transform = ProjectorWindow._promotion_glyph_transform(
            polygon,
            edge,
            canonical_size,
        )
        font = QFont(painter.font())
        font.setBold(True)
        font.setPixelSize(round(canonical_size * 0.68))
        painter.save()
        painter.setPen(PROJECTED_LEGAL_MOVE)
        painter.setFont(font)
        painter.setWorldTransform(transform, combine=True)
        painter.translate(canonical_size / 2, canonical_size / 2)
        painter.rotate(ProjectorWindow._promotion_rotation(viewer_edge))
        painter.drawText(
            QRectF(
                -canonical_size / 2,
                -canonical_size / 2,
                canonical_size,
                canonical_size,
            ),
            Qt.AlignmentFlag.AlignCenter,
            PROMOTION_GLYPHS[piece],
        )
        painter.restore()

    @staticmethod
    def _promotion_glyph_transform(
        polygon: NDArray[np.float32],
        edge: PromotionEdge,
        canonical_size: float = 256.0,
    ) -> QTransform:
        source = QPolygonF(
            [
                QPointF(0.0, 0.0),
                QPointF(canonical_size, 0.0),
                QPointF(canonical_size, canonical_size),
                QPointF(0.0, canonical_size),
            ]
        )
        ordered_polygon = (
            polygon[[0, 3, 2, 1]]
            if edge in ("left", "right")
            else polygon
        )
        return QTransform.quadToQuad(
            source,
            ProjectorWindow._polygon(ordered_polygon),
        )
