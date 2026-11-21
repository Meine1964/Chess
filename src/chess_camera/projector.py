from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
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

from .projector_calibration import (
    board_circle_polygon,
    board_quadrilateral_polygon,
    square_polygons,
)
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
TurnIndicatorCorner = Literal[
    "top-left",
    "top-right",
    "bottom-right",
    "bottom-left",
]
ProjectedResult = Literal["1-0", "0-1", "1/2-1/2"]
PROJECTED_ON_LIGHT_SQUARE = QColor("#007fff")
PROJECTED_ON_DARK_SQUARE = QColor("#00ffff")
PROJECTED_LEGAL_MOVE = QColor("#ffffff")
PROJECTED_ILLEGAL_MOVE = QColor("#ff0000")
_MAX_EDGE_STROKE_COMPENSATION = 1.5
PROMOTION_GLYPHS: dict[PromotionPiece, str] = {
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
}
_SEVEN_SEGMENT_DIGITS: dict[str, frozenset[str]] = {
    "-": frozenset("g"),
    "0": frozenset("abcdef"),
    "1": frozenset("bc"),
    "2": frozenset("abdeg"),
    "3": frozenset("abcdg"),
    "4": frozenset("bcfg"),
    "5": frozenset("acdfg"),
    "6": frozenset("acdefg"),
    "7": frozenset("abc"),
    "8": frozenset("abcdefg"),
    "9": frozenset("abcdfg"),
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
    projection_changed = Signal()

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
        self._show_turn_indicator_with_move = False
        self._promotion_edge: PromotionEdge | None = None
        self._promotion_viewer_edge: PromotionEdge | None = None
        self._promotion_piece: PromotionPiece | None = None
        self._promotion_destination: tuple[int, int] | None = None
        self._turn_indicator_corner: TurnIndicatorCorner | None = None
        self._white_clock_text: str | None = None
        self._black_clock_text: str | None = None
        self._result_text: ProjectedResult | None = None
        self._expired_player: Literal["white", "black"] | None = None
        self._clock_board_quarter_turns = 0
        self._edge_activity_indicator_active = False
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
        self._show_mode("black")

    def set_edge_activity_indicator(self, active: bool) -> None:
        if self._edge_activity_indicator_active == active:
            return
        self._edge_activity_indicator_active = active
        self.update()

    def set_turn_indicator(
        self,
        corner: TurnIndicatorCorner | None,
    ) -> None:
        self._turn_indicator_corner = corner
        self.update()

    def set_edge_clocks(
        self,
        white_clock: str | None,
        black_clock: str | None,
        *,
        board_quarter_turns: int,
        result: ProjectedResult | None = None,
        expired_player: Literal["white", "black"] | None = None,
    ) -> None:
        if (white_clock is None) != (black_clock is None):
            raise ValueError("Both projected clocks must be set or cleared.")
        for clock in (white_clock, black_clock):
            if clock is not None and (
                clock.count(":") != 1
                or any(character not in "0123456789:" for character in clock)
            ):
                raise ValueError("Projected clocks must use digits and a colon.")
        if result not in (None, "1-0", "0-1", "1/2-1/2"):
            raise ValueError(
                "The projected result must be 1-0, 0-1, or 1/2-1/2."
            )
        if expired_player is not None and result is None:
            raise ValueError("An expiration flag requires projected results.")
        self._white_clock_text = white_clock
        self._black_clock_text = black_clock
        self._result_text = result
        self._expired_player = expired_player
        self._clock_board_quarter_turns = board_quarter_turns % 4
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
        self._show_mode("pattern")

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
        self._show_mode("grid")

    def show_all_squares(self) -> None:
        if self._projector_grid is None:
            raise RuntimeError("No calibrated projector grid is available.")
        self._show_mode("squares")

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
        self._show_mode("selected_square")

    def show_move(
        self,
        source: tuple[int, int],
        destination: tuple[int, int],
        *,
        source_is_light: bool,
        destination_is_light: bool,
        include_turn_indicator: bool = False,
    ) -> None:
        self.show_move_squares(
            (
                (*source, source_is_light),
                (*destination, destination_is_light),
            ),
            include_turn_indicator=include_turn_indicator,
        )

    def show_move_squares(
        self,
        squares: tuple[tuple[int, int, bool], ...],
        *,
        include_turn_indicator: bool = False,
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
        self._show_turn_indicator_with_move = include_turn_indicator
        self._show_mode("move")

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
        self._show_mode("illegal")

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
        self._show_mode("correction")

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
        self._show_mode("reconciliation")

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
        self._show_mode("promotion_choices")

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
        self._show_mode("promotion_confirmation")

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
                for index, (piece, polygon) in enumerate(
                    zip(
                        PROMOTION_CHOICES,
                        choice_polygons,
                        strict=True,
                    )
                ):
                    if index == selected_index:
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(PROJECTED_LEGAL_MOVE)
                        painter.drawPolygon(self._polygon(polygon))
                    else:
                        self._draw_promotion_glyph(
                            painter,
                            polygon,
                            piece,
                            self._promotion_edge,
                            self._promotion_viewer_edge,
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
        if (
            self._display_mode
            in (
                "black",
                "move",
                "illegal",
                "correction",
                "reconciliation",
                "promotion_choices",
                "promotion_confirmation",
            )
            and self._projector_grid is not None
            and (
                (
                    self._white_clock_text is not None
                    and self._black_clock_text is not None
                )
                or self._result_text is not None
            )
        ):
            white_clock_polygon = self._edge_clock_polygon(
                self._projector_grid,
                6,
                8,
                board_quarter_turns=self._clock_board_quarter_turns,
            )
            black_clock_polygon = self._edge_clock_polygon(
                self._projector_grid,
                0,
                2,
                board_quarter_turns=self._clock_board_quarter_turns,
            )
            result_polygon = self._edge_result_polygon(
                self._projector_grid,
                board_quarter_turns=self._clock_board_quarter_turns,
            )
            stroke_scales = self._edge_stroke_compensation(
                [
                    white_clock_polygon,
                    black_clock_polygon,
                    result_polygon,
                ]
            )
            if (
                self._white_clock_text is not None
                and self._black_clock_text is not None
            ):
                self._draw_edge_clock(
                    painter,
                    white_clock_polygon,
                    self._white_clock_text,
                    stroke_scale=stroke_scales[0],
                )
                self._draw_edge_clock(
                    painter,
                    black_clock_polygon,
                    self._black_clock_text,
                    stroke_scale=stroke_scales[1],
                )
            if self._result_text is not None:
                self._draw_edge_result(
                    painter,
                    result_polygon,
                    self._result_text,
                    stroke_scale=stroke_scales[2],
                )
                if self._expired_player is not None:
                    start_row, end_row = (
                        (5, 6)
                        if self._expired_player == "white"
                        else (2, 3)
                    )
                    self._draw_expiration_flag(
                        painter,
                        self._edge_clock_polygon(
                            self._projector_grid,
                            start_row,
                            end_row,
                            board_quarter_turns=(
                                self._clock_board_quarter_turns
                            ),
                        ),
                    )
        if (
            (
                self._display_mode == "black"
                or (
                    self._display_mode == "move"
                    and self._show_turn_indicator_with_move
                )
            )
            and self._projector_grid is not None
            and self._turn_indicator_corner is not None
            and self._result_text is None
        ):
            polygon = self._turn_indicator_polygon(
                self._projector_grid,
                self._turn_indicator_corner,
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PROJECTED_LEGAL_MOVE)
            painter.drawPolygon(self._polygon(polygon))
        if (
            self._edge_activity_indicator_active
            and self._projector_grid is not None
        ):
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(PROJECTED_LEGAL_MOVE)
            painter.drawPolygon(
                self._polygon(
                    self._edge_activity_indicator_polygon(
                        self._projector_grid
                    )
                )
            )
        painter.end()
        event.accept()

    def _show_mode(self, mode: DisplayMode) -> None:
        self._display_mode = mode
        self.projection_changed.emit()
        self.update()

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
    def _turn_indicator_polygon(
        grid: NDArray[np.float32],
        corner: TurnIndicatorCorner,
    ) -> NDArray[np.float32]:
        centers = {
            "top-left": (-0.5, -0.5),
            "top-right": (8.5, -0.5),
            "bottom-right": (8.5, 8.5),
            "bottom-left": (-0.5, 8.5),
        }
        x, y = centers[corner]
        return board_circle_polygon(
            grid,
            x,
            y,
            0.14,
        )

    @staticmethod
    def _edge_activity_indicator_polygon(
        grid: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        return board_circle_polygon(
            grid,
            8.5,
            4.5,
            0.08,
        )

    @staticmethod
    def _edge_clock_polygon(
        grid: NDArray[np.float32],
        start_row: int,
        end_row: int,
        *,
        board_quarter_turns: int,
    ) -> NDArray[np.float32]:
        if grid.shape != (9, 9, 2):
            raise ValueError("The projector grid must have shape (9, 9, 2).")
        if not 0 <= start_row < end_row <= 8:
            raise ValueError("Clock rows must define a non-empty board edge.")

        def grid_point(row: int, column: int) -> NDArray[np.float32]:
            for _ in range(board_quarter_turns % 4):
                row, column = column, 8 - row
            return grid[row, column]

        start_boundary = grid_point(start_row, 0)
        end_boundary = grid_point(end_row, 0)
        start_inward = grid_point(start_row, 1)
        end_inward = grid_point(end_row, 1)
        end_outward = end_boundary + end_boundary - end_inward
        start_outward = start_boundary + start_boundary - start_inward
        return np.asarray(
            (
                start_boundary,
                end_boundary,
                end_outward,
                start_outward,
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _edge_result_polygon(
        grid: NDArray[np.float32],
        *,
        board_quarter_turns: int,
    ) -> NDArray[np.float32]:
        logical_points = (
            (3.4, 0.0),
            (4.6, 0.0),
            (4.6, -1.0),
            (3.4, -1.0),
        )
        rotated_points: list[tuple[float, float]] = []
        for row, column in logical_points:
            for _ in range(board_quarter_turns % 4):
                row, column = column, 8.0 - row
            rotated_points.append((column, row))
        return board_quadrilateral_polygon(
            grid,
            (
                rotated_points[0],
                rotated_points[1],
                rotated_points[2],
                rotated_points[3],
            ),
        )

    @staticmethod
    def _draw_edge_clock(
        painter: QPainter,
        polygon: NDArray[np.float32],
        text: str,
        *,
        stroke_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        canonical_width = 256.0
        canonical_height = 128.0
        transform = ProjectorWindow._edge_clock_paint_transform(
            polygon,
            canonical_width,
            canonical_height,
        )
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROJECTED_LEGAL_MOVE)
        painter.setWorldTransform(transform, combine=True)
        margin = 8.0
        cell_width = (canonical_width - margin * 2) / len(text)
        for index, character in enumerate(text):
            left = margin + index * cell_width
            if character == ":":
                radius = min(cell_width, canonical_height) * 0.1
                center_x = left + cell_width / 2
                for center_y in (canonical_height * 0.38, canonical_height * 0.68):
                    painter.drawEllipse(
                        QPointF(center_x, center_y),
                        radius * stroke_scale[0],
                        radius * stroke_scale[1],
                    )
                continue
            ProjectorWindow._draw_seven_segment_digit(
                painter,
                QRectF(
                    left + cell_width * 0.08,
                    canonical_height * 0.08,
                    cell_width * 0.84,
                    canonical_height * 0.84,
                ),
                character,
                thickness_scale=stroke_scale,
            )
        painter.restore()

    @staticmethod
    def _draw_edge_result(
        painter: QPainter,
        polygon: NDArray[np.float32],
        result: ProjectedResult,
        *,
        stroke_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        if result != "1/2-1/2":
            ProjectorWindow._draw_edge_clock(
                painter,
                polygon,
                result,
                stroke_scale=stroke_scale,
            )
            return
        canonical_width = 256.0
        canonical_height = 128.0
        transform = ProjectorWindow._edge_clock_paint_transform(
            polygon,
            canonical_width,
            canonical_height,
        )
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROJECTED_LEGAL_MOVE)
        painter.setWorldTransform(transform, combine=True)
        margin = 8.0
        cell_width = (canonical_width - margin * 2) / 3
        for index in (0, 2):
            left = margin + index * cell_width
            digit_left = left + cell_width * 0.28
            digit_width = cell_width * 0.44
            numerator_width = 14.0
            numerator_left = (
                left + cell_width / 2 - numerator_width + 3.5
            )
            ProjectorWindow._draw_seven_segment_digit(
                painter,
                QRectF(numerator_left, 8.0, numerator_width, 45.0),
                "1",
                thickness=7.0,
                thickness_scale=stroke_scale,
            )
            fraction_bar_height = 6.0 * stroke_scale[1]
            painter.drawRect(
                QRectF(
                    left + cell_width * 0.16,
                    64.0 - fraction_bar_height / 2,
                    cell_width * 0.68,
                    fraction_bar_height,
                )
            )
            ProjectorWindow._draw_seven_segment_digit(
                painter,
                QRectF(digit_left, 75.0, digit_width, 45.0),
                "2",
                thickness=7.0,
                thickness_scale=stroke_scale,
            )
        dash_left = margin + cell_width
        ProjectorWindow._draw_seven_segment_digit(
            painter,
            QRectF(
                dash_left + cell_width * 0.08,
                canonical_height * 0.08,
                cell_width * 0.84,
                canonical_height * 0.84,
            ),
            "-",
            thickness_scale=stroke_scale,
        )
        painter.restore()

    @staticmethod
    def _edge_stroke_compensation(
        polygons: list[NDArray[np.float32]],
        *,
        canonical_width: float = 256.0,
        canonical_height: float = 128.0,
    ) -> list[tuple[float, float]]:
        if not polygons:
            raise ValueError("At least one edge polygon is required.")
        densities: list[tuple[float, float]] = []
        for polygon in polygons:
            if polygon.shape != (4, 2):
                raise ValueError("Each edge polygon must have shape (4, 2).")
            horizontal_density = (
                float(np.linalg.norm(polygon[1] - polygon[0]))
                + float(np.linalg.norm(polygon[2] - polygon[3]))
            ) / (2.0 * canonical_width)
            vertical_density = (
                float(np.linalg.norm(polygon[3] - polygon[0]))
                + float(np.linalg.norm(polygon[2] - polygon[1]))
            ) / (2.0 * canonical_height)
            if horizontal_density <= 0.0 or vertical_density <= 0.0:
                raise ValueError("Edge polygons must have positive area.")
            densities.append((horizontal_density, vertical_density))
        reference_horizontal = max(density[0] for density in densities)
        reference_vertical = max(density[1] for density in densities)
        return [
            (
                min(
                    reference_horizontal / horizontal_density,
                    _MAX_EDGE_STROKE_COMPENSATION,
                ),
                min(
                    reference_vertical / vertical_density,
                    _MAX_EDGE_STROKE_COMPENSATION,
                ),
            )
            for horizontal_density, vertical_density in densities
        ]

    @staticmethod
    def _draw_expiration_flag(
        painter: QPainter,
        polygon: NDArray[np.float32],
    ) -> None:
        canonical_size = 128.0
        transform = ProjectorWindow._edge_clock_paint_transform(
            polygon,
            canonical_size,
            canonical_size,
        )
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(PROJECTED_LEGAL_MOVE)
        painter.setWorldTransform(transform, combine=True)
        painter.drawRect(QRectF(30.0, 18.0, 10.0, 92.0))
        painter.drawPolygon(
            QPolygonF(
                [
                    QPointF(40.0, 18.0),
                    QPointF(102.0, 38.0),
                    QPointF(40.0, 60.0),
                ]
            )
        )
        painter.restore()

    @staticmethod
    def _edge_clock_paint_transform(
        polygon: NDArray[np.float32],
        width: float,
        height: float,
    ) -> QTransform:
        source = QPolygonF(
            [
                QPointF(0.0, 0.0),
                QPointF(width, 0.0),
                QPointF(width, height),
                QPointF(0.0, height),
            ]
        )
        perspective = cast(
            QTransform,
            QTransform.quadToQuad(
                source,
                ProjectorWindow._polygon(polygon),
            ),
        )
        return (
            ProjectorWindow._edge_clock_content_transform(width, height)
            * perspective
        )

    @staticmethod
    def _edge_clock_content_transform(
        width: float,
        height: float,
    ) -> QTransform:
        transform = QTransform()
        transform.translate(width / 2, height / 2)
        transform.rotate(180.0)
        transform.scale(1.0, 0.7)
        transform.translate(-width / 2, -height / 2)
        return transform

    @staticmethod
    def _draw_seven_segment_digit(
        painter: QPainter,
        bounds: QRectF,
        digit: str,
        *,
        thickness: float | None = None,
        thickness_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        segments = _SEVEN_SEGMENT_DIGITS[digit]
        width = bounds.width()
        height = bounds.height()
        if thickness is None:
            thickness = min(width * 0.22, height * 0.09)
        vertical_thickness = thickness * thickness_scale[0]
        horizontal_thickness = thickness * thickness_scale[1]
        horizontal_width = width - vertical_thickness * 2
        vertical_height = height / 2 - horizontal_thickness * 1.5
        rectangles = {
            "a": QRectF(
                bounds.left() + vertical_thickness,
                bounds.top(),
                horizontal_width,
                horizontal_thickness,
            ),
            "b": QRectF(
                bounds.right() - vertical_thickness,
                bounds.top() + horizontal_thickness,
                vertical_thickness,
                vertical_height,
            ),
            "c": QRectF(
                bounds.right() - vertical_thickness,
                bounds.center().y() + horizontal_thickness / 2,
                vertical_thickness,
                vertical_height,
            ),
            "d": QRectF(
                bounds.left() + vertical_thickness,
                bounds.bottom() - horizontal_thickness,
                horizontal_width,
                horizontal_thickness,
            ),
            "e": QRectF(
                bounds.left(),
                bounds.center().y() + horizontal_thickness / 2,
                vertical_thickness,
                vertical_height,
            ),
            "f": QRectF(
                bounds.left(),
                bounds.top() + horizontal_thickness,
                vertical_thickness,
                vertical_height,
            ),
            "g": QRectF(
                bounds.left() + vertical_thickness,
                bounds.center().y() - horizontal_thickness / 2,
                horizontal_width,
                horizontal_thickness,
            ),
        }
        corner_radius = min(vertical_thickness, horizontal_thickness) * 0.35
        for segment in segments:
            painter.drawRoundedRect(
                rectangles[segment],
                corner_radius,
                corner_radius,
            )

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
