from collections.abc import Collection, Mapping
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSizePolicy, QWidget

from .chess_position import PieceCode, Position, piece_asset_filename

LICHESS_LIGHT = QColor("#f0d9b5")
LICHESS_DARK = QColor("#b58863")
LICHESS_SELECTED = QColor("#d7da4f")
LICHESS_ILLEGAL = QColor("#ef5350")
_LAST_MOVE_LIGHTENING = 135
_PIECE_ASSET_DIRECTORY = (
    Path(__file__).resolve().parent / "assets" / "pieces" / "cburnett"
)
_PIECE_CODES = frozenset("KQRBNPkqrbnp")


def rotate_square(
    row: int,
    column: int,
    clockwise_quarter_turns: int,
) -> tuple[int, int]:
    if not 0 <= row < 8 or not 0 <= column < 8:
        raise ValueError("Square must be inside the 8x8 board.")
    turns = clockwise_quarter_turns % 4
    for _turn in range(turns):
        row, column = column, 7 - row
    return row, column


def square_at_position(
    position: QPointF,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    board_size = min(width, height)
    if board_size <= 0:
        return None
    left = (width - board_size) / 2
    top = (height - board_size) / 2
    local_x = position.x() - left
    local_y = position.y() - top
    if not 0 <= local_x < board_size or not 0 <= local_y < board_size:
        return None
    square_size = board_size / 8
    return int(local_y / square_size), int(local_x / square_size)


class ChessboardWidget(QWidget):
    square_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self._selected_square: tuple[int, int] | None = None
        self._last_move: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._illegal_source: tuple[int, int] | None = None
        self._occupied_squares: frozenset[tuple[int, int]] = frozenset()
        self._pieces: Position = {}
        self._piece_renderers: dict[PieceCode, QSvgRenderer] = {}
        self._piece_pixmaps: dict[tuple[PieceCode, int, float], QPixmap] = {}
        self.setMinimumSize(320, 320)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.setToolTip(
            "Run or load projector calibration, then click a square to "
            "illuminate the matching physical square."
        )

    @property
    def selected_square(self) -> tuple[int, int] | None:
        return self._selected_square

    def set_selected_square(
        self,
        square: tuple[int, int] | None,
    ) -> None:
        if square is not None:
            row, column = square
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError("Selected square must be inside the 8x8 board.")
        self._selected_square = square
        self.update()

    @property
    def last_move(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        return self._last_move

    def set_last_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]] | None,
    ) -> None:
        if move is not None:
            for row, column in move:
                if not 0 <= row < 8 or not 0 <= column < 8:
                    raise ValueError(
                        "Move squares must be inside the 8x8 board."
                    )
        if move == self._last_move:
            return
        self._last_move = move
        self.update()

    @property
    def illegal_source(self) -> tuple[int, int] | None:
        return self._illegal_source

    def set_illegal_source(
        self,
        square: tuple[int, int] | None,
    ) -> None:
        if square is not None:
            row, column = square
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Illegal-move source must be inside the 8x8 board."
                )
        if square == self._illegal_source:
            return
        self._illegal_source = square
        self.update()

    @property
    def occupied_squares(self) -> frozenset[tuple[int, int]]:
        return self._occupied_squares

    def set_occupied_squares(
        self,
        squares: Collection[tuple[int, int]],
    ) -> None:
        occupied = frozenset(squares)
        for row, column in occupied:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Occupied squares must be inside the 8x8 board."
                )
        if occupied == self._occupied_squares:
            return
        self._occupied_squares = occupied
        self.update()

    @property
    def pieces(self) -> Position:
        return self._pieces.copy()

    def set_pieces(
        self,
        pieces: Mapping[tuple[int, int], PieceCode],
    ) -> None:
        position: Position = {}
        for square, piece in pieces.items():
            row, column = square
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError("Piece squares must be inside the 8x8 board.")
            if piece not in _PIECE_CODES:
                raise ValueError(f"Unknown piece code: {piece}")
            position[(row, column)] = piece
        for piece in set(position.values()):
            self._piece_renderer(piece)
        if position == self._pieces:
            return
        self._pieces = position
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if (
            self.isEnabled()
            and event.button() == Qt.MouseButton.LeftButton
        ):
            square = square_at_position(
                event.position(),
                self.width(),
                self.height(),
            )
            if square is not None:
                self.square_clicked.emit(*square)
                event.accept()
                return
        super().mousePressEvent(event)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#262421"))

        board_size = min(self.width(), self.height())
        left = (self.width() - board_size) / 2
        top = (self.height() - board_size) / 2
        square_size = board_size / 8
        for row in range(8):
            for column in range(8):
                rectangle = QRectF(
                    left + column * square_size,
                    top + row * square_size,
                    square_size,
                    square_size,
                )
                color = (
                    LICHESS_LIGHT
                    if (row + column) % 2 == 0
                    else LICHESS_DARK
                )
                painter.fillRect(rectangle, color)
                if (
                    self._last_move is not None
                    and (row, column) in self._last_move
                ):
                    painter.fillRect(
                        rectangle,
                        color.lighter(_LAST_MOVE_LIGHTENING),
                    )
                if self._selected_square == (row, column):
                    painter.fillRect(rectangle, LICHESS_SELECTED)
                    painter.setPen(
                        QPen(QColor("#c62828"), max(2.0, square_size * 0.05))
                    )
                    painter.drawRect(rectangle.adjusted(2, 2, -2, -2))
                if self._illegal_source == (row, column):
                    painter.fillRect(rectangle, LICHESS_ILLEGAL)
                    painter.setPen(
                        QPen(QColor("#b71c1c"), max(2.0, square_size * 0.07))
                    )
                    painter.drawRect(rectangle.adjusted(2, 2, -2, -2))
                piece = self._pieces.get((row, column))
                if (
                    piece is None
                    and (row, column) in self._occupied_squares
                ):
                    diameter = square_size * 0.44
                    center = rectangle.center()
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#111111"))
                    painter.drawEllipse(
                        QRectF(
                            center.x() - diameter / 2,
                            center.y() - diameter / 2,
                            diameter,
                            diameter,
                        )
                    )
                if piece is not None:
                    piece_size = max(1, round(square_size * 0.92))
                    pixmap = self._piece_pixmap(
                        piece,
                        piece_size,
                        self.devicePixelRatioF(),
                    )
                    piece_rectangle = QRectF(
                        rectangle.center().x() - piece_size / 2,
                        rectangle.center().y() - piece_size / 2,
                        piece_size,
                        piece_size,
                    )
                    painter.drawPixmap(
                        piece_rectangle,
                        pixmap,
                        QRectF(pixmap.rect()),
                    )

                label_color = (
                    LICHESS_DARK
                    if color == LICHESS_LIGHT
                    else LICHESS_LIGHT
                )
                painter.setPen(label_color)
                font = painter.font()
                font.setBold(True)
                font.setPixelSize(max(10, round(square_size * 0.18)))
                painter.setFont(font)
                label_margin = max(2.0, square_size * 0.05)
                if column == 7:
                    painter.drawText(
                        rectangle.adjusted(
                            label_margin,
                            label_margin,
                            -label_margin,
                            -label_margin,
                        ),
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignTop,
                        str(8 - row),
                    )
                if row == 7:
                    painter.drawText(
                        rectangle.adjusted(
                            label_margin,
                            label_margin,
                            -label_margin,
                            -label_margin,
                        ),
                        Qt.AlignmentFlag.AlignLeft
                        | Qt.AlignmentFlag.AlignBottom,
                        chr(ord("a") + column),
                    )

        painter.end()
        event.accept()

    def _piece_renderer(self, piece: PieceCode) -> QSvgRenderer:
        renderer = self._piece_renderers.get(piece)
        if renderer is not None:
            return renderer
        asset_path = _PIECE_ASSET_DIRECTORY / piece_asset_filename(piece)
        renderer = QSvgRenderer(str(asset_path), self)
        if not renderer.isValid():
            raise RuntimeError(
                f"Chess piece asset could not be loaded: {asset_path}"
            )
        self._piece_renderers[piece] = renderer
        return renderer

    def _piece_pixmap(
        self,
        piece: PieceCode,
        logical_size: int,
        device_pixel_ratio: float,
    ) -> QPixmap:
        cache_key = (piece, logical_size, device_pixel_ratio)
        cached = self._piece_pixmaps.get(cache_key)
        if cached is not None:
            return cached
        pixel_size = max(1, round(logical_size * device_pixel_ratio))
        pixmap = QPixmap(pixel_size, pixel_size)
        pixmap.fill(Qt.GlobalColor.transparent)
        piece_painter = QPainter(pixmap)
        piece_painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._piece_renderer(piece).render(
            piece_painter,
            QRectF(0, 0, pixel_size, pixel_size),
        )
        piece_painter.end()
        pixmap.setDevicePixelRatio(device_pixel_ratio)
        self._piece_pixmaps[cache_key] = pixmap
        return pixmap
