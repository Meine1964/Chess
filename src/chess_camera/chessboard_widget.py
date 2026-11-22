from collections.abc import Collection, Mapping
from pathlib import Path

from PySide6.QtCore import QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QPixmap,
)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QSizePolicy, QStyle, QWidget

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
    copy_fen_requested = Signal()
    copy_board_requested = Signal()
    flip_requested = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self._selected_square: tuple[int, int] | None = None
        self._last_move: tuple[tuple[int, int], tuple[int, int]] | None = None
        self._illegal_source: tuple[int, int] | None = None
        self._guidance_squares: tuple[tuple[int, int], ...] = ()
        self._reconciliation_squares: frozenset[
            tuple[int, int]
        ] = frozenset()
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
        self._copy_fen_action = QAction("Copy FEN", self)
        self._copy_fen_action.triggered.connect(
            self.copy_fen_requested.emit
        )
        self._copy_board_action = QAction("Copy Board", self)
        self._copy_board_action.triggered.connect(
            self.copy_board_requested.emit
        )
        self._flipped = False
        self._flip_action = QAction(
            self.style().standardIcon(
                QStyle.StandardPixmap.SP_BrowserReload
            ),
            "Flip board",
            self,
        )
        self._flip_action.setCheckable(True)
        self._flip_action.triggered.connect(self._request_flip)
        self.addActions(
            (
                self._copy_fen_action,
                self._copy_board_action,
                self._flip_action,
            )
        )
        self.setContextMenuPolicy(
            Qt.ContextMenuPolicy.ActionsContextMenu
        )

    @property
    def flipped(self) -> bool:
        return self._flipped

    def _request_flip(self) -> None:
        self.flip_requested.emit(self._flip_action.isChecked())

    def set_flipped(self, flipped: bool) -> None:
        self._flip_action.setChecked(flipped)
        if flipped == self._flipped:
            return
        self._flipped = flipped
        self.update()

    def _board_square(self, view_row: int, view_column: int) -> tuple[int, int]:
        if self._flipped:
            return 7 - view_row, 7 - view_column
        return view_row, view_column

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
    def guidance_move(
        self,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        if len(self._guidance_squares) < 2:
            return None
        return self._guidance_squares[0], self._guidance_squares[1]

    @property
    def guidance_squares(self) -> tuple[tuple[int, int], ...]:
        return self._guidance_squares

    def set_guidance_move(
        self,
        move: tuple[tuple[int, int], tuple[int, int]] | None,
    ) -> None:
        self.set_guidance_squares(() if move is None else move)

    def set_guidance_squares(
        self,
        squares: Collection[tuple[int, int]],
    ) -> None:
        guidance_squares = tuple(dict.fromkeys(squares))
        for row, column in guidance_squares:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Guidance squares must be inside the 8x8 board."
                )
        if guidance_squares == self._guidance_squares:
            return
        self._guidance_squares = guidance_squares
        self.update()

    @property
    def reconciliation_squares(self) -> frozenset[tuple[int, int]]:
        return self._reconciliation_squares

    def set_reconciliation_squares(
        self,
        squares: Collection[tuple[int, int]],
    ) -> None:
        reconciliation_squares = frozenset(squares)
        for row, column in reconciliation_squares:
            if not 0 <= row < 8 or not 0 <= column < 8:
                raise ValueError(
                    "Reconciliation squares must be inside the 8x8 board."
                )
        if reconciliation_squares == self._reconciliation_squares:
            return
        self._reconciliation_squares = reconciliation_squares
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

    def rendered_board_pixmap(self) -> QPixmap:
        pixmap = self.grab()
        board_size = min(pixmap.width(), pixmap.height())
        return pixmap.copy(
            QRect(
                (pixmap.width() - board_size) // 2,
                (pixmap.height() - board_size) // 2,
                board_size,
                board_size,
            )
        )

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
                self.square_clicked.emit(*self._board_square(*square))
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
        for view_row in range(8):
            for view_column in range(8):
                row, column = self._board_square(view_row, view_column)
                rectangle = QRectF(
                    left + view_column * square_size,
                    top + view_row * square_size,
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
                if (
                    (row, column) in self._guidance_squares
                ):
                    painter.fillRect(rectangle, QColor("#ffffff"))
                    painter.setPen(
                        QPen(QColor("#eeeeee"), max(2.0, square_size * 0.05))
                    )
                    painter.drawRect(rectangle.adjusted(2, 2, -2, -2))
                if (
                    self._illegal_source == (row, column)
                    or (row, column) in self._reconciliation_squares
                ):
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
                if view_column == 7:
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
                if view_row == 7:
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
