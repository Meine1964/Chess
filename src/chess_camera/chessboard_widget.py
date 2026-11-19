from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from collections.abc import Collection

from PySide6.QtGui import QColor, QMouseEvent, QPaintEvent, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

LICHESS_LIGHT = QColor("#f0d9b5")
LICHESS_DARK = QColor("#b58863")
LICHESS_SELECTED = QColor("#d7da4f")


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
        self._occupied_squares: frozenset[tuple[int, int]] = frozenset()
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
                if self._selected_square == (row, column):
                    painter.fillRect(rectangle, LICHESS_SELECTED)
                    painter.setPen(
                        QPen(QColor("#c62828"), max(2.0, square_size * 0.05))
                    )
                    painter.drawRect(rectangle.adjusted(2, 2, -2, -2))
                if (row, column) in self._occupied_squares:
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
