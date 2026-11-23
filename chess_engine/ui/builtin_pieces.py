"""Procedurally drawn piece set used when no piece images are installed."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

SIZE = 150


def _poly(points: list[tuple[float, float]]) -> QPolygonF:
    return QPolygonF([QPointF(x, y) for x, y in points])


def draw_piece(name: str, is_white: bool) -> QPixmap:
    """Return a flat, original piece drawing on a transparent 150x150 pixmap."""
    pixmap = QPixmap(SIZE, SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(1.5, 1.5)  # draw in a 100x100 space
    fill = QColor("#f4f1ea") if is_white else QColor("#2d2a2a")
    line = QColor("#2d2a2a") if is_white else QColor("#d9d4c7")
    painter.setBrush(QBrush(fill))
    painter.setPen(QPen(line, 2.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

    # shared base
    painter.drawRoundedRect(QRectF(22, 80, 56, 10), 4, 4)
    if name == "pawn":
        painter.drawPolygon(_poly([(36, 80), (42, 52), (58, 52), (64, 80)]))
        painter.drawEllipse(QPointF(50, 38), 13, 13)
    elif name == "rook":
        painter.drawRect(QRectF(31, 42, 38, 38))
        painter.drawRect(QRectF(27, 24, 46, 18))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(line))
        for x in (36, 53):
            painter.drawRect(QRectF(x, 24, 7, 9))
    elif name == "bishop":
        painter.drawPolygon(_poly([(36, 80), (44, 54), (56, 54), (64, 80)]))
        painter.drawEllipse(QPointF(50, 38), 14, 20)
        painter.drawEllipse(QPointF(50, 16), 5, 5)
        painter.drawLine(QPointF(44, 34), QPointF(56, 28))
    elif name == "knight":
        path = QPainterPath()
        path.moveTo(30, 80)
        path.lineTo(36, 50)
        path.cubicTo(30, 38, 40, 22, 56, 20)
        path.lineTo(54, 12)
        path.lineTo(64, 18)
        path.cubicTo(78, 24, 80, 50, 72, 80)
        path.closeSubpath()
        painter.drawPath(path)
        painter.setBrush(QBrush(line))
        painter.drawEllipse(QPointF(56, 30), 2.6, 2.6)
    elif name == "queen":
        painter.drawPolygon(_poly([(28, 80), (22, 34), (38, 52), (50, 26), (62, 52), (78, 34), (72, 80)]))
        for x, y in ((22, 30), (38, 22), (50, 18), (62, 22), (78, 30)):
            painter.drawEllipse(QPointF(x, y), 4.5, 4.5)
    elif name == "king":
        painter.drawPolygon(_poly([(30, 80), (34, 46), (66, 46), (70, 80)]))
        painter.drawRoundedRect(QRectF(34, 36, 32, 14), 5, 5)
        painter.drawLine(QPointF(50, 10), QPointF(50, 34))
        painter.drawLine(QPointF(42, 18), QPointF(58, 18))
    painter.end()
    return pixmap
