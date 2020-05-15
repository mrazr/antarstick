from typing import Union

from PyQt5.Qt import (QGraphicsItem, QGraphicsObject, QGraphicsSceneHoverEvent,
                      QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                      pyqtSignal)
from PyQt5.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF,
                          QTimerEvent, pyqtProperty)
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap

IDLE_COLORS = {
    "gray": QColor(50, 50, 50, 100),
    "red": QColor(255, 0, 0, 200),
    "green": QColor(0, 255, 0, 200),
}

HOVER_COLORS = {
    "gray": QColor(255, 125, 0, 150),
    "red": IDLE_COLORS["red"].lighter(120),
    "green": IDLE_COLORS["green"].lighter(120),
}

PRESS_COLORS = {
    "gray": HOVER_COLORS["gray"].darker(120),
    "red": IDLE_COLORS["red"].darker(120),
    "green": IDLE_COLORS["green"].darker(120),
}


class CheckbuttonLogic:

    def __init__(self):
        self.down = False

    def idle_color(self) -> QColor:
        if self.down:
            return IDLE_COLORS["green"]
        return IDLE_COLORS["gray"]

    def hover_left_color(self) -> QColor:
        return self.idle_color()

    def hover_enter_color(self) -> QColor:
        if self.down:
            return HOVER_COLORS["green"]
        return HOVER_COLORS["gray"]

    def press_color(self) -> QColor:
        if self.down:
            return PRESS_COLORS["green"]
        return PRESS_COLORS["gray"]

    def release_color(self) -> QColor:
        if self.down:
            return HOVER_COLORS["green"]
        return HOVER_COLORS["gray"]

    def is_down(self) -> bool:
        return self.down

    def do_click(self):
        self.down = not self.down


class PushbuttonLogic:

    def __init__(self, color: str):
        self.color = color.lower()

    def idle_color(self) -> QColor:
        return IDLE_COLORS[self.color]

    def hover_left_color(self) -> QColor:
        return self.idle_color()

    def hover_enter_color(self) -> QColor:
        return HOVER_COLORS[self.color]

    def press_color(self) -> QColor:
        return PRESS_COLORS[self.color]

    def release_color(self) -> QColor:
        return HOVER_COLORS[self.color]

    def is_down(self) -> bool:
        return False

    def do_click(self):
        pass


class Button(QGraphicsObject):
    font: QFont = QFont("monospace", 16)

    clicked = pyqtSignal('PyQt_PyObject')

    def __init__(self, btn_id: str, label: str, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.label = QGraphicsSimpleTextItem(label, self)
        self.label.setFont(Button.font)
        self.label.setBrush(QBrush(QColor(255, 255, 255, 255)))
        self.btn_id = btn_id
        self.rect = QRectF(0, 0, 0, 0)

        self.base_color = "gray"

        self.hor_margin = 10
        self.ver_margin = 5
        self.current_timer = 0

        self.logic: Union[CheckbuttonLogic, PushbuttonLogic] = PushbuttonLogic("gray")
        self.fill_color_current = self.logic.idle_color()

        self.color_animation = QPropertyAnimation(self, b"current_fill_color")
        self.color_animation.setEasingCurve(QEasingCurve.Linear)

        self.hovered = False
        self.setAcceptHoverEvents(True)
        self.setZValue(4)
        self.set_height(30)

        self.pixmap: QPixmap = None
        self.max_pixmap_height = 128

    def set_height(self, h: int):
        self.prepareGeometryChange()
        self.rect.setHeight(h)
        self.ver_margin = int(0.25 * h)
        font: QFont = self.label.font()
        font.setPixelSize(h - 2 * self.ver_margin)
        self.label.setFont(font)
        self.rect.setWidth(self.label.boundingRect().width() + 2 * self.hor_margin)
        self._reposition_text()

    def set_width(self, w: int):
        self.rect.setWidth(w)
        self.hor_margin = self.ver_margin
        self._reposition_text()

    def scale_button(self, factor: float):
        self.rect.setHeight(int(factor * self.rect.height()))
        self.rect.setWidth(int(factor * self.rect.width()))

    def _reposition_text(self):
        x = self.rect.width() / 2 - self.label.boundingRect().width() / 2
        y = self.rect.height() / 2 - self.label.boundingRect().height() / 2
        self.label.setPos(QPointF(x, y))

    def boundingRect(self):
        return self.rect

    def paint(self, painter: QPainter, options, widget=None):
        painter.setBrush(QBrush(self.fill_color_current))
        painter.setPen(QPen(QColor(0, 0, 0, 0)))
        if self.pixmap is not None:
            painter.drawPixmap(self.hor_margin, self.ver_margin, self.pixmap)
        painter.drawRoundedRect(self.rect, 5, 5)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        self.fill_color_current = self.logic.press_color()
        self.scene().update()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        self.logic.do_click()
        self.fill_color_current = self.logic.release_color()
        if self.scene() is not None:
            self.scene().update(self.sceneBoundingRect())
        self.clicked.emit({"btn_id": self.btn_id, "btn_label": self.label})

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        self.hovered = True
        self.color_animation.setDuration(200)
        self.color_animation.setStartValue(self.logic.idle_color())
        self.color_animation.setEndValue(self.logic.hover_enter_color())
        self.scene().update()
        if self.current_timer >= 0:
            self.killTimer(self.current_timer)
        self.color_animation.start()
        self.current_timer = self.startTimer(self.color_animation.duration() // 80)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        self.hovered = False
        self.color_animation.setDuration(200)
        self.color_animation.setStartValue(self.logic.hover_enter_color())
        self.color_animation.setEndValue(self.logic.hover_left_color())
        self.scene().update()
        if self.current_timer > 0:
            self.killTimer(self.current_timer)
        self.color_animation.start()
        self.current_timer = self.startTimer(self.color_animation.duration() // 80)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent):
        pass

    @pyqtProperty(QColor)
    def current_fill_color(self):
        return self.fill_color_current

    @current_fill_color.setter
    def current_fill_color(self, color: QColor):
        self.fill_color_current = color

    def timerEvent(self, ev: QTimerEvent):
        self.scene().update()
        if self.color_animation.state() == QPropertyAnimation.Stopped:
            self.killTimer(ev.timerId())
            self.current_timer = 0

    def set_base_color(self, color: str):
        if isinstance(self.logic, CheckbuttonLogic):
            return
        if IDLE_COLORS[color.lower()] is None:
            return
        self.logic.color = color.lower()
        self.fill_color_current = self.logic.idle_color()

    def is_on(self):
        return self.logic.is_down()

    def set_is_check_button(self, value: bool):
        if value:
            self.logic = CheckbuttonLogic()
        else:
            self.logic = PushbuttonLogic("gray")

    def set_pixmap(self, pixmap: QPixmap):
        self.pixmap = pixmap.scaledToHeight(self.max_pixmap_height) if pixmap is not None else None
        self.fit_to_contents()

    def fit_to_contents(self):
        self.prepareGeometryChange()
        width = 2 * self.hor_margin
        height = 2 * self.ver_margin + self.label.boundingRect().height()
        if self.pixmap is not None:
            width += max(self.pixmap.width(), self.label.boundingRect().width())
            height += self.ver_margin + self.pixmap.height()
        else:
            width += self.label.boundingRect().width()
        self.rect.setWidth(width)
        self.rect.setHeight(height)

        self.label.setPos(0.5 * width - 0.5 * self.label.boundingRect().width() + 0.0 * self.hor_margin,
                          height - self.label.boundingRect().height() - self.ver_margin)
        self.update()
