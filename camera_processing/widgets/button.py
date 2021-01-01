from typing import Union, Optional, List
from enum import IntEnum

from PyQt5.Qt import (QGraphicsItem, QGraphicsObject, QGraphicsSceneHoverEvent,
                      QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                      pyqtSignal)
from PyQt5.QtCore import (QEasingCurve, QPointF, QPropertyAnimation, QRectF,
                          QTimerEvent, pyqtProperty, Qt, QMarginsF)
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap


class ButtonState(IntEnum):
    DEFAULT = 0
    HOVERED = 1
    PRESSED = 2

    CHECKED_DEFAULT = 3
    CHECKED_HOVERED = 4
    CHECKED_PRESSED = 5


class ButtonColor(IntEnum):
    GRAY = 0
    RED = 1
    GREEN = 2


class ButtonMode(IntEnum):
    Button = 0
    Label = 1

IDLE_COLORS = [
    QColor(50, 50, 50, 200),
    QColor(255, 0, 0, 200),
    QColor(0, 255, 0, 200),
]

HOVER_COLORS = [
    QColor(255, 125, 0, 150),
    IDLE_COLORS[ButtonColor.RED].lighter(120),
    IDLE_COLORS[ButtonColor.GREEN].lighter(120),
]

PRESS_COLORS = [
    HOVER_COLORS[ButtonColor.GRAY].darker(120),
    IDLE_COLORS[ButtonColor.RED].darker(120),
    IDLE_COLORS[ButtonColor.GREEN].darker(120),
]



class CheckbuttonLogic:

    def __init__(self):
        self.down = False
        #self.color_idle = idle_checked_colors[0]
        #self.color_checked = idle_checked_colors[1]
        self.colors: List[QColor] = [
            IDLE_COLORS[ButtonColor.GRAY],
            HOVER_COLORS[ButtonColor.GRAY],
            PRESS_COLORS[ButtonColor.GRAY],

            IDLE_COLORS[ButtonColor.GREEN],
            HOVER_COLORS[ButtonColor.GREEN],
            PRESS_COLORS[ButtonColor.GREEN],
        ]
        #self.checked_label = checked_label

    def idle_color(self) -> QColor:
        if self.down:
            #return IDLE_COLORS[self.color_checked]
            return self.colors[ButtonState.CHECKED_DEFAULT]
        #return IDLE_COLORS[self.color_idle] if self.colors is None else self.colors[0]
        return self.colors[ButtonState.DEFAULT]

    def hover_left_color(self) -> QColor:
        return self.idle_color()

    def hover_enter_color(self) -> QColor:
        if self.down:
            return self.colors[ButtonState.CHECKED_HOVERED]
            #return HOVER_COLORS[self.color_checked]
        return self.colors[ButtonState.HOVERED]
        #return HOVER_COLORS[self.color_idle] if self.colors is None else self.colors[1]

    def press_color(self) -> QColor:
        if self.down:
            return self.colors[ButtonState.CHECKED_PRESSED]
        return self.colors[ButtonState.PRESSED]
        #return PRESS_COLORS[ButtonState.PRESSED] if self.colors is None else self.colors[2]

    def release_color(self) -> QColor:
        if self.down:
            return self.colors[ButtonState.CHECKED_HOVERED]
            #return HOVER_COLORS[self.color_checked]
        return self.colors[ButtonState.HOVERED]
        #return HOVER_COLORS[self.color_idle] if self.colors is None else self.colors[1]

    def is_down(self) -> bool:
        return self.down

    def do_click(self):
        self.down = not self.down

    def reset_state(self):
        self.down = False

    def set_colors(self, base_color: Optional[List[ButtonColor]] = None, colors: Optional[List[QColor]] = None):
        if base_color is not None:
            self.colors = [
                IDLE_COLORS[base_color[0]],
                HOVER_COLORS[base_color[0]],
                PRESS_COLORS[base_color[0]],

                IDLE_COLORS[base_color[1]],
                HOVER_COLORS[base_color[1]],
                PRESS_COLORS[base_color[1]],
            ]
        elif colors is not None:
            self.colors = colors


class PushbuttonLogic:

    def __init__(self, color: str):
        #self.color = color.lower()
        self.colors: List[QColor] = [
            IDLE_COLORS[ButtonColor.GRAY],
            HOVER_COLORS[ButtonColor.GRAY],
            PRESS_COLORS[ButtonColor.GRAY]
        ]

    def idle_color(self) -> QColor:
        #return IDLE_COLORS[self.color] if self.colors is None else self.colors[0]
        return self.colors[ButtonState.DEFAULT]

    def hover_left_color(self) -> QColor:
        return self.idle_color()

    def hover_enter_color(self) -> QColor:
        #return HOVER_COLORS[self.color] if self.colors is None else self.colors[1]
        return self.colors[ButtonState.HOVERED]

    def press_color(self) -> QColor:
        #return PRESS_COLORS[self.color] if self.colors is None else self.colors[2]
        return self.colors[ButtonState.PRESSED]

    def release_color(self) -> QColor:
        return self.hover_enter_color()
        #return HOVER_COLORS[self.color]

    def is_down(self) -> bool:
        return False

    def do_click(self):
        pass

    def reset_state(self):
        pass

    def set_colors(self, base_color: Optional[List[ButtonColor]] = None, colors: Optional[List[QColor]] = None):
        if base_color is not None:
            base_color = base_color[0]
            self.colors: List[QColor] = [
                IDLE_COLORS[base_color],
                HOVER_COLORS[base_color],
                PRESS_COLORS[base_color]
            ]
        elif colors is not None:
            self.colors = colors


class Button(QGraphicsObject):
    font: QFont = QFont("monospace", 16)

    clicked = pyqtSignal('PyQt_PyObject')
    hovered_ = pyqtSignal('PyQt_PyObject')

    def __init__(self, btn_id: str, label: str, tooltip: str = "", parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.scaling = 1.0
        self.label = QGraphicsSimpleTextItem(label, self)
        #self.font_ = QFont(self.font)
        self.label.setFont(Button.font)
        #self.label.setDefaultTextColor(QColor(255, 255, 255, 255))
        #self.label.setTextInteractionFlags(Qt.TextEditable)
        self.text_color_enabled = QColor(255, 255, 255, 255)
        self.text_color_disabled = QColor(200, 200, 200, 255)
        self.fill_color_disabled = QColor(125, 125, 125, 200)
        self.label.setBrush(QBrush(self.text_color_enabled))
        self.btn_id = btn_id
        self.rect = QRectF(0, 0, 0, 0)

        self.tooltip = QGraphicsSimpleTextItem(tooltip, self)
        self.tooltip.setBrush(QColor(255, 255, 255, 200))
        self.tooltip.setFont(Button.font)
        self.tooltip.setVisible(False)
        self.tooltip.setZValue(25)
        self.tooltip_shown = False

        self.base_color = ButtonColor.GRAY

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
        self.set_height(12)

        self.pixmap: QPixmap = None
        self.max_pixmap_height = 128
        self.disabled = False

        self.mode = ButtonMode.Button

    def set_height(self, h: int):
        self.prepareGeometryChange()
        self.ver_margin = int(0.25 * h)
        font: QFont = self.label.font()
        font.setPointSize(h)
        self.label.setFont(font)
        self.rect.setWidth(self.label.boundingRect().width() + 2 * self.hor_margin)
        self.rect.setHeight(self.label.boundingRect().height() + 2 * self.ver_margin)
        self._reposition_text()

    def set_width(self, w: int):
        if self.pixmap is not None:
            return
        self.prepareGeometryChange()
        self.rect.setWidth(w)
        self.hor_margin = self.ver_margin
        if self.label.boundingRect().width() > self.rect.width():
            w = self.rect.width() - 2 * self.hor_margin
            factor = w / self.label.boundingRect().width()
            h = factor * self.label.boundingRect().height()
            font = self.label.font()
            font.setPixelSize(max(h, 12))
            self.label.setFont(font)
        self._reposition_text()

    def set_button_height(self, h: int):
        self.rect.setHeight(h)
        self._reposition_text()

    def scale_button(self, factor: float):
        #self.scaling = factor
        factor = 1.0
        self.rect.setHeight(int(factor * self.rect.height()))
        self.rect.setWidth(int(factor * self.rect.width()))
        #self.font.setPointSize(int(factor * self.font.pointSize()))
        #self.label.setFont(self.font)
        self.label.setScale(self.scaling)
        self.fit_to_contents()

    def _reposition_text(self):
        x = self.rect.width() / 2 - self.label.boundingRect().width() / 2
        y = self.rect.height() / 2 - self.label.boundingRect().height() / 2
        self.label.setPos(QPointF(x, y))
        self.update()

    def boundingRect(self):
        return self.rect

    def paint(self, painter: QPainter, options, widget=None):
        painter.setBrush(QBrush(self.fill_color_current))
        painter.setPen(QPen(QColor(0, 0, 0, 0)))

        painter.drawRoundedRect(self.rect, 5, 5)

        if self.pixmap is not None:
            painter.drawPixmap(self.hor_margin * self.scaling, self.ver_margin * self.scaling, self.pixmap)
        if self.tooltip_shown:
            painter.setBrush(QBrush(QColor(50, 50, 50, 200)))
            painter.drawRoundedRect(self.tooltip.boundingRect().translated(self.tooltip.pos())
                                    .marginsAdded(QMarginsF(5, 0, 5, 0)), 5, 5)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        if self.disabled or self.mode == ButtonMode.Label:
            return
        self.fill_color_current = self.logic.press_color()
        self.scene().update()

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        if self.disabled or self.mode == ButtonMode.Label:
            return
        self.click_button()

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent):
        if self.disabled or self.mode == ButtonMode.Label:
            return
        self.hovered_.emit({'btn_id': self.btn_id, 'button': self, 'hovered': True})
        self.hovered = True

        if len(self.tooltip.text()) > 0:
            self.tooltip_shown = True
            self.tooltip.setVisible(True)
            view = self.scene().views()[0]
            rect_ = view.mapFromScene(self.tooltip.sceneBoundingRect()).boundingRect()
            pos = self.boundingRect().topRight()
            mouse_pos = view.mapFromScene(event.scenePos())
            if mouse_pos.x() + rect_.width() >= view.viewport().width():
                pos = QPointF(-self.tooltip.boundingRect().width(), 0)

            self.tooltip.setPos(pos)

        self.color_animation.setDuration(200)
        self.color_animation.setStartValue(self.logic.idle_color())
        self.color_animation.setEndValue(self.logic.hover_enter_color())
        self.scene().update()
        if self.current_timer >= 0:
            self.killTimer(self.current_timer)
        self.color_animation.start()
        self.current_timer = self.startTimer(self.color_animation.duration() // 80)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent):
        if self.disabled or self.mode == ButtonMode.Label:
            return

        self.hovered_.emit({'btn_id': self.btn_id, 'button': self, 'hovered': False})
        self.hovered = False
        self.tooltip_shown = False
        self.tooltip.setVisible(False)

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

    def set_base_color(self, colors: List[ButtonColor]):
        #if isinstance(self.logic, CheckbuttonLogic):
        #    return
        #if IDLE_COLORS[color] is None:
        #    return
        #self.logic.color = color.lower()
        self.logic.set_colors(colors)
        self.fill_color_current = self.logic.idle_color()

    def is_on(self):
        return self.logic.is_down()

    def set_on(self, value: bool):
        if isinstance(self.logic, CheckbuttonLogic):
            self.logic.down = value
            self.fill_color_current = self.logic.press_color() if value else self.logic.idle_color()
            self.update()

    def set_is_check_button(self, value: bool):
        if value:
            self.logic = CheckbuttonLogic()
        else:
            self.logic = PushbuttonLogic("gray")

    def set_pixmap(self, pixmap: QPixmap):
        if pixmap is not None:
            self.pixmap = pixmap.scaledToHeight(self.max_pixmap_height * self.scaling) if pixmap is not None else None
            self.fit_to_contents()

    def fit_to_contents(self):
        self.prepareGeometryChange()
        width = 2 * self.hor_margin * self.label.scale()
        height = 2 * self.ver_margin * self.label.scale() + self.label.boundingRect().height() * self.label.scale()
        if self.pixmap is not None:
            width += max(self.pixmap.width(), self.label.boundingRect().width() * self.label.scale())
            height += self.ver_margin * self.scaling + self.pixmap.height()
        else:
            width += self.label.boundingRect().width() * self.label.scale()
        self.rect.setWidth(width)
        self.rect.setHeight(height)

        self.label.setPos(0.5 * width - 0.5 * self.label.boundingRect().width() * self.label.scale() + 0.0 * self.hor_margin,
                          height - self.label.boundingRect().height() * self.label.scale() - self.ver_margin * self.scaling)
        self.update()

    def adjust_text_to_button(self):
        height_diff = 0.75 * self.rect.height() - self.label.boundingRect().height()
        fac = height_diff / (0.75 * self.rect.height())
        self.label.setTransformOriginPoint(self.label.boundingRect().center())
        self.label.setScale(1.0 + fac)
        self._reposition_text()

    def set_label(self, text: str, direction: str = 'horizontal'):
        if direction == 'vertical':
            text = '\n'.join(list(text))
        self.label.setText(text)
        self._reposition_text()
        #self.fit_to_contents()

    def click_button(self, artificial_emit: bool = False):
        if self.disabled:
            return
        self.logic.do_click()
        self.fill_color_current = self.logic.release_color() if not artificial_emit else self.logic.idle_color()
        self.clicked.emit({"btn_id": self.btn_id, "btn_label": self.label, "button": self, 'checked': self.is_on()})
        if artificial_emit:
            self.hovered_.emit({'btn_id': self.btn_id, 'button': self, 'hovered': False})
            self.update()
        if self.scene() is not None:
            self.scene().update()

    def set_opacity(self, opacity: float):
        self.setOpacity(opacity)
        self.update()

    def set_default_state(self):
        self.logic.reset_state()
        self.fill_color_current = self.logic.idle_color()
        self.tooltip.setVisible(False)
        self.tooltip_shown = False
        self.update()

    def set_disabled(self, disabled: bool):
        self.disabled = disabled
        if disabled:
            self.label.setBrush(QBrush(self.text_color_disabled))
            self.fill_color_current = self.fill_color_disabled
        else:
            self.label.setBrush(QBrush(self.text_color_enabled))
            self.fill_color_current = self.logic.idle_color()
        self.update()

    def set_tooltip(self, tooltip: str):
        self.tooltip.setVisible(False)
        self.tooltip_shown = False
        self.tooltip.setText(tooltip)

    def set_custom_color(self, colors: List[QColor]):
        if isinstance(self.logic, CheckbuttonLogic):
            if len(colors) < 2:
                return
            self.logic.set_colors(colors=[
                colors[0],
                colors[0].lighter(120),
                colors[0].darker(120),

                colors[1],
                colors[1].lighter(120),
                colors[1].darker(120)
            ])
        else:
            self.logic.set_colors(colors=[
                colors[0],
                colors[0].lighter(120),
                colors[0].darker(120)
            ])
        #self.logic.set_colors([color, color.lighter(120), color.darker(120)])
        self.fill_color_current = self.logic.idle_color()
        self.update()

    def set_mode(self, mode: ButtonMode):
        self.mode = mode