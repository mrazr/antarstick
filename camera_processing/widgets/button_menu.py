from typing import Callable, Dict, List, Optional

from PyQt5.Qt import (QGraphicsItem, QGraphicsObject, QGraphicsSceneHoverEvent,
                      QGraphicsSceneMouseEvent, QGraphicsSimpleTextItem,
                      pyqtSignal)
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPen

from camera_processing.widgets.button import Button


class ButtonMenu(QGraphicsObject):

    def __init__(self, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.buttons: Dict[str, Button] = {}
        self.layout_direction = "horizontal"
        self.rect = QRectF(0, 0, 0, 0)
        self.hor_padding = 5
        self.ver_padding = 5

        self.fill_brush = QBrush(QColor(100, 100, 100, 200))
        self.outline_pen = QPen(QColor(255, 150, 0, 200))
        self.outline_pen.setWidth(2)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
    
    def boundingRect(self):
        return self.rect
    
    def paint(self, painter: QPainter, options, widget=None):
        painter.setBrush(self.fill_brush)
        painter.setPen(self.outline_pen)
        painter.drawRoundedRect(self.rect, 5, 5)
    
    def set_height(self, h: int):
        if self.layout_direction == "vertical":
            return
        self.rect.setHeight(h)
        for _, button in self.buttons.items():
            button.set_height(h - 2 * self.ver_padding)
        self._center_buttons()
    
    def set_width(self, w: int):
        if self.layout_direction == "horizonal":
            return
        self.rect.setWidth(w)
        self._center_buttons()
    
    def _center_buttons(self):
        buttons = list(self.buttons.values())
        if self.layout_direction == "horizontal":
            menu_width = 2 * self.hor_padding + sum(map(lambda btn: btn.boundingRect().width(), buttons), 0)
            menu_width += (len(self.buttons) - 1) * self.hor_padding
            self.rect.setWidth(menu_width)
            offset = self.hor_padding
            y = self.rect.height() / 2 - buttons[0].boundingRect().height() / 2 #+ self.ver_padding
            for i, button in enumerate(buttons):
                button.setPos(offset, y)
                offset += button.boundingRect().width() + self.hor_padding
        else:
            menu_height = 2 * self.ver_padding + sum(map(lambda btn: btn.boundingRect().height(), buttons), 0)
            self.rect.setWidth(menu_height)
            offset = self.ver_padding
            x = self.rect.width() / 2 - buttons[0].boundingRect().width() / 2 + self.hor_padding
            for i, button in enumerate(buttons):
                button.setPos(x, offset)
                offset += button.boundingRect().height() + self.ver_padding
        self.scene().update(self.boundingRect())

    def set_layout_direction(self, direction: str):
        self.layout_direction = direction
        self._center_buttons()
    
    def add_button(self, btn_id: str, label: str, base_color: str = "gray", is_checkable: bool = False, call_back: Optional[Callable[[], None]] = None):
        btn = Button(label, self)
        btn.set_is_check_button(is_checkable)
        btn.set_base_color(base_color)
        if call_back is not None:
            btn.clicked.connect(call_back)
        self.buttons[btn_id] = btn
        self._center_buttons()
