from typing import Callable, Dict, Optional

from PyQt5.Qt import (QGraphicsItem, QGraphicsObject)
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap

from camera_processing.widgets.button import Button


class ButtonMenu(QGraphicsObject):

    def __init__(self, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.buttons: Dict[str, Button] = {}
        self.hidden_buttons: Dict[str, Button] = {}
        self.layout_direction = "horizontal"
        self.rect = QRectF(0, 0, 0, 0)
        self.hor_padding = 5
        self.ver_padding = 5

        self.fill_brush = QBrush(QColor(100, 100, 100, 200))
        self.outline_pen = QPen(QColor(255, 150, 0, 200))
        self.outline_pen.setWidth(2)
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
        if len(self.buttons) == 0:
            return
        visible_buttons = list(self.buttons.values())
        if len(visible_buttons) == 0:
            return
        if self.layout_direction == "horizontal":
            menu_width = 2 * self.hor_padding + sum(map(lambda btn: btn.boundingRect().width(), visible_buttons), 0)
            menu_width += (len(visible_buttons) - 1) * self.hor_padding
            self.rect.setWidth(menu_width)
            offset = self.hor_padding
            y = self.rect.height() / 2 - visible_buttons[0].boundingRect().height() / 2 #+ self.ver_padding
            for i, button in enumerate(visible_buttons):
                button.setPos(offset, y)
                offset += button.boundingRect().width() + self.hor_padding
        else:
            menu_height = self.ver_padding * (1 + len(visible_buttons)) + sum(map(lambda btn: btn.boundingRect().height(), visible_buttons), 0)
            menu_width = 2 * self.hor_padding + max(map(lambda btn: btn.boundingRect().width(), visible_buttons))
            self.rect.setHeight(menu_height)
            self.rect.setWidth(menu_width)
            offset = self.ver_padding
            x_base = self.rect.width() / 2
            for i, button in enumerate(visible_buttons):
                x = x_base - button.boundingRect().width() / 2
                button.setPos(x, offset)
                offset += button.boundingRect().height() + self.ver_padding
        self.scene().update(self.boundingRect())

    def set_layout_direction(self, direction: str):
        self.layout_direction = direction
        self._center_buttons()
    
    def add_button(self, btn_id: str, label: str, base_color: str = "gray", is_checkable: bool = False, call_back: Optional[Callable[[], None]] = None, pixmap: QPixmap = None) -> Button:
        old_btn = self.remove_button(btn_id)
        if old_btn is not None:
            old_btn.setParentItem(None)
            self.scene().removeItem(old_btn)
            old_btn.deleteLater()
        btn = Button(btn_id, label, self)
        btn.set_is_check_button(is_checkable)
        btn.set_base_color(base_color)
        btn.set_pixmap(pixmap)
        if call_back is not None:
            btn.clicked.connect(call_back)
        self.buttons[btn_id] = btn
        self._center_buttons()
        return btn
    
    def is_button_checked(self, button_id: str) -> bool:
        button = self.buttons[button_id]
        if button is None:
            return False
        return button.is_on()

    def remove_button(self, btn_id: str) -> Button:
        if btn_id not in self.buttons:
            return None
        btn = self.buttons[btn_id]
        del self.buttons[btn_id]
        if len(self.buttons) == 0:
            self.rect.setWidth(0)
            self.rect.setHeight(0)
        self._center_buttons()
        return btn

    def get_button(self, btn_id: str) -> Button:
        if btn_id in self.buttons:
            return self.buttons[btn_id]
        return None

    def hide_button(self, btn_id: str):
        if btn_id not in self.buttons:
            return
        button = self.buttons[btn_id]
        del self.buttons[btn_id]
        self.hidden_buttons[btn_id] = button
        button.setVisible(False)
        self._center_buttons()

    def show_button(self, btn_id: str):
        if btn_id not in self.hidden_buttons:
            return
        button = self.hidden_buttons[btn_id]
        del self.hidden_buttons[btn_id]
        self.buttons[btn_id] = button
        button.setVisible(True)
        self._center_buttons()
