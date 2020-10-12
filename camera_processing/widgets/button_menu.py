from typing import Callable, Dict, Optional
from math import ceil

from PyQt5.Qt import (QGraphicsItem, QGraphicsObject)
from PyQt5.QtCore import QRectF, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QPainter, QPen, QPixmap

from camera_processing.widgets.button import Button, ButtonColor


class ButtonMenu(QGraphicsObject):
    close_requested = pyqtSignal()

    def __init__(self, scaling: float, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.scaling = 1.0
        self.buttons: Dict[str, Button] = {}
        self.hidden_buttons: Dict[str, Button] = {}
        self.layout_direction = "horizontal"
        self.rect = QRectF(0, 0, 0, 0)
        self.hor_padding = 5
        self.ver_padding = 5

        self.close_button = Button('btn_close', 'cancel', parent=self)
        self.close_button.set_base_color([ButtonColor.RED])
        self.close_button.clicked.connect(lambda _: self.close_requested.emit())
        self.close_button.setVisible(False)
        self.close_button_shown = False

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
        height = 0
        for _, button in self.buttons.items():
            button.set_height(h)
            height = button.boundingRect().height()
        self.rect.setHeight(height + 2 * self.ver_padding)
        self.center_buttons()

    def set_width(self, w: int):
        if self.layout_direction == "horizontal":
            return
        self.rect.setWidth(w)
        self.center_buttons()

    def center_buttons(self):
        if len(self.buttons) == 0:
            return
        visible_buttons = list(self.buttons.values())

        if self.close_button_shown:
            # self.close_button.set_height(visible_buttons[0].boundingRect().height())
            self.close_button.set_height(12)
            if self.layout_direction == 'vertical':
                self.close_button.set_width(visible_buttons[0].boundingRect().width())
            # visible_buttons.append(self.close_button)

        if len(visible_buttons) == 0:
            return

        if self.layout_direction == "horizontal":
            menu_width = 2 * self.hor_padding + sum(map(lambda btn: btn.boundingRect().width(), visible_buttons), 0)
            menu_width += (len(visible_buttons) - 1) * self.hor_padding
            if self.close_button_shown:
                menu_width += self.close_button.boundingRect().width() + self.hor_padding
            menu_height = visible_buttons[0].boundingRect().height() + 2 * self.ver_padding
            self.rect.setWidth(menu_width)
            self.rect.setHeight(menu_height)
            offset = self.hor_padding
            y = self.rect.height() / 2 - visible_buttons[0].boundingRect().height() / 2  # + self.ver_padding
            for i, button in enumerate(visible_buttons):
                button.setPos(offset, y)
                offset += button.boundingRect().width() + self.hor_padding
            if self.close_button_shown:
                self.close_button.setPos(offset, y)
        else:
            menu_height = self.ver_padding * self.scaling * (1 + len(visible_buttons)) + \
                          sum(map(lambda btn: btn.boundingRect().height(), visible_buttons), 0)
            menu_height += self.close_button.boundingRect().height()
            view_size = self.scene().views()[0].geometry().size()
            view = self.scene().views()[0]
            rows = len(visible_buttons)
            columns = 1

            if menu_height > view_size.height():
                rows = max(int(view_size.height() / visible_buttons[0].boundingRect().height()), 1)
                columns = int(ceil(len(visible_buttons) / rows))

            # print(f'rows = {rows}, columns = {columns}')
            # print(f'visible buttons = {len(visible_buttons)}')

            button_width = max(map(lambda btn: btn.boundingRect().width(), visible_buttons))
            button_height = max(map(lambda btn: btn.boundingRect().height(), visible_buttons))

            menu_height = self.scaling * self.ver_padding * (1 + rows) + rows * button_height
            if self.close_button_shown:
                menu_height += self.close_button.boundingRect().height() + self.ver_padding
            # menu_width = 2 * self.hor_padding * self.scaling + columns * button_width + max(0, columns - 1) * self.hor_padding * self.scaling
            menu_width = self.scaling * self.hor_padding * (1 + columns) + columns * button_width

            self.rect.setHeight(menu_height)
            self.rect.setWidth(menu_width)
            offset = 0
            x = 0
            for i, button in enumerate(visible_buttons):
                r = i % rows
                c = int(i / rows)
                x = c * (button_width + self.scaling * self.hor_padding) + self.scaling * self.hor_padding
                offset = r * (button.boundingRect().height() + self.scaling * self.ver_padding) + \
                         self.scaling * self.ver_padding
                button.setPos(x, offset)
            if self.close_button_shown:
                self.close_button.setPos(x, offset + visible_buttons[0].boundingRect().height() + self.ver_padding)
        self.scene().update(self.boundingRect())

    def set_layout_direction(self, direction: str):
        self.layout_direction = direction
        self.center_buttons()

    def add_button(self, btn_id: str, label: str, base_color: ButtonColor = ButtonColor.GRAY,
                   is_checkable: bool = False, call_back: Optional[Callable[[], None]] = None,
                   pixmap: QPixmap = None) -> Button:
        old_btn = self.remove_button(btn_id)
        if old_btn is not None:
            old_btn.setParentItem(None)
            self.scene().removeItem(old_btn)
            old_btn.deleteLater()
        btn = Button(btn_id, label, parent=self)
        # btn.scale_button(self.scaling)
        btn.set_is_check_button(is_checkable)
        if not is_checkable:
            btn.set_base_color([base_color])
        btn.set_pixmap(pixmap)
        if call_back is not None:
            btn.clicked.connect(call_back)
        self.buttons[btn_id] = btn
        # self._center_buttons()
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
        self.center_buttons()
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
        self.center_buttons()

    def show_button(self, btn_id: str):
        if btn_id not in self.hidden_buttons:
            return
        button = self.hidden_buttons[btn_id]
        del self.hidden_buttons[btn_id]
        self.buttons[btn_id] = button
        button.setVisible(True)
        self.center_buttons()

    def reset_button_states(self):
        for button in self.buttons.values():
            button.set_default_state()

    def show_close_button(self, show: bool):
        self.close_button.setVisible(show)
        self.close_button_shown = show

    def scale_menu(self, factor: float):
        self.scaling = factor
        for btn in self.buttons.values():
            btn.scale_button(self.scaling)
        self.center_buttons()
