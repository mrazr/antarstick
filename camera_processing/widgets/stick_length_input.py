from typing import Optional, Callable, Any

from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, pyqtSignal, Qt, QPointF
from PyQt5.QtGui import QPainter, QKeyEvent, QBrush, QColor, QTextCursor, QFont
from PyQt5.QtWidgets import QGraphicsTextItem, QGraphicsObject, QGraphicsItem, QGraphicsRectItem, QWidget, \
    QStyleOptionGraphicsItem

from camera_processing.widgets.button import Button, ButtonColor


class NumberInputItem(QGraphicsTextItem):

    text_changed = pyqtSignal()
    accepted = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsItem.__init__(self, parent)
        self.setTabChangesFocus(True)
        self.number_mode = False
        self.valid_input = True

    def keyPressEvent(self, event: QKeyEvent) -> None:
        allowed_keys = [Qt.Key_Backspace, Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Left, Qt.Key_Right, Qt.Key_Direction_L,
                        Qt.Key_Delete, Qt.Key_Minus]
        if event.key() == Qt.Key_Tab:
            return
        if self.number_mode:
            if event.key() not in allowed_keys and (event.key() < Qt.Key_0 or event.key() > Qt.Key_9):
                return
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.valid_input:
                self.accepted.emit()
            return
        elif event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        elif event.key() == Qt.Key_Left:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Left)
            self.setTextCursor(cursor)
        elif event.key() == Qt.Key_Right:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.Right)
            self.setTextCursor(cursor)
        else:
            super().keyPressEvent(event)
            self.text_changed.emit()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        pass

    def get_value(self) -> Optional[int]:
        text = self.toPlainText()
        if len(text) == 0:
            return None
        return int(text)

    def paint(self, painter: QtGui.QPainter, option: QStyleOptionGraphicsItem, widget: QWidget) -> None:
        pen = painter.pen()
        if self.valid_input:
            pen.setColor(QColor(100, 100, 100, 200))
        else:
            pen.setColor(QColor(200, 0, 0, 200))

        painter.setPen(pen)
        painter.fillRect(self.boundingRect(), pen.color())
        super().paint(painter, option, widget)

    def set_is_number_mode(self, is_number_mode: bool):
        self.number_mode = is_number_mode

    def set_is_valid(self, is_valid: bool):
        self.valid_input = is_valid
        self.update()


class TextInputWidget(QGraphicsObject):

    input_entered = pyqtSignal(str)
    input_cancelled = pyqtSignal(str)
    font = QFont('monospace', 16)

    def __init__(self, mode: str = 'number', label: str = '', getter=None, setter=None, parser=str, validator=None,
                 parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.getter: Callable[[], str] = getter
        self.setter: Callable[[str], None] = setter
        self.parser: Callable[[str], Any] = parser
        self.validator: Callable[[Any], bool] = validator
        self.accept_button = Button("btn_accept", "Accept", parent=self)
        self.accept_button.set_base_color([ButtonColor.GREEN])
        self.accept_button.clicked.connect(self.handle_input_accepted)
        self.cancel_button = Button("btn_cancel", "Cancel", parent=self)
        self.cancel_button.set_base_color([ButtonColor.RED])
        self.cancel_button.clicked.connect(lambda: self.input_cancelled.emit(self.text_field.toPlainText()))
        self.accept_button.set_height(12)
        self.cancel_button.set_height(12)
        self.hor_padding = 2
        self.background_rect = QGraphicsRectItem(parent=self)
        self.background_rect.setBrush(QBrush(QColor(50, 50, 50, 200)))
        self.text_field = NumberInputItem(self)
        self.text_field.setFont(self.font)
        self.text_field.setTextInteractionFlags(Qt.TextEditable)
        self.text_field.setDefaultTextColor(Qt.white)
        self.text_field.text_changed.connect(self.adjust_layout)
        self.text_field.set_is_number_mode(mode == 'number')
        if self.getter is not None:
            self.text_field.setPlainText(str(self.getter()))
            self.adjust_layout()
        self.text_field.accepted.connect(lambda: self.accept_button.click_button(artificial_emit=True))
        self.text_field.cancelled.connect(lambda: self.cancel_button.click_button(artificial_emit=True))
        self.text_label = QGraphicsTextItem(self)
        self.text_label.setFont(self.font)
        self.text_label.setPlainText(label)
        self.text_label.setDefaultTextColor(Qt.white)
        self.setVisible(False)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = ...) -> None:
        pass

    def boundingRect(self) -> QRectF:
        return self.background_rect.boundingRect().united(self.accept_button.boundingRect())\
            .united(self.cancel_button.boundingRect())

    def adjust_layout(self):
        self.accept_button.fit_to_contents()
        self.cancel_button.fit_to_contents()
        total_width = 3 * self.hor_padding + self.text_field.boundingRect().width() \
                      + self.text_label.boundingRect().width()
        total_width = max(total_width, 3 * self.hor_padding + self.accept_button.boundingRect().width() +
                          self.cancel_button.boundingRect().width())
        total_height = self.text_field.boundingRect().height() + self.accept_button.boundingRect().height() \
                       + 3 * self.hor_padding
        self.background_rect.setRect(0, 0, total_width, total_height)
        self.setTransformOriginPoint(QPointF(0.5 * total_width, 0.5 * total_height))
        self.text_label.setPos(-0.5 * total_width + self.hor_padding,
                               -0.5 * total_height + self.hor_padding)
        self.text_field.setPos(self.hor_padding + self.text_label.pos().x() + self.text_label.boundingRect().width(),
                               self.text_label.pos().y())
        self.background_rect.setPos(-0.5 * total_width, -0.5 * total_height)
        self.accept_button.set_width((total_width - 3 * self.hor_padding) * 0.5)
        self.cancel_button.set_width((total_width - 3 * self.hor_padding) * 0.5)
        self.accept_button.setPos(-0.5 * total_width + self.hor_padding, 2 * self.hor_padding)
        self.cancel_button.setPos(self.accept_button.pos() +
                                  QPointF(self.hor_padding + self.accept_button.boundingRect().width(), 0.0))
        self.accept_button.set_disabled(len(self.text_field.toPlainText()) == 0)
        if self.validator is not None:
            try:
                value = self.parser(self.text_field.toPlainText())
                is_valid = self.validator(value)
            except ValueError:
                is_valid = False
            self.text_field.set_is_valid(is_valid)
            self.accept_button.set_disabled(not is_valid)
        self.update()

    def get_value(self) -> str:
        return self.text_field.toPlainText()

    def set_value(self, text: str):
        self.text_field.setPlainText(text)
        self.adjust_layout()
        self.set_focus()

    def set_focus(self):
        self.text_field.setFocus(Qt.PopupFocusReason)
        cursor = self.text_field.textCursor()
        cursor.movePosition(QTextCursor.EndOfLine)
        self.text_field.setTextCursor(cursor)

    def set_getter_setter_parser_validator(self, getter: Callable[[], Any], setter: Callable[[Any], None],
                                           parser: Callable[[str], Any], validator: Callable[[Any], bool]):
        self.getter = getter
        self.setter = setter
        self.parser = parser
        self.validator = validator
        if self.getter is not None:
            self.text_field.setPlainText(str(self.getter()))
            self.adjust_layout()

    def set_label(self, label: str):
        self.text_label.setPlainText(label)
        self.adjust_layout()

    def handle_input_accepted(self):
        self.setter(self.parser(self.text_field.toPlainText()))
        self.input_entered.emit(self.text_field.toPlainText())

    def handle_input_cancelled(self):
        self.input_cancelled.emit(str(self.getter()))
