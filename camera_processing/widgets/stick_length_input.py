from typing import Optional

from PyQt5.QtGui import QPainter, QKeyEvent, QBrush, QColor, QTextCursor
from PyQt5.QtWidgets import QGraphicsTextItem, QGraphicsObject, QGraphicsItem, QGraphicsRectItem, QWidget, \
    QStyleOptionGraphicsItem
from PyQt5.QtCore import QRectF, pyqtSignal, Qt, QPointF

from camera_processing.widgets.button import Button


class NumberInputWidget(QGraphicsTextItem):

    text_changed = pyqtSignal()
    accepted = pyqtSignal()
    cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsItem.__init__(self, parent)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if len(event.text()) == 0:
            return
        allowed_keys = [Qt.Key_Backspace, Qt.Key_Escape, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Right, Qt.Key_Direction_L, Qt.Key_Delete]
        text = event.text()
        if (48 > ord(text) or ord(text) > 57) and event.key() not in allowed_keys:
            return
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.accepted.emit()
            return
        elif event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            return
        else:
            QGraphicsTextItem.keyPressEvent(self, event)
        self.text_changed.emit()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        pass
        #if event.text() not in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
        #    return
        #value = self.toPlainText()
        #self.setPlainText(f"{value} cm")
        #self.update()

    def get_value(self) -> Optional[int]:
        text = self.toPlainText()
        if len(text) == 0:
            return None
        return int(text)

class StickLengthInput(QGraphicsObject):

    input_entered = pyqtSignal()
    input_cancelled = pyqtSignal()

    def __init__(self, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.accept_button = Button("btn_accept", "OK", parent=self)
        self.accept_button.set_base_color("green")
        self.accept_button.clicked.connect(self.input_entered.emit)
        self.cancel_button = Button("btn_cancel", "X", parent=self)
        self.cancel_button.set_base_color("red")
        self.cancel_button.clicked.connect(self.input_cancelled.emit)
        self.accept_button.set_height(30)
        self.cancel_button.set_height(30)
        self.hor_padding = 2
        self.background_rect = QGraphicsRectItem(parent=self)
        self.background_rect.setBrush(QBrush(QColor(50, 50, 50, 200)))
        self.text_field = NumberInputWidget(self)
        self.text_field.setFont(Button.font)
        self.text_field.setTextInteractionFlags(Qt.TextEditable)
        self.text_field.setPlainText("60")
        self.text_field.setDefaultTextColor(Qt.white)
        self.text_field.text_changed.connect(self.adjust_layout)
        self.text_field.accepted.connect(lambda: self.accept_button.click_button(artificial_emit=True))
        self.text_field.cancelled.connect(lambda: self.cancel_button.click_button(artificial_emit=True))
        self.cm_text = QGraphicsTextItem(self)
        self.cm_text.setFont(Button.font)
        self.cm_text.setPlainText("cm")
        self.cm_text.setDefaultTextColor(Qt.white)
        self.setVisible(False)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: Optional[QWidget] = ...) -> None:
        pass

    def boundingRect(self) -> QRectF:
        return self.background_rect.boundingRect().united(self.accept_button.boundingRect())\
            .united(self.cancel_button.boundingRect())

    def adjust_layout(self):
        total_width = 5 * self.hor_padding + self.text_field.boundingRect().width() \
                      + self.accept_button.boundingRect().width() + self.cancel_button.boundingRect().width() \
                      + self.cm_text.boundingRect().width()
        total_height = self.text_field.boundingRect().height()
        self.background_rect.setRect(0, 0, total_width, total_height)
        self.setTransformOriginPoint(QPointF(0.5 * total_width, 0.5 * total_height))
        #viewport = self.scene().views()[0].viewport().rect()
        #center = self.scene().views()[0].mapToScene(viewport.center())
        #self.setPos(QPointF(center.x(),
        #                    center.y()))
        self.text_field.setPos(-0.5 * total_width + self.hor_padding,
                               -total_height * 0.5)
        self.cm_text.setPos(self.hor_padding + self.text_field.pos().x() + self.text_field.boundingRect().width(), self.text_field.pos().y())
        self.background_rect.setPos(-0.5 * total_width, -0.5 * total_height)
        self.accept_button.setPos(QPointF(self.hor_padding + self.cm_text.boundingRect().width() + self.cm_text.pos().x(), -0.5 * self.accept_button.boundingRect().height()))
        self.cancel_button.setPos(QPointF(self.hor_padding + self.accept_button.pos().x() + self.accept_button.boundingRect().width(),
                                          -0.5 * self.cancel_button.boundingRect().height()))
        self.update()

    def get_length(self):
        return self.text_field.get_value()

    def set_length(self, length_cm: int):
        self.text_field.setPlainText(str(length_cm))
        self.adjust_layout()

    def set_focus(self):
        self.text_field.setFocus(Qt.PopupFocusReason)
        cursor = self.text_field.textCursor()
        cursor.movePosition(QTextCursor.EndOfLine)
        self.text_field.setTextCursor(cursor)
