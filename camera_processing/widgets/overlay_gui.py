import sys
from pathlib import Path
from typing import Dict, Any

from PyQt5.Qt import QBrush, QColor, QPen
from PyQt5.QtCore import QPoint, QPointF, QRectF, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QPainter, QPixmap, QFontMetrics
from PyQt5.QtWidgets import QGraphicsItem, QGraphicsObject

from camera_processing.widgets.button_menu import ButtonMenu
from camera_processing.widgets.cam_graphics_view import CamGraphicsView


class OverlayGui(QGraphicsObject):

    reset_view_requested = pyqtSignal()
    edit_sticks_clicked = pyqtSignal()
    link_sticks_clicked = pyqtSignal()
    delete_sticks_clicked = pyqtSignal()
    redetect_sticks_clicked = pyqtSignal()
    process_photos_clicked = pyqtSignal()
    process_photos_count_clicked = pyqtSignal(int)
    #sticks_length_clicked = pyqtSignal()

    def __init__(self, view: CamGraphicsView, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)

        self.view = view
        self.view.view_changed.connect(self.handle_cam_view_changed)
        self.setZValue(43)
        self.setAcceptHoverEvents(False)
        self.setAcceptedMouseButtons(Qt.AllButtons)
        self.top_menu = ButtonMenu(self)
        self.mouse_zoom_pic = QPixmap()
        self.mouse_pan_pic = QPixmap()
        self.icons_rect = QRectF(0, 0, 0, 0)
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        #self.top_menu.setVisible(False)
        self.loading_screen_shown = True
        self.process_photo_popup = ButtonMenu(self)

    def initialize(self):
        path = Path(sys.argv[0]).parent / "camera_processing/gui_resources/"

        self.mouse_zoom_pic = QPixmap(str(path / "mouse_zoom.png"))
        self.mouse_zoom_pic = self.mouse_zoom_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.mouse_pan_pic = QPixmap(str(path / "mouse_pan.png"))
        self.mouse_pan_pic = self.mouse_pan_pic.scaledToWidth(80, Qt.SmoothTransformation)

        self.top_menu.add_button("detect_sticks", "Re-detect sticks", call_back=self.redetect_sticks_clicked.emit)
        self.top_menu.add_button("edit_sticks", "Edit sticks", is_checkable=True, call_back=self.edit_sticks_clicked.emit)
        self.top_menu.add_button("link_sticks", "Link sticks", is_checkable=True, call_back=self.link_sticks_clicked.emit)
        #self.top_menu.add_button("show_overlay", "Show overlay")
        #self.top_menu.add_button("show_linked_cameras", "Show linked cameras")
        self.top_menu.add_button("reset_view", "Reset view", call_back=self.reset_view_requested.emit)
        self.top_menu.add_button("delete_sticks", "Delete selected sticks", call_back=self.delete_sticks_clicked.emit, base_color="red")
        self.top_menu.hide_button("delete_sticks")
        process_btn = self.top_menu.add_button("process_photos", "Process photos", is_checkable=True) #call_back=self.process_photos_clicked.emit)
        process_btn.clicked.connect(lambda _: self.handle_process_photos_clicked())
        self.top_menu.set_height(40)

        self.top_menu.setPos(QPoint(0, 0))

        self.icons_rect = QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height())
    
    @pyqtSlot()
    def handle_cam_view_changed(self):
        self.prepareGeometryChange()
        self.setPos(self.view.mapToScene(QPoint(0, 0)))
        self.top_menu.setPos(self.boundingRect().width() / 2.0 - self.top_menu.boundingRect().width() / 2.0, 2)
        self.process_photo_popup.setPos(self.boundingRect().width() / 2.0 - self.process_photo_popup.boundingRect().width() / 2.0,
                                        self.boundingRect().height() / 2.0 - self.process_photo_popup.boundingRect().height() / 2.0)

    def boundingRect(self):
        return QRectF(self.view.viewport().rect())
    
    def paint(self, painter: QPainter, options, widget=None):
        painter.save()

        painter.setWorldMatrixEnabled(False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#aa555555")))
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)
        painter.drawRoundedRect(QRectF(5, 5, self.mouse_pan_pic.width() * 1.3,
                                       3 * self.mouse_pan_pic.height()), 20, 20)

        painter.setPen(QPen(QColor("#e5c15f")))
        painter.drawPixmap(QPointF(15, 5), self.mouse_pan_pic, QRectF(self.mouse_pan_pic.rect()))
        rect = QRectF(self.mouse_pan_pic.rect())
        painter.drawPixmap(QPointF(15, 1.5 * rect.height()), self.mouse_zoom_pic,
                           QRectF(self.mouse_zoom_pic.rect()))
        font = painter.font()
        font.setPointSize(10)
        painter.drawText(QRectF(15, 1.1 * rect.height(), rect.width(), 30),
                         Qt.AlignHCenter, "Pan view")
        painter.drawText(QRectF(15, 2.6 * rect.height(), rect.width(), 30),
                         Qt.AlignHCenter, "Zoom in/out")

        if self.loading_screen_shown:
            painter.fillRect(0, 0, self.boundingRect().width(), self.boundingRect().height(), QBrush(QColor(255, 255, 255, 255)))
            fm = QFontMetrics(font)
            font.setPixelSize(int(self.boundingRect().width() / 15))
            painter.setFont(font)
            painter.drawText(self.boundingRect(), Qt.AlignCenter, 'initializing')
        painter.setWorldMatrixEnabled(True)
        painter.restore()

    def edit_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("edit_sticks")
    
    def link_sticks_button_pushed(self) -> bool:
        return self.top_menu.is_button_checked("link_sticks")

    def show_loading_screen(self, show: bool):
        self.loading_screen_shown = show
        self.top_menu.setVisible(not self.loading_screen_shown)
        self.update()

    def enable_delete_sticks_button(self, val: bool):
        self.top_menu.show_button("delete_sticks") if val else self.top_menu.hide_button("delete_sticks")
        self.handle_cam_view_changed()

    def initialize_process_photos_popup(self, photo_count: int, loading_time: float):
        loading_time = 0.4 # Bit of a magic number here for now. For my PC it was 0.15s per image.
        self.process_photo_popup.set_height(40)
        step = photo_count // 5
        self.process_photo_popup.add_button('100', f'100: {int(round(100 * loading_time))} s', call_back=self.handle_process_photos_count_clicked)

        count = step
        for i in range(1, 5):
            self.process_photo_popup.add_button(str(count), f'{count}: {int(round(i * count * loading_time))} s', call_back=self.handle_process_photos_count_clicked)

        self.process_photo_popup.add_button(photo_count, f'All: {int(round(photo_count * loading_time))} s', call_back=self.handle_process_photos_count_clicked)
        cancel_btn = self.process_photo_popup.add_button('btn_cancel', 'Cancel', base_color='red')
        cancel_btn.clicked.connect(self.handle_process_photos_count_cancel_clicked)
        self.process_photo_popup.setVisible(False)

    def handle_process_photos_count_cancel_clicked(self):
        self.process_photo_popup.setVisible(False)
        self.top_menu.get_button('process_photos').click_button(True)
        self.process_photo_popup.reset_button_states()

    def handle_process_photos_count_clicked(self, btn_data: Dict[str, Any]):
        self.process_photo_popup.setVisible(False)
        self.process_photo_popup.reset_button_states()
        btn = self.top_menu.get_button('process_photos')
        btn.click_button(True)
        btn.set_disabled(True)
        self.process_photos_count_clicked.emit(int(btn_data['btn_id']))

    def handle_process_photos_clicked(self):
        is_down = self.top_menu.get_button('process_photos').is_on()
        self.process_photo_popup.setVisible(is_down)

    def enable_process_photos_button(self, enable: bool):
        self.top_menu.get_button('process_photos').set_disabled(not enable)
