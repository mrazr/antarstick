from typing import List, Optional, Tuple

from PyQt5.Qt import (QColor, QGraphicsItem, QGraphicsObject, QPainter, QPen,
                      QPointF, QRectF, QStyleOptionGraphicsItem, pyqtSignal)

from camera import Camera
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.stick_widget import StickWidget
from dataset import Dataset
from stick import Stick


class StickLinkManager(QGraphicsObject):

    def __init__(self, dataset: Dataset, camera: Camera, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.dataset = dataset
        self.camera = camera
        self.source: StickWidget = None
        self.target: StickWidget = None
        self.target_point = QPointF()
        self.anchored = False
        self.primary_camera: CustomPixmap = None
        self.secondary_cameras: List[CustomPixmap] = []
        self.links: List[Tuple[int, int, int]] = []
        self.dataset.sticks_linked.connect(self.update_links)
        self.dataset.sticks_unlinked.connect(self.handle_sticks_unlinked)
    
    def boundingRect(self):
        return QRectF(self.scene().views()[0].viewport().rect())

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        if self.source is not None:
            if self.anchored:
                pen = QPen(QColor(0, 255, 20, 255))
            else:
                pen = QPen(QColor(180, 200, 0, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawLine(self.source.mapToScene(self.source.mid_handle.rect().center()), self.target_point)
        
        if len(self.links) > 0:
            hue_step = int(360 / len(self.links))
            hue = 0

            for sw in self.primary_camera.stick_widgets:
                if sw.stick.id not in self.dataset.stick_views_map:
                    continue
                link = self.dataset.stick_views_map[sw.stick.id]
                for cam in self.secondary_cameras:
                    if link[1] != cam.camera.id:
                        continue
                    pen = QPen(QColor(0, 0, 0, 200))
                    pen.setWidth(4)
                    painter.setPen(pen)
                    sw2 = next(filter(lambda _sw: _sw.stick.id == link[2], cam.stick_widgets))
                    p1 = sw.mapToScene(sw.mid_handle.rect().center())
                    p2 = sw2.mapToScene(sw2.mid_handle.rect().center())
                    pen.setColor(QColor.fromHsv(hue, 100, 255, 255))
                    hue += hue_step
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.drawLine(p1, p2)



    def handle_stick_widget_link_requested(self, stick: StickWidget):
        for sw in self.primary_camera.stick_widgets:
            sw.set_available_for_linking(False)
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_available_for_linking(True)

        self.source = stick
        self.target_point = stick.mapToScene(QPointF(stick.mid_handle.rect().center()))
        self.update()
    
    def set_target(self, point: QPointF):
        if self.source is not None:
            self.target_point = point
            self.update()
    
    def accept(self):
        if self.source is not None and self.target is not None:
            self.dataset.link_sticks(self.source.stick, self.target.stick)
            self.links = self.dataset.get_cameras_stick_links(self.camera)
            self.cancel()

    def cancel(self):
        self.source = None
        self.target = None
        self.update()
    
    def anchor_to(self, stick_widget: StickWidget):
        self.target = stick_widget.mapToScene(QPointF(stick_widget.mid_handle.rect().center()))
        self.anchored = True
        self.update()
    
    def unanchor(self):
        self.anchored = False
    
    def handle_stick_widget_hover(self, entered: bool, stick_widget: StickWidget):
        self.anchored = entered
        if self.anchored:
            self.target = stick_widget
            self.target_point = self.target.mapToScene(self.target.mid_handle.rect().center())
        else:
            self.target = None
        self.update()
    
    def set_secondary_cameras(self, cameras: List[CustomPixmap]):
        self.secondary_cameras = cameras
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.hovered.connect(self.handle_stick_widget_hover)
    
    def update_links(self):
        self.links = self.dataset.get_cameras_stick_links(self.camera)
        self.update()

    def handle_sticks_unlinked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        print("handling unlinkage")
        self.update_links()