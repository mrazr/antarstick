from typing import List, Optional, Tuple

from PyQt5.Qt import (QColor, QGraphicsItem, QGraphicsLineItem,
                      QGraphicsObject, QLineF,
                      QPainter, QPen, QPointF, QRectF,
                      QStyleOptionGraphicsItem, pyqtSignal)

from camera import Camera
from camera_processing.widgets.button import Button
from camera_processing.widgets.custom_pixmap import CustomPixmap
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from stick import Stick


class StickLink(QGraphicsObject):

    break_link_clicked = pyqtSignal([StickWidget])

    def __init__(self, sw1: StickWidget, sw2: Optional[StickWidget] = None, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.stick1 = sw1
        self.stick2 = sw2
        self.btn_break_link = Button("unlink", "X", self)
        self.btn_break_link.set_base_color("red")
        self.btn_break_link.setVisible(False)
        self.btn_break_link.clicked.connect(lambda: self.break_link_clicked.emit(self.stick1))

        self.color = QColor(0, 255, 0, 255)
        self.line_item = QGraphicsLineItem(0, 0, 0, 0, self)
        self.line_item.setPen(QPen(QColor(0, 255, 0, 255), 2.0))

        self.temporary_target: QPointF = None
        self.setAcceptHoverEvents(False)
        self.set_color(self.color)
        if self.stick2 is not None:
            self.update_line()

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def boundingRect(self):
        return self.line_item.boundingRect().united(self.btn_break_link.boundingRect())
    
    def set_color(self, color: QColor):
        self.color = color
        self.line_item.setPen(QPen(color, 4))
        if self.stick1 is not None:
            self.stick1.set_highlight_color(self.color)
        if self.stick2 is not None:
            self.stick2.set_highlight_color(self.color)
        self.update()
    
    def update_line(self):
        p1 = self.mapFromItem(self.stick1, self.stick1.mid_handle.rect().center())
        p2 = None
        if self.temporary_target is not None:
            p2 = self.mapFromScene(self.temporary_target)
        if p2 is None:
            p2 = self.mapFromItem(self.stick2, self.stick2.mid_handle.rect().center())
        self.line_item.setLine(QLineF(p1, p2))
        self.btn_break_link.setPos(self.line_item.line().center())
        self.update()
    
    def set_temporary_target(self, target: QPointF):
        self.stick2 = None
        self.temporary_target = target
        self.update_line()
    
    def set_target_stick(self, stick_widget: StickWidget):
        self.temporary_target = None
        self.stick2 = stick_widget
        self.update_line()

    def handle_stick_changed(self, sw: StickWidget):
        if sw.stick.id != self.stick1.stick.id:
            return
        self.update_line()


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
        self.stick_links: List[StickLink] = []
        self.current_link_item: StickLink = None
        self.dataset.sticks_linked.connect(self.handle_sticks_linked)
        self.dataset.sticks_unlinked.connect(self.handle_sticks_unlinked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
    
    def boundingRect(self):
        return QRectF(self.scene().views()[0].viewport().rect())

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def handle_stick_widget_link_requested(self, stick: StickWidget):
        self.cancel()
        self.current_link_item = StickLink(stick, parent=self)
        self.update()
    
    def set_target(self, point: QPointF):
        if self.current_link_item is not None:
            self.current_link_item.set_temporary_target(point)
            self.update()

    def accept(self):
        if self.current_link_item is not None:
            if self.current_link_item.stick2 is not None:
                self.dataset.link_sticks(self.current_link_item.stick1.stick, self.current_link_item.stick2.stick)
                self.cancel()

    def cancel(self):
        if self.current_link_item is not None:
            self.scene().removeItem(self.current_link_item)
            self.current_link_item.setEnabled(False)
            self.current_link_item.deleteLater()
            self.current_link_item = None
        self.anchored = False
        self.update()
    
    def anchor_to(self, stick_widget: StickWidget):
        self.target = stick_widget.mapToScene(QPointF(stick_widget.mid_handle.rect().center()))
        self.anchored = True
        self.update()
    
    def unanchor(self):
        self.anchored = False
    
    def handle_stick_widget_hover(self, entered: bool, stick_widget: StickWidget):
        if self.current_link_item is None:
            return
        self.anchored = entered
        if self.anchored:
            self.current_link_item.set_target_stick(stick_widget)
        else:
            self.current_link_item.set_temporary_target(stick_widget.mapToScene(stick_widget.mid_handle.rect().center()))
        self.update()
    
    def set_secondary_cameras(self, cameras: List[CustomPixmap]):
        self.secondary_cameras = cameras
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.hovered.connect(self.handle_stick_widget_hover)
    
    def update_links(self):
        self.links = self.dataset.get_cameras_stick_links(self.camera)
        self.update()
    
    def handle_sticks_linked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        if (self.current_link_item is not None) and False:
            self.stick_links.append(self.current_link_item)
            self.current_link_item = None
        else:
            source = stick1 if stick1.camera_id == self.camera.id else stick2
            target = stick2 if source == stick1 else stick1

            source_sw = next(filter(lambda sw: sw.stick.id == source.id, self.primary_camera.stick_widgets))
            target_sw = None

            for pixmap in self.secondary_cameras:
                if pixmap.camera.id != target.camera_id:
                    continue
                target_sw = next(filter(lambda sw: sw.stick.id == target.id, pixmap.stick_widgets))

            link = StickLink(source_sw, target_sw, self)
            link.btn_break_link.setVisible(True)
            link.break_link_clicked.connect(self.handle_break_link_clicked)
            source_sw.stick_changed.connect(link.handle_stick_changed)
            source_sw.set_is_linked(True)
            target_sw.set_is_linked(True)
            self.stick_links.append(link)
        self.color_stick_links()
        self.cancel()
        
    def handle_sticks_unlinked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        link: StickLink = next(filter(lambda l: l.stick1.stick.id == stick1.id or l.stick2.stick.id == stick1.id, self.stick_links))
        link.stick1.set_is_linked(False)
        link.stick2.set_is_linked(False)
        self.stick_links.remove(link)
        self.scene().removeItem(link)
        link.setEnabled(False)
        link.setParentItem(None)
        link.deleteLater()
        self.color_stick_links()
        self.update()

    def start(self):
        self.setVisible(True)

        for sw in self.primary_camera.stick_widgets:
            sw.set_mode(StickMode.LINK)
            sw.set_available_for_linking(False)
        
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_available_for_linking(True)

    def stop(self):
        self.cancel()
        for sw in self.primary_camera.stick_widgets:
            sw.set_mode(StickMode.DISPLAY)
        
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_available_for_linking(False)
        
        self.setVisible(False)
    
    def handle_break_link_clicked(self, sw: StickWidget):
        self.dataset.unlink_stick(sw.stick)
    
    def color_stick_links(self):
        if len(self.stick_links) == 0:
            return
        hue_step = 360 // len(self.stick_links)

        for i, link in enumerate(self.stick_links):
            link.set_color(QColor.fromHsv(i * hue_step, 255, 255, 150))

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        to_remove = cam2 if cam1.id == self.camera.id else cam1
        self.secondary_cameras = list(filter(lambda pix: pix.camera.id != to_remove.id, self.secondary_cameras))
