import math
from typing import List, Optional, Tuple, Dict

from PyQt5 import QtCore
from PyQt5.Qt import (QColor, QGraphicsItem, QGraphicsLineItem,
                      QGraphicsObject, QLineF,
                      QPainter, QPen, QPointF, QRectF,
                      QStyleOptionGraphicsItem, pyqtSignal)
from PyQt5.QtCore import QMarginsF
from PyQt5.QtWidgets import QGraphicsSceneHoverEvent

from camera import Camera
from camera_processing.widgets.button import Button, ButtonColor
from camera_processing.widgets.camera_view import CameraView
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from stick import Stick


class StickLink(QGraphicsObject):

    break_link_clicked = pyqtSignal([StickWidget])

    def __init__(self, sw1: StickWidget, sw2: Optional[StickWidget] = None, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.stick1 = sw1
        self.stick2 = sw2
        self.btn_break_link = Button("unlink", "Break link", tooltip='', parent=self)
        #self.btn_break_link.set_base_color([ButtonColor.RED])
        #self.btn_break_link.set_custom_color([QColor(50, 50, 50, 200), QColor(200, 0, 0, 200)])
        self.btn_break_link.setVisible(False)
        self.btn_break_link.clicked.connect(lambda: self.break_link_clicked.emit(self.stick1))

        self.color = QColor(0, 255, 0, 255)
        self.line_item = QGraphicsLineItem(0, 0, 0, 0, self)
        self.line_item.setPen(QPen(QColor(0, 255, 0, 255), 1.0))

        self.temporary_target: QPointF = None
        self.setAcceptHoverEvents(False)
        self.set_color(self.color)
        self.setZValue(10)
        if self.stick2 is not None:
            self.update_line()
        self.faded = False

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def boundingRect(self):
        return self.line_item.boundingRect().united(self.btn_break_link.boundingRect())
    
    def set_color(self, color: QColor):
        self.color = color
        self.line_item.setPen(QPen(color, 4))
        #self.btn_break_link.set_custom_color([color])
        if self.stick1 is not None:
            self.stick1.set_frame_color(self.color)
        if self.stick2 is not None:
            self.stick2.set_frame_color(self.color)
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

    def fade_out(self, faded: bool):
        self.faded = faded
        if self.faded:
            color = QColor(200, 200, 200, 100)
        else:
            color = self.color
        pen = self.line_item.pen()
        pen.setColor(color)
        self.line_item.setPen(pen)
        self.update()


class StickLinkManager(QGraphicsObject):

    def __init__(self, dataset: Dataset, camera: Camera, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.dataset = dataset
        self.camera = camera
        self.source: Optional[StickWidget] = None
        self.target: Optional[StickWidget] = None
        self.target_point = QPointF()
        self.anchored = False
        self.primary_camera: CameraView = None
        self.secondary_cameras: List[CameraView] = []
        self.links: List[Tuple[int, int, int]] = []
        #self.stick_links: List[StickLink] = []
        self.stick_links: Dict[int, Tuple[List[StickLink], QColor]] = dict({})
        self.stick_links_list: List[StickLink] = []
        self.unused_colors: List[QColor] = []
        self.current_link_item: StickLink = None
        self.dataset.sticks_linked.connect(self.handle_sticks_linked)
        self.dataset.sticks_unlinked.connect(self.handle_sticks_unlinked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.rect = QRectF()

    def boundingRect(self):
        return self.rect

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def handle_stick_widget_link_requested(self, stick: StickWidget):
        self.cancel()
        self.current_link_item = StickLink(stick, parent=None)
        self.scene().addItem(self.current_link_item)
        self.toggle_highlight_source_sticks(False)
        self.toggle_highlight_target_sticks(True)
        self.update()
    
    def set_target(self, point: QPointF):
        if self.current_link_item is not None:
            self.current_link_item.set_temporary_target(point)
            self.update()

    def accept(self):
        if self.current_link_item is not None:
            if self.current_link_item.stick2 is not None:
                stick1 = self.current_link_item.stick1.stick
                stick2 = self.current_link_item.stick2.stick

                # Unlink stick1 from other sticks to which it might be linked
                self.dataset.unlink_stick_(self.current_link_item.stick1.stick)

                # Also destroy a potential link between stick2 and some other stick from the same
                # camera as stick1
                stick2_view = self.dataset.get_stick_view_from_camera(self.current_link_item.stick2.stick,
                                                                      self.dataset.get_camera(stick1.camera_id))
                if stick2_view is not None:
                    self.dataset.unlink_sticks(stick2, stick2_view)
                self.dataset.link_sticks_(self.current_link_item.stick1.stick, self.current_link_item.stick2.stick)
                self.cancel()

    def cancel(self):
        if self.current_link_item is not None:
            self.scene().removeItem(self.current_link_item)
            self.current_link_item.setEnabled(False)
            self.current_link_item.deleteLater()
            self.current_link_item = None
            self.toggle_highlight_target_sticks(False)
            self.toggle_highlight_source_sticks(True)
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
    
    def set_secondary_cameras(self, cameras: List[CameraView]):
        for cam in self.secondary_cameras:
            cam.stick_widgets_out_of_sync.disconnect(self.handle_stick_widgets_out_of_sync)
        self.secondary_cameras = cameras
        for cam in self.secondary_cameras:
            cam.stick_widgets_out_of_sync.connect(self.handle_stick_widgets_out_of_sync)
            for sw in cam.stick_widgets:
                sw.hovered.connect(self.handle_stick_widget_hover)
    
    def update_links(self):
        self.links = self.dataset.get_cameras_stick_links(self.camera)
        self.update()
    
    def handle_sticks_linked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        if (self.current_link_item is not None) and False: #TODO remove this, I guess
            #self.stick_links.append(self.current_link_item)
            self.current_link_item = None
        else:
            source = stick2 if stick1.camera_id == self.camera.id else stick1
            target = stick2 if source.id == stick1.id else stick1

            source_sw = None
            for pixmap in self.secondary_cameras:
                if pixmap.camera.id != source.camera_id:
                    continue
                source_sw = next(filter(lambda sw: sw.stick.id == source.id, pixmap.stick_widgets))

            if source_sw is None:
                return
            #target_sw = None

            target_sw = next(filter(lambda sw: sw.stick.id == target.id, self.primary_camera.stick_widgets))

            if target.id not in self.stick_links:
                color = self.get_new_link_group_color()
                self.stick_links[target.id] = ([], color)

            links, color = self.stick_links[target.id]

            link = StickLink(source_sw, target_sw, None)
            self.scene().addItem(link)
            #link.btn_break_link.setVisible(True)
            link.set_color(color)
            link.break_link_clicked.connect(self.handle_break_link_clicked)
            source_sw.stick_changed.connect(link.handle_stick_changed)
            source_sw.set_is_linked(True)
            target_sw.set_is_linked(True)
            #self.stick_links.append(link)
            links.append(link)
            self.stick_links[target.id] = (links, color)
            self.stick_links_list.append(link)
            link.setVisible(self.isVisible())
        #self.color_stick_links()
        self.cancel()
        
    def handle_sticks_unlinked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        primary_stick = stick1 if stick1.camera_id == self.camera.id else stick2

        # Check whether we are handling event that is relevant to this StickLinkManager, if not, return
        # A camera configuration such as this can occur:
        # M3 <- M4 -> M1
        # symmetrically we would also have on other camera tabs:
        #       M3 -> M4
        # M4 <- M1
        # And on tab for M4 we would connect some stick from M4 to a stick from both M3 and M1.
        # In Dataset this is represented as equivalence between those three sticks, because obviously it is the same
        # stick in real world seen from three cameras.
        # When we break any of the stick links now, say between M3 and M4, all three tabs M1, M3 and M4 will receive
        # signal that the two sticks were unlinked. Obviously, this event is relevant only to M3 and M4 though, so M1
        # should exit, which happens in the IF that follows.
        if primary_stick.id not in self.stick_links:
            return
        secondary_stick = stick2 if primary_stick.id == stick1.id else stick1
        link_list, color = self.stick_links[primary_stick.id]
        #link_list: List[StickLink] = list(filter(lambda l: l.stick1.stick.id == stick1.id or l.stick2.stick.id == stick1.id, self.stick_links))
        link = list(filter(lambda l: l.stick1.stick.id == secondary_stick.id, link_list))
        if len(link) == 0:
            return
        link = link[0]

        if link.stick1.stick.id == secondary_stick.id:
            link.stick1.set_is_linked(False)
        else:
            link.stick2.set_is_linked(False)

        link_list = list(filter(lambda l: l.stick1.stick.id != secondary_stick.id and l.stick2.stick.id != secondary_stick.id, link_list))
        if len(link_list) == 0:
            if link.stick1.stick.id == primary_stick.id:
                link.stick1.set_is_linked(False)
            else:
                link.stick2.set_is_linked(False)

            link.stick2.set_is_linked(False)
            del self.stick_links[primary_stick.id]
            self.unused_colors.append(color)
        else:
            self.stick_links[primary_stick.id] = (link_list, color)
        self.scene().removeItem(link)
        self.stick_links_list.remove(link)
        link.setEnabled(False)
        link.setParentItem(None)
        link.deleteLater()
        self.update()

    def start(self):
        self.setVisible(True)
        self.setAcceptHoverEvents(True)
        for link in self.stick_links_list:
            link.setVisible(True)

        for sw in self.primary_camera.stick_widgets:
            #sw.set_mode(StickMode.LINK)
            sw.hovered.connect(self.handle_stick_widget_hover)
            sw.set_available_for_linking(True)
            #sw.highlight(QColor(0, 255, 50, 200), animated=True)

        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_mode(StickMode.LINK)
                sw.set_available_for_linking(False)
                sw.set_is_link_source(True)
                #sw.highlight(None)
                #sw.highlight(QColor(0, 255, 50, 200), animated=True)
        self.toggle_highlight_source_sticks(True)

    def stop(self):
        self.cancel()
        self.setAcceptHoverEvents(False)
        for sw in self.primary_camera.stick_widgets:
            sw.set_available_for_linking(False)
            #sw.set_mode(StickMode.DISPLAY)
            #sw.highlight(None)
            sw.hovered.disconnect(self.handle_stick_widget_hover)

        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_mode(StickMode.DISPLAY)
                sw.set_available_for_linking(False)
                sw.set_is_link_source(False)
        self.toggle_highlight_target_sticks(False)
        self.toggle_highlight_source_sticks(False)

        for link in self.stick_links_list:
            link.setVisible(False)
        self.setVisible(False)
    
    def handle_break_link_clicked(self, sw: StickWidget):
        self.dataset.unlink_stick_(sw.stick)
    
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

    def handle_stick_widgets_out_of_sync(self, cp: CameraView):
        for sw in cp.stick_widgets:
            sw.hovered.connect(self.handle_stick_widget_hover)

    def get_new_link_group_color(self) -> QColor:
        if len(self.unused_colors) > 0:
            return self.unused_colors.pop()
        num_groups = len(self.stick_links.keys())
        step = 60
        offset = 0 if num_groups < 6 else step / ((num_groups + 6) // 6)
        hue = (num_groups % 6) * step + offset
        return QColor.fromHsvF(hue / 360.0, 1.0, 1.0, 1.0)

    def toggle_highlight_source_sticks(self, highlight: bool):
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def toggle_highlight_target_sticks(self, highlight: bool):
        for sw in self.primary_camera.stick_widgets:
            sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        found: List[Tuple[float, StickLink, float]] = []
        for stick_links, _ in self.stick_links.values():
            for stick_link in stick_links:
                stick_link.btn_break_link.setVisible(False)
                stick_link.fade_out(False)
                e1 = stick_link.line_item.line().unitVector()
                e1 = e1.p2() - e1.p1()
                e2 = stick_link.line_item.line().normalVector().unitVector()
                e2 = e2.p2() - e2.p1()
                sp = event.scenePos() - stick_link.line_item.line().p1()
                lx = e1.x() * sp.x() + e2.x() * sp.y()
                ly = e1.y() * sp.x() + e2.y() * sp.y()
                if 0 < lx < stick_link.line_item.line().length() and abs(ly) < 50:
                    t = (lx - 0.5 * stick_link.btn_break_link.boundingRect().width()) / stick_link.line_item.line().length()
                    found.append((abs(ly), stick_link, t))

        if len(found) > 0:
            found.sort(key=lambda tup: tup[0])
            dist_stick_link_t = found[0]
            stick_link = dist_stick_link_t[1]
            stick_link.fade_out(False)
            t = dist_stick_link_t[2]
            q = stick_link.line_item.line().pointAt(t)
            #stick_link.btn_break_link.setTransformOriginPoint(stick_link.btn_break_link.boundingRect().center())
            stick_link.btn_break_link.setPos(q)
            stick_link.btn_break_link.setVisible(True)
            for stick_links, _ in self.stick_links.values():
                for stick_link_ in stick_links:
                    if stick_link_ == stick_link:
                        continue
                    stick_link_.fade_out(True)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        super().hoverLeaveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        super().hoverEnterEvent(event)

    def set_rect(self, rect: QRectF):
        self.setPos(rect.topLeft())
        self.rect = rect
        self.rect.moveTo(0.0, 0.0)
        self.update()


