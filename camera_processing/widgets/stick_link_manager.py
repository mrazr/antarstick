import math
from typing import List, Optional, Tuple, Dict

from PyQt5.Qt import (QColor, QGraphicsItem, QGraphicsLineItem,
                      QGraphicsObject, QLineF,
                      QPainter, QPen, QPointF, QRectF,
                      QStyleOptionGraphicsItem, pyqtSignal)
from PyQt5.QtWidgets import QGraphicsSceneHoverEvent

from camera import Camera
from camera_processing.widgets.button import Button
from camera_processing.widgets.camera_view import CameraView
from camera_processing.widgets.split_view import SplitView
from camera_processing.widgets.stick_widget import StickMode, StickWidget
from dataset import Dataset
from stick import Stick


class StickLink(QGraphicsObject):

    break_link_clicked = pyqtSignal('PyQt_PyObject')

    def __init__(self, sw1: StickWidget, sw2: Optional[StickWidget] = None, parent: Optional[QGraphicsItem] = None):
        QGraphicsObject.__init__(self, parent)
        self.stick1 = sw1
        self.stick2 = sw2
        #self.btn_break_link.set_base_color([ButtonColor.RED])
        #self.btn_break_link.set_custom_color([QColor(50, 50, 50, 200), QColor(200, 0, 0, 200)])

        self.color = QColor(0, 255, 0, 255)
        self.line_item = QGraphicsLineItem(0, 0, 0, 0, self)
        self.line_item.setPen(QPen(QColor(0, 255, 0, 255), 1.0))
        self.btn_break_link = Button("unlink", "Break link", tooltip='', parent=self.line_item)
        self.btn_break_link.setVisible(False)
        #self.btn_break_link.clicked.connect(lambda: self.break_link_clicked.emit(self.stick1))
        self.btn_break_link.clicked.connect(self.handle_break_link_clicked)

        self.temporary_target: QPointF = None
        self.setAcceptHoverEvents(False)
        self.set_color(self.color)
        self.setZValue(10)
        if self.stick2 is not None:
            self.update_line()
        self.faded = False
        self.stick1.stick_changed.connect(self.handle_stick_changed)

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def boundingRect(self):
        return self.line_item.boundingRect().united(self.btn_break_link.boundingRect())

    def handle_break_link_clicked(self):
        self.break_link_clicked.emit(self)
    
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
        if self.stick2 is not None:
            self.stick2.set_frame_color(self.color)
        self.stick2.stick_changed.connect(self.handle_stick_changed)
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

    def remove_sticks(self):
        self.stick1.stick_changed.disconnect(self.handle_stick_changed)
        self.stick2.stick_changed.disconnect(self.handle_stick_changed)

    def __eq__(self, other):
        if other is None:
            return False
        if self.stick1.stick == other.stick1.stick:
            selfstick2 = self.stick2 if self.stick2 is None else self.stick2.stick
            otherstick2 = other.stick2 if other.stick2 is None else other.stick2.stick
            return selfstick2 == otherstick2
        return False


class StickLinkManager(QGraphicsObject):

    def __init__(self, parent: QGraphicsItem = None):
        QGraphicsObject.__init__(self, parent)
        self.source: Optional[StickWidget] = None
        self.target: Optional[StickWidget] = None
        self.target_point = QPointF()
        self.anchored = False
        self.unused_colors: List[QColor] = []
        self.current_link_item: StickLink = None
        self.stick_links_list: List[StickLink] = []
        self.rect = QRectF()
        self.color_links: Dict[QColor, List[StickLink]] = {}

    def boundingRect(self):
        return self.rect

    def paint(self, painter: QPainter, options: QStyleOptionGraphicsItem, widget=None):
        pass

    def set_target(self, point: QPointF):
        if not self.anchored:
            if self.current_link_item is not None:
                self.current_link_item.set_temporary_target(point)
                self.update()

    def cancel(self):
        if self.current_link_item is not None:
            #self.stick_links_list.remove(self.current_link_item)
            self.stick_links_list.pop()
            self.scene().removeItem(self.current_link_item)
            self.current_link_item.setEnabled(False)
            self.current_link_item.deleteLater()
            self.current_link_item = None
            self.toggle_highlight_target_sticks(False)
            self.toggle_highlight_source_sticks(True)
        self.anchored = False
        self.update()

    def confirm(self) -> Optional[StickLink]:
        if self.current_link_item is not None:
            if self.current_link_item.stick2 is not None:
                link = self.current_link_item
                self.current_link_item = None
                return link
        return None

    def set_target_stick(self, entered: bool, stick_widget: StickWidget) -> StickLink:
        if self.current_link_item is None or stick_widget.link_source:
            return None
        self.anchored = entered
        if self.anchored:
            self.current_link_item.set_target_stick(stick_widget)
        else:
            self.current_link_item.set_temporary_target(
                stick_widget.mapToScene(stick_widget.mid_handle.rect().center()))
        self.update()
        return self.current_link_item

    def set_source_stick(self, sw: StickWidget) -> StickLink:
        self.current_link_item = StickLink(sw, parent=None)
        self.stick_links_list.append(self.current_link_item)
        color = self.get_new_link_group_color()
        self.current_link_item.set_color(color)
        a = self.color_links.setdefault(color.hue(), [])
        a.append(self.current_link_item)
        self.scene().addItem(self.current_link_item)

        return self.current_link_item

    def get_new_link_group_color(self) -> QColor:
        if len(self.unused_colors) > 0:
            return self.unused_colors.pop()
        num_groups = len(self.stick_links_list)
        step = 60
        offset = 0 if num_groups < 6 else step / ((num_groups + 6) // 6)
        hue = int((num_groups % 6) * step + offset)
        #return QColor.fromHsvF(hue / 360.0, 1.0, 1.0, 1.0)
        return QColor.fromHsv(hue, 255, 255, 255)

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self.current_link_item is not None:
            return
        found: List[Tuple[float, StickLink, float]] = []
        for stick_link in self.stick_links_list:
            if stick_link == self.current_link_item:
                continue
            stick_link.btn_break_link.setVisible(False)
            stick_link.fade_out(False)
            e1 = stick_link.line_item.line().unitVector()
            e1 = e1.p2() - e1.p1()
            e2 = stick_link.line_item.line().normalVector().unitVector()
            e2 = e2.p2() - e2.p1()
            sp = event.scenePos() - stick_link.line_item.line().p1()
            proj = QPointF.dotProduct(sp, e1) * e1
            perp = sp - proj
            dist = math.sqrt(QPointF.dotProduct(perp, perp))
            if dist < 50:
                t = math.sqrt(QPointF.dotProduct(proj, proj)) / stick_link.line_item.line().length()
                if 0.1 < t < 0.9 and QPointF.dotProduct(proj, e1) > 0.0:
                    found.append((dist, stick_link, t))

        if len(found) > 0:
            found.sort(key=lambda tup: tup[0])
            dist_stick_link_t = found[0]
            stick_link = dist_stick_link_t[1]
            stick_link.fade_out(False)
            t = dist_stick_link_t[2]
            p = stick_link.line_item.line().pointAt(t)
            q = stick_link.line_item.mapFromScene(p)
            stick_link.btn_break_link.setPos(q - QPointF(stick_link.btn_break_link.boundingRect().width() * 0.5,
                                                         stick_link.btn_break_link.boundingRect().height() * 0.5))
            stick_link.btn_break_link.setVisible(True)
            for stick_link_ in self.stick_links_list:
                if stick_link_ == stick_link:
                    continue
                stick_link_.fade_out(True)

    def remove_link(self, link: StickLink):
        if link in self.stick_links_list:
            if link == self.current_link_item:
                self.current_link_item = None
            j = self.color_links[link.color.hue()]
            j.remove(link)
            if len(j) == 0:
                self.unused_colors.append(link.color)
            self.stick_links_list.remove(link)
            link.remove_sticks()
            self.scene().removeItem(link)
            link.setEnabled(False)
            link.deleteLater()

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        super().hoverLeaveEvent(event)

    def hoverEnterEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        super().hoverEnterEvent(event)

    def set_rect(self, rect: QRectF):
        self.setPos(rect.topLeft())
        self.rect = rect
        self.rect.moveTo(0.0, 0.0)
        self.update()

    def change_link_color(self, link: StickLink, color: QColor):
        j = self.color_links.get(link.color.hue(), [])
        if len(j) > 0:
            j.remove(link)
        link.set_color(color)
        k = self.color_links.setdefault(color.hue(), [])
        k.append(link)

    def remove_all_links(self):
        lk = self.stick_links_list.copy()
        for link in lk:
            self.remove_link(link)
        lk.clear()

    def hide_links(self):
        for link in self.stick_links_list:
            link.setVisible(False)
        self.setAcceptHoverEvents(False)

    def show_links(self):
        for link in self.stick_links_list:
            link.setVisible(True)
        self.setAcceptHoverEvents(True)


class StickLinkingStrategy:
    def __init__(self, view: StickLinkManager):
        self.view = view

    def accept(self):
        pass

    def cancel(self):
        pass

    def set_secondary_cameras(self, cameras: List[CameraView]):
        pass

    def update_links(self):
        pass

    def handle_sticks_linked(self, stick1: Stick, stick2: Stick):
        pass

    def handle_sticks_unlinked(self, stick1: Stick, stick2: Stick):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def handle_break_link_clicked(self, sw: StickWidget):
        pass

    #def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
    #    pass

    def handle_stick_widgets_out_of_sync(self, cam: CameraView):
        pass

    def set_manager(self, manager):
        pass

    def highlight_source_sticks(self, highlight: bool):
        pass

    def highlight_target_sticks(self, highlight: bool):
        pass

    def handle_stick_widget_link_requested(self, stick: StickWidget):
        self.cancel()
        self.view.anchored = False
        self.view.set_source_stick(stick)
        self.highlight_source_sticks(False)
        self.highlight_target_sticks(True)
        self.view.update()

    def handle_stick_widget_hovered(self, entered: bool, stick_widget: StickWidget):
        self.view.set_target_stick(entered, stick_widget)

    def set_target(self, pos: QPointF):
        self.view.set_target(pos)


class CameraToCameraStickLinkingStrategy(StickLinkingStrategy):

    def __init__(self, dataset: Dataset, camera: Camera, view: StickLinkManager):
        self.view: StickLinkManager = view
        self.dataset = dataset
        self.camera = camera

        self.primary_camera: CameraView = None
        self.secondary_cameras: List[CameraView] = []

        self.links: List[Tuple[int, int, int]] = []
        self.stick_links: Dict[int, Tuple[List[StickLink], QColor]] = dict({})
        self.stick_links_list: List[StickLink] = []

        self.dataset.sticks_linked.connect(self.handle_sticks_linked)
        self.dataset.sticks_unlinked.connect(self.handle_sticks_unlinked)
        self.dataset.cameras_unlinked.connect(self.handle_cameras_unlinked)
        self.is_started = False

    def accept(self):
        if self.view.current_link_item is not None:
            if self.view.current_link_item.stick2 is not None:
                stick1 = self.view.current_link_item.stick1.stick
                stick2 = self.view.current_link_item.stick2.stick
                self.dataset.unlink_stick_(self.view.current_link_item.stick1.stick)

                # Also destroy a potential link between stick2 and some other stick from the same
                # camera as stick1
                stick2_view = self.dataset.get_stick_view_from_camera(self.view.current_link_item.stick2.stick,
                                                                      self.dataset.get_camera(stick1.camera_id))
                if stick2_view is not None:
                    self.dataset.unlink_sticks(stick2, stick2_view)
                self.dataset.link_sticks_(self.view.current_link_item.stick1.stick, self.view.current_link_item.stick2.stick)
                self.cancel()

    def cancel(self):
        if self.view.current_link_item is not None:
            #self.view.scene().removeItem(self.view.current_link_item)
            #self.view.current_link_item.setEnabled(False)
            #self.view.current_link_item.deleteLater()
            #self.view.current_link_item = None
            self.view.remove_link(self.view.current_link_item)
        self.highlight_target_sticks(False)
        self.highlight_source_sticks(True)
        self.view.anchored = False
        self.view.update()

    def set_secondary_cameras(self, cameras: List[CameraView]):
        for cam in self.secondary_cameras:
            cam.stick_widgets_out_of_sync.disconnect(self.handle_stick_widgets_out_of_sync)
        self.secondary_cameras = cameras
        for cam in self.secondary_cameras:
            cam.stick_widgets_out_of_sync.connect(self.handle_stick_widgets_out_of_sync)
            for sw in cam.stick_widgets:
                sw.hovered.connect(self.handle_stick_widget_hovered)

    def update_links(self):
        self.links = self.dataset.get_cameras_stick_links(self.camera)

    def handle_sticks_linked(self, stick1: Stick, stick2: Stick):
        if stick1.camera_id != self.camera.id and stick2.camera_id != self.camera.id:
            return
        if (self.view.current_link_item is not None) and False: #TODO remove this, I guess
            #self.view.stick_links.append(self.view.current_link_item)
            self.view.current_link_item = None
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

            #if target.id not in self.stick_links:
            #    color = self.view.get_new_link_group_color()
            #    self.stick_links[target.id] = ([], color)
            #    #self.view.stick_links[target.id] = ([], color)

            #links, color = self.view.stick_links[target.id]

            if self.view.current_link_item is not None:
                link = self.view.confirm()
            else:
                link = self.view.set_source_stick(source_sw)
                self.view.set_target_stick(True, target_sw)
                self.view.confirm()
            links, color = self.stick_links.setdefault(target.id, ([], link.color))
            #self.view.scene().addItem(link)
            #link.btn_break_link.setVisible(True)
            #link.set_color(color)
            self.view.change_link_color(link, color)
            link.break_link_clicked.connect(self.handle_break_link_clicked)
            #source_sw.stick_changed.connect(link.handle_stick_changed)
            source_sw.set_is_linked(True)
            target_sw.set_is_linked(True)
            #self.view.stick_links.append(link)
            links.append(link)
            #self.view.stick_links[target.id] = (links, color)
            #self.view.stick_links_list.append(link)
            self.stick_links[target.id] = (links, color)
            #self.view.stick_links_list.append(link)
            link.setVisible(self.view.isVisible())
        #self.color_stick_links()
        if self.is_started:
            self.init_state()
        #self.cancel()

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
        #if primary_stick.id not in self.manager.stick_links:
        #    return
        if primary_stick.id not in self.stick_links:
            return
        secondary_stick = stick2 if primary_stick.id == stick1.id else stick1
        #link_list, color = self.manager.stick_links[primary_stick.id]
        link_list, color = self.stick_links[primary_stick.id]
        #link_list: List[StickLink] = list(filter(lambda l: l.stick1.stick.id == stick1.id or l.stick2.stick.id == stick1.id, self.manager.stick_links))
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
            #del self.manager.stick_links[primary_stick.id]
            del self.stick_links[primary_stick.id]
            self.view.unused_colors.append(color)
        else:
            #self.view.stick_links[primary_stick.id] = (link_list, color)
            self.stick_links[primary_stick.id] = (link_list, color)
        #self.view.scene().removeItem(link)
        #self.stick_links_list.remove(link)
        #self.view.stick_links_list.remove(link)
        #link.setEnabled(False)
        #link.setParentItem(None)
        #link.deleteLater()
        self.view.remove_link(link)
        self.view.update()
        if self.is_started:
            self.init_state()

    def start(self):
        self.is_started = True
        self.view.setVisible(True)
        self.view.setAcceptHoverEvents(True)
        for link in self.view.stick_links_list:
            link.setVisible(True)

        for sw in self.primary_camera.stick_widgets:
            sw.hovered.connect(self.handle_stick_widget_hovered)

        self.init_state()

    def init_state(self):
        for sw in self.primary_camera.stick_widgets:
            sw.set_mode(StickMode.LinkTarget)

        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.set_mode(StickMode.LinkSource)
        self.highlight_source_sticks(True)

    def stop(self):
        self.is_started = False
        self.cancel()
        self.view.setAcceptHoverEvents(False)
        for sw in self.primary_camera.stick_widgets:
            sw.set_available_for_linking(False)
            #sw.set_mode(StickMode.DISPLAY)
            #sw.highlight(None)
            sw.hovered.disconnect(self.handle_stick_widget_hovered)

        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                #sw.set_mode(StickMode.Display)
                #sw.set_available_for_linking(False)
                #sw.set_is_link_source(False)
                sw.set_mode(StickMode.LinkSource)
        self.highlight_target_sticks(False)
        self.highlight_source_sticks(False)

        for link in self.view.stick_links_list:
            link.setVisible(False)
        self.view.setVisible(False)

    def handle_break_link_clicked(self, link: StickLink):
        self.dataset.unlink_stick_(link.stick1.stick)

    def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
        if cam1.id != self.camera.id and cam2.id != self.camera.id:
            return
        to_remove = cam2 if cam1.id == self.camera.id else cam1
        self.secondary_cameras = list(filter(lambda pix: pix.camera.id != to_remove.id, self.secondary_cameras))

    def handle_stick_widgets_out_of_sync(self, cam: CameraView):
        for sw in cam.stick_widgets:
            sw.hovered.connect(self.view.handle_stick_widget_hover)

    def highlight_source_sticks(self, highlight):
        for cam in self.secondary_cameras:
            for sw in cam.stick_widgets:
                sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def highlight_target_sticks(self, highlight: bool):
        for sw in self.primary_camera.stick_widgets:
            sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def handle_stick_widget_link_requested(self, stick: StickWidget):
        super().handle_stick_widget_link_requested(stick)
        #self.cancel()
        #self.view.current_link_item = StickLink(stick, parent=None)
        #self.view.scene().addItem(self.current_link_item)
        #self.highlight_source_sticks(False)
        #self.highlight_target_sticks(True)
        #self.view.update()

    def hide_links_from_camera(self, camera: Camera):
        for link in self.view.stick_links_list:
            if link.stick1.stick.camera_id == camera.id or link.stick2.stick.camera_id == camera.id:
                link.setVisible(False)

    def show_links_from_camera(self, camera: Camera):
        for link in self.view.stick_links_list:
            if link.stick1.stick.camera_id == camera.id or link.stick2.stick.camera_id == camera.id:
                link.setVisible(True)


class MovedSticksLinkingStrategy(StickLinkingStrategy):
    def __init__(self, split_view: Optional[SplitView] = None):
        self.split_view = split_view
        self.view: Optional[StickLinkManager] = None
        self.stick_stick_links: Dict[StickWidget, StickLink] = {}

    def accept(self):
        if self.view.current_link_item is None:
            return
        if self.view.anchored:
            link = self.view.current_link_item
            link1 = self.stick_stick_links.get(link.stick1, None)
            link2 = self.stick_stick_links.get(link.stick2, None)
            if link1 is not None:
                self.view.remove_link(link1)
                del self.stick_stick_links[link1.stick1]
                del self.stick_stick_links[link1.stick2]
            if link2 is not None:
                self.view.remove_link(link2)
                del self.stick_stick_links[link2.stick1]
                del self.stick_stick_links[link2.stick2]
            self.view.confirm()
            self.stick_stick_links[link.stick1] = link
            self.stick_stick_links[link.stick2] = link
        else:
            self.view.remove_link(self.view.current_link_item)
        for sw in self.split_view.source_widgets:
            sw.set_available_for_linking(False)
        for sw in self.split_view.target_widgets:
            sw.set_available_for_linking(False)
        #self.highlight_target_sticks(True)
        #self.highlight_source_sticks(True)
        self.start()
        return
        stick1 = self.manager.current_link_item.stick1.stick
        stick2 = self.manager.current_link_item.stick2.stick
        self.dataset.unlink_stick_(self.manager.current_link_item.stick1.stick)

        # Also destroy a potential link between stick2 and some other stick from the same
        # camera as stick1
        stick2_view = self.dataset.get_stick_view_from_camera(self.manager.current_link_item.stick2.stick,
                                                              self.dataset.get_camera(stick1.camera_id))
        if stick2_view is not None:
            self.dataset.unlink_sticks(stick2, stick2_view)
        self.dataset.link_sticks_(self.manager.current_link_item.stick1.stick, self.manager.current_link_item.stick2.stick)
        self.cancel()

    def cancel(self):
        super().cancel()

    def set_secondary_cameras(self, cameras: List[CameraView]):
        #super().set_secondary_cameras(cameras)
        pass

    def update_links(self):
        #super().update_links()
        pass

    def handle_sticks_linked(self, stick1: Stick, stick2: Stick):
        #super().handle_sticks_linked(stick1, stick2)
        pass

    def handle_sticks_unlinked(self, stick1: Stick, stick2: Stick):
        #super().handle_sticks_unlinked(stick1, stick2)
        pass

    def start(self):
        self.view.setVisible(True)
        self.view.setAcceptHoverEvents(True)
        for link in self.view.stick_links_list:
            link.setVisible(True)

        for sw in self.split_view.source_widgets:
            sw.hovered.connect(self.handle_stick_widget_hovered)
            sw.set_mode(StickMode.LinkSource)
            #sw.set_available_for_linking(True)
            #sw.set_is_link_source(True)

        for sw in self.split_view.target_widgets:
            sw.hovered.connect(self.handle_stick_widget_hovered)
            #sw.set_mode(StickMode.LinkSource)
            sw.set_mode(StickMode.LinkTarget)
            #sw.set_mode(StickMode.LINK)
            #sw.set_available_for_linking(True)
            #sw.set_is_link_source(True)
        #self.highlight_source_sticks(True)
        #self.highlight_target_sticks(True)

    def stop(self):
        #super().stop()
        pass

    def handle_break_link_clicked(self, link: StickLink):
        self.view.remove_link(link)

    #def handle_cameras_unlinked(self, cam1: Camera, cam2: Camera):
    #    super().handle_cameras_unlinked(cam1, cam2)

    def handle_stick_widgets_out_of_sync(self, cam: CameraView):
        #super().handle_stick_widgets_out_of_sync(cam)
        pass

    def set_view(self, view: StickLinkManager):
        self.view = view

    def highlight_source_sticks(self, highlight: bool):
        for sw in self.split_view.source_widgets:
            sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def highlight_target_sticks(self, highlight: bool):
        for sw in self.split_view.target_widgets:
            sw.highlight(QColor(250, 125, 0, 200) if highlight else None, animated=True)

    def set_split_view(self, split_view: SplitView):
        self.split_view = split_view
        for source_sw in self.split_view.source_widgets:
            target_sw = next(filter(lambda sw: sw.stick == source_sw.stick, self.split_view.target_widgets))
            source_sw.link_initiated.connect(self.handle_stick_widget_link_requested)
            target_sw.link_initiated.connect(self.handle_stick_widget_link_requested)
            #source_sw.set_is_link_source(False)
            #target_sw.set_is_link_source(False)
            source_sw.set_mode(StickMode.Display)
            target_sw.set_mode(StickMode.Display)
            self.view.set_source_stick(source_sw)
            self.view.set_target_stick(True, target_sw)
            self.accept()

    def handle_stick_widget_link_requested(self, stick: StickWidget):
        super().handle_stick_widget_link_requested(stick)
        if stick in self.split_view.source_widgets:
            #self.highlight_source_sticks(False)
            #self.highlight_target_sticks(True)
            #for sw in self.split_view.target_widgets:
            #    #sw.set_available_for_linking(True)
            #    #sw.set_is_link_source(False)
            #    sw.set_mode(StickMode.LinkTarget)
            for sw in self.split_view.source_widgets:
                #sw.set_is_link_source(False)
                sw.set_mode(StickMode.Display)
        else:
            #self.highlight_source_sticks(True)
            #self.highlight_target_sticks(False)
            for sw in self.split_view.source_widgets:
                sw.set_mode(StickMode.LinkTarget)
                #sw.set_available_for_linking(True)
                #sw.set_is_link_source(False)
            for sw in self.split_view.target_widgets:
                #sw.set_is_link_source(False)
                sw.set_mode(StickMode.Display)

    def reset(self):
        self.stick_stick_links.clear()
        self.view.remove_all_links()

