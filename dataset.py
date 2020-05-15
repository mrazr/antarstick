from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import jsonpickle
from numpy import zeros
from PyQt5.QtCore import QObject
from PyQt5.QtCore import pyqtSignal as Signal

from camera import Camera
from stick import Stick


class Dataset(QObject):
    """Class for representing dataset, where a dataset is considered a set of cameras.
    ...

    Attributes
    ----------
    path : Path
        absolute path to the JSON file representing this Dataset
    stick_views_map : Dict[StickID, Tuple[CameraID, OtherCameraID, OtherCamera_StickID]]
        mapping that relates multiple camera views of one particular stick in real world. StickID is the id of a Stick that belongs to
        a Camera identified by `CameraID`. `OtherCamera_StickID` is a view of the same stick but seen from the Camera identified by
        `OtherCameraID`.
        If a particular real world stick is seen only on one Camera, then StickID returns None.
    cameras : List[Camera]
        list of cameras that form this Dataset
    next_camera_id : int
        next global(within this dataset) id to be assigned to
        a newly added Camera

    Methods
    -------
    add_camera(folder: Path)
        Adds a new Camera which is comprised of photos in the
        folder `folder`.
    remove_camera(camera_id: int)
        Removes a camera with identified by the camera id `camera_id`.
    save() -> bool
        Saves the state of this dataset along with
        the measurements in its cameras.
        Returns True if successful
    save_as(path: Path) -> bool
        Saves the state in a new file given by the `path` parameter.
        Returns True if successful
    load_from(path: Path) -> bool
        Loads to self the dataset specified by `path`.
        Returns True if successful
    create_new_stick() -> Stick
        Creates a new zero valued Stick. Stick creation is handled
        by this method so as to assure assigning unique global IDs per Dataset.
    create_new_sticks(count: int) -> List[Stick]
        Creates `count` new zero valued Stick-s.

    Signals
    -------
    camera_added : Signal(Camera)
        Emitted when a Camera is added to Dataset. Argument is the Camera
        that was added.
    camera_removed : Signal(int)
        Emitted when a Camera is remove from dataset.
        Argument is the id of the removed camera.
    camera_sticks_detected: Signal(Camera)
        Emitted when Stick-s are detected for a certain Camera
    """

    camera_added = Signal(Camera)
    camera_removed = Signal(Camera)
    camera_sticks_detected = Signal(Camera)
    cameras_linked = Signal([Camera, Camera])
    cameras_unlinked = Signal([Camera, Camera])
    stick_removed = Signal([Stick])
    stick_created = Signal([Stick])
    all_sticks_removed = Signal([Camera])
    new_sticks_created = Signal('PyQt_PyObject')
    sticks_linked = Signal([Stick, Stick])
    sticks_unlinked = Signal([Stick, Stick])
    loading_finished = Signal()

    def __init__(self, path: Optional[Path] = None):
        super(Dataset, self).__init__()
        if path is not None:
            if path.exists():
                self.load_from(path)
        else:
            self.path = Path(".")
        self.stick_views_map: Dict[int, Tuple[int, int, int]] = dict({})
        self.cameras: List[Camera] = []
        self.next_camera_id = 0
        self.next_stick_id = 0
        self.linked_cameras: Set[Tuple[int, int]] = set()
        self.unused_stick_ids: List[int] = []

    def add_camera(self, folder: Path):
        camera = Camera(folder, self.next_camera_id)
        if self.path:
            camera.measurements_path = self.path.parent / f"camera{camera.id}.csv"
        self.cameras.append(camera)
        self.next_camera_id += 1
        camera.stick_removed.connect(self.handle_stick_removed)
        camera.sticks_removed.connect(self.handle_sticks_removed)
        self.camera_added.emit(camera)

    def remove_camera(self, camera_id: int):
        old_camera_count = len(self.cameras)
        camera: Camera = next(filter(lambda cam: cam.id == camera_id, self.cameras))

        camera_links  = list(filter(lambda link: link[0] == camera.id or link[1] == camera.id, self.linked_cameras))

        for link in camera_links:
            other_cam_id = link[1] if link[0] == camera.id else link[0]
            camera2 = next(filter(lambda cam: cam.id == other_cam_id, self.cameras))
            self.unlink_cameras(camera, camera2)

        camera.remove_sticks()
        self.disconnect_camera_signals(camera)
        self.cameras = list(filter(lambda camera: camera.id != camera_id, self.cameras))
        if old_camera_count > len(self.cameras):
            self.camera_removed.emit(camera)

    def save(self) -> bool:
        if self.path == Path("."):
            return False
        try:
            with open(self.path, "w") as output_file:
                for camera in self.cameras:
                    path = self.path.parent / f"camera{camera.id}.csv"
                    camera.save_measurements(path)
                state = self.__dict__.copy()
                state['cameras'] = [camera.get_state() for camera in self.cameras]
                output_file.write(jsonpickle.encode(state))
        except OSError as err:
            print(f"Could not open {self.path} for writing: {err.strerror}")
            return False
        return True

    def save_as(self, path: Path) -> bool:
        self.path = path
        return self.save()

    def load_from(self, path: Path) -> bool:
        try:
            with open(path, "r") as dataset_file:
                decoded = jsonpickle.decode(dataset_file.read())
                self.path = decoded['path']
                stick_views_map = decoded['stick_views_map']
                self.stick_views_map = dict({})
                self.next_camera_id = decoded['next_camera_id']
                linked_cameras = decoded['linked_cameras']
                self.unused_stick_ids = list(map(int, decoded['unused_stick_ids']))
                self.cameras = [Camera.build_from_state(camera_state) for camera_state in decoded['cameras']]
                for camera in self.cameras:
                    self.connect_camera_signals(camera)
                    self.camera_added.emit(camera)

            self.loading_finished.emit()
            for left_cam_id, right_cam_id in linked_cameras:
                left_cam = self.get_camera(left_cam_id)
                right_cam = self.get_camera(right_cam_id)
                self.link_cameras(left_cam, right_cam)

            for k, v in stick_views_map.items():
                s1 = self.get_stick_by_id(int(k))
                s2 = self.get_stick_by_id(v[2])
                self.link_sticks(s1, s2)
            return True
        except OSError:  # TODO do actual handling
            return False

    def create_new_stick(self, camera: Camera) -> Stick:
        stick_id = -1
        if len(self.unused_stick_ids) > 0:
            stick_id = self.unused_stick_ids.pop()
        else:
            stick_id = self.next_stick_id
            self.next_stick_id += 1
        return Stick(stick_id, camera.id, zeros((1, 2)), zeros((1, 2)))

    def create_new_sticks(self, camera: Camera, count: int) -> List[Stick]:
        return list(map(lambda _: self.create_new_stick(camera), range(count)))
    
    def link_cameras(self, cam1: Camera, cam2: Camera):
        self.linked_cameras.add((cam1.id, cam2.id))
        self.cameras_linked.emit(cam1, cam2)
    
    def unlink_cameras(self, cam1: Camera, cam2: Camera):
        link = (cam1.id, cam2.id)
        link2 = (cam2.id, cam1.id)
        self.linked_cameras = set(filter(lambda _link: _link != link and _link != link2, self.linked_cameras))
        for stick in cam1.sticks:
            if stick.id in self.stick_views_map:
                link = self.stick_views_map[stick.id]
                if link[1] == cam2.id:
                    self.unlink_stick(stick)
        self.cameras_unlinked.emit(cam1, cam2)
    
    def link_sticks(self, stick1: Stick, stick2: Stick):
        camera1: Camera = next(filter(lambda cam: cam.id == stick1.camera_id, self.cameras))
        camera2: Camera = next(filter(lambda cam: cam.id == stick2.camera_id, self.cameras))

        if stick1.id in self.stick_views_map:
            self.unlink_stick(stick1)
        if stick2.id in self.stick_views_map:
            self.unlink_stick(stick2)

        self.stick_views_map[stick1.id] = (camera1.id, camera2.id, stick2.id)
        self.stick_views_map[stick2.id] = (camera2.id, camera1.id, stick1.id)

        self.sticks_linked.emit(stick1, stick2)
    
    def unlink_stick(self, stick: Stick):
        link = self.stick_views_map.get(stick.id)
        if link is None:
            return
        stick2_id = link[2]
        camera2: Camera = next(filter(lambda cam: cam.id == link[1], self.cameras))
        stick2 = next(filter(lambda s: s.id == stick2_id, camera2.sticks))

        del self.stick_views_map[stick.id]
        del self.stick_views_map[stick2_id]
        self.sticks_unlinked.emit(stick, stick2)
    
    def get_cameras_stick_links(self, camera: Camera) -> List[Tuple[int, int, int]]:
        camera_links = filter(lambda k_v: k_v[1][0] == camera.id, self.stick_views_map.items())
        return list(map(lambda t: (t[0], t[1][1], t[1][2]), camera_links))
    
    def remove_stick(self, stick: Stick):
        camera: Camera = next(filter(lambda cam: cam.id == stick.camera_id, self.cameras))
        if stick.id in self.stick_views_map:
            self.unlink_stick(stick)
        self.stick_removed.emit(stick)
        
        camera.sticks = list(filter(lambda s: s.id != stick.id, camera.sticks))
        self.unused_stick_ids.append(stick.id)
    
    def get_stick_by_id(self, id: int) -> Optional[Stick]:
        for cam in self.cameras:
            for stick in cam.sticks:
                if stick.id == id:
                    return stick
        return None

    def get_camera(self, id_or_path_str: Union[int, str]) -> Camera:
        if isinstance(id_or_path_str, str):
            return next(filter(lambda cam: str(cam.folder) == id_or_path_str, self.cameras))
        return next(filter(lambda cam: cam.id == id_or_path_str, self.cameras))

    def handle_stick_removed(self, stick: Stick):
        self.unlink_stick(stick)
        self.unused_stick_ids.append(stick.id)

    def handle_sticks_removed(self, sticks: List[Stick]):
        for stick in sticks:
            self.handle_stick_removed(stick)

    def connect_camera_signals(self, camera: Camera):
        camera.stick_removed.connect(self.handle_stick_removed)
        camera.sticks_removed.connect(self.handle_sticks_removed)

    def disconnect_camera_signals(self, camera: Camera):
        camera.stick_removed.disconnect(self.handle_stick_removed)
        camera.sticks_removed.disconnect(self.handle_sticks_removed)
