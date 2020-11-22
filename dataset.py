from pathlib import PurePath, PosixPath, Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any

import jsonpickle
import json
from numpy import zeros
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import QMessageBox

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
        self.path: Path = None
        if path is not None:
            if path.exists():
                self.load_from(path)
        else:
            self.path = Path(".")
        #self.stick_views_map: Dict[int, Tuple[int, int, int]] = dict({})
        self.stick_views_map: Dict[int, List[Stick]] = dict({})
        #self.camera_folders: List[Path] = []
        self.cameras_ids: Dict[str, int] = dict({})
        #self.cameras: Dict[int, Camera] = dict({})
        self.stick_local_to_global_ids: Dict[int, Dict[int, int]] = dict({})  # Camera.id -> Stick.local_id -> Stick.global_id
        self.sticks: Dict[int, Stick] = dict({})
        self.cameras: List[Camera] = []
        self.next_camera_id = 0
        self.next_stick_id = 0
        self.linked_cameras: Set[Tuple[int, int]] = set() #, int]] = set()
        self.unused_stick_ids: List[int] = []
        self.required_state_fields = set(self.get_state().keys())
        #self.camera_groups: Dict[int, List[Camera]] = {}
        #self.available_link_groups = list(range(12))

    def add_camera(self, folder: Path, camera_id: int = -1, first_time_add: bool = True) -> bool:
        if not folder.exists():
            return False
        if not folder.is_absolute():
            camera = Camera.load_from_path(self.path.parent / folder)
        else:
            camera = Camera.load_from_path(folder)
        if first_time_add:
            camera_id = self.next_camera_id
            self.next_camera_id += 1
            #try:
            #    self.cameras_ids[str(folder.relative_to(self.path.parent))] = camera_id
            #except ValueError:
            self.cameras_ids[str(folder)] = camera_id
        if camera is None:  # We didn't find camera.json file in `folder`
            # If this is the first time this dataset is adding this camera, meaning that `dataset.camera_folders`
            # does not contain `folder` we create a brand new camera.json and a corresponding Camera object
            # All future attempts to add this Camera, even when working with other Dataset than this one will result in
            # reading from the file `folder/camera.json` that is now being created.
            if first_time_add:
                camera = Camera(folder, camera_id)
                #if self.path:
                #    camera.measurements_path = self.path.parent / f"camera{camera.id}.csv"
                #self.camera_folders.append(folder)
            else:
                # And this is a case when dataset contains `folder` in `self.camera_folders`, so `folder/camera.json`
                # should exist but it does not. This should then offer the user to locate the `camera.json` file
                # manually through file dialog, or let the application create a new camera.json and possibly re-measure
                # all the photos. I'm going to handle these cases for I guess there might be a time when a user wants to
                # share the measurements with a fellow scientist, so they ideally only share the dataset.json file with
                # the measurements csv files and camera.json files without the actual gigabytes of photos. It might even
                # be that camera.json files will not be necessary for the visualization part.
                # TODO handle case when loading camera referenced in dataset file, and the camera.json is not found
                return False
        #else:
        #    if first_time_add:
        #        self.camera_folders.append(folder)
        #        camera.id = self.next_camera_id
        #        self.next_camera_id += 1
        camera.id = camera_id
        self.cameras.append(camera)
        for stick in camera.sticks:
            stick.camera_id = camera.id
            self.register_stick(stick, camera)
        #camera.stick_removed.connect(self.handle_stick_removed)
        #camera.sticks_removed.connect(self.handle_sticks_removed)
        self.connect_camera_signals(camera)
        self.camera_added.emit(camera)
        return True

    def remove_camera(self, camera_id: int):
        old_camera_count = len(self.cameras)
        camera: Camera = next(filter(lambda cam: cam.id == camera_id, self.cameras))

        camera.remove_sticks()

        del self.cameras_ids[str(camera.folder)]

        camera_links = list(filter(lambda link: link[0] == camera.id or link[1] == camera.id, self.linked_cameras))

        for link in camera_links:
            other_cam_id = link[1] if link[0] == camera.id else link[0]
            camera2 = next(filter(lambda cam: cam.id == other_cam_id, self.cameras))
            self.unlink_cameras(camera, camera2)

        self.disconnect_camera_signals(camera)
        self.cameras = list(filter(lambda camera_: camera_.id != camera_id, self.cameras))
        if old_camera_count > len(self.cameras):
            self.camera_removed.emit(camera)

    def save(self) -> bool:
        for camera in self.cameras:
            # path = self.path.parent / f"camera{camera.id}.csv"
            #camera.save_measurements()
            camera.save()
        if self.path == Path("."):
            return False
        try:
            #with open(self.path, "w") as output_file:
            #    state = self.__dict__.copy()
            #    state['cameras'] = [camera.get_state() for camera in self.cameras]
            #    output_file.write(jsonpickle.encode(state))
            with open(self.path, "w") as output_file:
                #for camera in self.cameras:
                #    camera.save()
                state = self.get_state()
                json.dump(state, output_file, indent=1)
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
                state = json.load(dataset_file)
                if set(state) != self.required_state_fields:
                    msg_box = QMessageBox(QMessageBox.Critical, 'Invalid file',
                                          f'The file {str(path)} is not a valid dataset file.',
                                          QMessageBox.Close)
                    msg_box.exec_()
                    return False
                self.path = Path(state['path'])
                stick_views_map = state['stick_views_map']
                self.next_camera_id = state['next_camera_id']
                linked_cameras = state['linked_cameras'] #[tuple(link) for link in state['linked_cameras']]  #set(state['linked_cameras'])
                self.unused_stick_ids = state['unused_stick_ids']
                self.next_stick_id = state['next_stick_id']
                self.cameras_ids = state['cameras_ids']
                self.stick_local_to_global_ids = dict({}) #state['stick_local_to_global_ids']

                for camera_id, stick_id_map  in state['stick_local_to_global_ids'].items():
                    self.stick_local_to_global_ids[int(camera_id)] = dict({})
                    for local_id, global_id in stick_id_map.items():
                        self.stick_local_to_global_ids[int(camera_id)][int(local_id)] = int(global_id)
                failed_camera_loads: List[Path] = []
                for path_str, cam_id in self.cameras_ids.items():
                    if not self.add_camera(Path(path_str), camera_id=cam_id, first_time_add=False):
                        failed_camera_loads.append(Path(path_str))

                if len(failed_camera_loads) > 0:
                    cams = "".join(list(map(lambda cam: f'{str(cam)},\n', failed_camera_loads)))
                    msg_box = QMessageBox(QMessageBox.Critical, 'Cameras not found',
                                          f'Following cameras could not be opened:\n{cams}',
                                          QMessageBox.Close)
                    msg_box.exec_()
                    return False

            self.loading_finished.emit()
            for left_cam_id, right_cam_id in linked_cameras:
                left_cam = self.get_camera(left_cam_id)
                right_cam = self.get_camera(right_cam_id)
                self.link_cameras(left_cam, right_cam)

            for stick_views in stick_views_map:
                stick1 = self.get_stick_by_id(stick_views[0])
                for stick2_id in stick_views[1:]:
                    stick2 = self.get_stick_by_id(stick2_id)
                    self.link_sticks_(stick2, stick1)

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
        #cam1_group = list(filter(lambda link: link[0] == cam1.id or link[1] == cam1.id, self.linked_cameras))
        #cam2_group = list(filter(lambda link: link[0] == cam2.id or link[1] == cam2.id, self.linked_cameras))
        #cam1_group = -1 if len(cam1_group) == 0 else cam1_group[0][2]
        #cam2_group = -1 if len(cam2_group) == 0 else cam2_group[0][2]
        #camera_group = max(cam1_group, cam2_group)
        #if camera_group < 0:
        #    camera_group = max(list(self.camera_groups.keys()), default=-1) + 1
        #    self.camera_groups[camera_group] = []
        #if cam1 not in self.camera_groups[camera_group]:
        #    self.camera_groups[camera_group].append(cam1)
        #if cam2 not in self.camera_groups[camera_group]:
        #    self.camera_groups[camera_group].append(cam2)
        #camera_group = -1 if len(self.available_link_groups) == 0 else self.available_link_groups[0]
        #if camera_group >= 0:
        #    self.available_link_groups.remove(camera_group)
        #else:
        #    print(f'Reached maximum number of linked cameras')
        self.linked_cameras.add((cam1.id, cam2.id)) #, camera_group))
        self.cameras_linked.emit(cam1, cam2) #, camera_group)
    
    def unlink_cameras(self, cam1: Camera, cam2: Camera):
        link1 = (cam1.id, cam2.id)
        link2 = (cam2.id, cam1.id)
        #link = list(filter(lambda _link: _link == link1 or _link == link2, self.linked_cameras))[0]
        #self.available_link_groups.append(link[2])
        #self.available_link_groups.sort()
        self.linked_cameras = set(filter(lambda _link: _link != link1 and _link != link2, self.linked_cameras))
        for stick in cam1.sticks:
            links = self.stick_views_map.get(stick.id, [])
            for linked_stick in links:
                if linked_stick.camera_id == cam2.id:
                    self.unlink_sticks(stick, linked_stick)
            #if stick.id in self.stick_views_map:
            #    links = self.stick_views_map()
            #    #self.unlink_stick_(stick)
            #    #link = self.stick_views_map[stick.id]
            #    #if link[1] == cam2.id:
            #    #    self.unlink_stick(stick)
        self.cameras_unlinked.emit(cam1, cam2) #, link[2])
    
    #def link_sticks(self, stick1: Stick, stick2: Stick):
    #    #camera1: Camera = next(filter(lambda cam: cam.id == stick1.camera_id, self.cameras))
    #    #camera2: Camera = next(filter(lambda cam: cam.id == stick2.camera_id, self.cameras))

    #    # First destroy possible links between stick1 and some other stick from the same Camera as stick2 and vice versa
    #    stick_view = self.get_stick_view_from_camera(stick1, stick2.camera_id)
    #    if stick_view.id == stick2.id:
    #        return
    #    if stick_view is not None:
    #        self.unlink_sticks(stick1, stick_view)

    #    stick_view = self.get_stick_view_from_camera(stick2, stick1.camera_id)
    #    if stick_view is not None:
    #        self.unlink_sticks(stick2, stick_view)

    #    #if stick1.id in self.stick_views_map:
    #    #    self.unlink_stick(stick1)
    #    #if stick2.id in self.stick_views_map:
    #    #    self.unlink_stick(stick2)

    #    self.stick_views_map[stick1.id] = stick2.id  #(camera1.id, camera2.id, stick2.id)
    #    self.stick_views_map[stick2.id] = stick1.id  #(camera2.id, camera1.id, stick1.id)

    #    self.sticks_linked.emit(stick1, stick2)

    def link_sticks_(self, stick1: Stick, stick2: Stick):
        #self.unlink_stick_(stick1)
        #stick_view = self.get_stick_view_from_camera(stick2, stick1.camera_id)
        #if stick_view is not None:
        #    self.unlink_sticks(stick2, stick_view)
        for stick_view in stick2.stick_views:
            stick_view.stick_views.append(stick1)
            self.stick_views_map[stick_view.id] = stick_view.stick_views
            stick1.stick_views.append(stick_view)
            self.sticks_linked.emit(stick1, stick_view)
        stick2.stick_views.append(stick1)
        stick1.stick_views.append(stick2)
        self.stick_views_map[stick2.id] = stick2.stick_views
        self.stick_views_map[stick1.id] = stick1.stick_views
        self.sticks_linked.emit(stick1, stick2)

    #def unlink_stick(self, stick: Stick):
    #    link = self.stick_views_map.get(stick.id)
    #    if link is None or len(link) == 0:
    #        return
    #    stick2 = link[0]
    #    #camera2: Camera = next(filter(lambda cam: cam.id == stick2_id.camera, self.cameras))
    #    #stick2 = self.get_stick_by_id(stick2_id)

    #    del self.stick_views_map[stick.id]
    #    del self.stick_views_map[stick2.id]
    #    self.sticks_unlinked.emit(stick, stick2)

    def unlink_sticks(self, stick1: Stick, stick2: Stick):
        stick1.stick_views = list(filter(lambda sw: sw.id != stick2.id, stick1.stick_views))
        stick2.stick_views = list(filter(lambda sw: sw.id != stick1.id, stick2.stick_views))

        if len(stick1.stick_views) == 0:
            del self.stick_views_map[stick1.id]
        else:
            self.stick_views_map[stick1.id] = stick1.stick_views

        if len(stick2.stick_views) == 0:
            del self.stick_views_map[stick2.id]
        else:
            self.stick_views_map[stick2.id] = stick2.stick_views

        self.sticks_unlinked.emit(stick1, stick2)

    def unlink_stick_(self, stick: Stick):
        stick_views = stick.stick_views
        for stick_view in stick_views:
            self.unlink_sticks(stick, stick_view)

    def get_cameras_stick_links(self, camera: Camera) -> List[Tuple[int, int, int]]:
        camera_links = filter(lambda k_v: k_v[1][0].camera_id == camera.id, self.stick_views_map.items())
        return list(map(lambda t: (t[0], t[1][1], t[1][2]), camera_links))
    
    #def remove_stick(self, stick: Stick):
    #    camera: Camera = next(filter(lambda cam: cam.id == stick.camera_id, self.cameras))
    #    if stick.id in self.stick_views_map:
    #        self.unlink_stick_(stick)
    #    self.stick_removed.emit(stick)
    #
    #    camera.sticks = list(filter(lambda s: s.id != stick.id, camera.sticks))
    #    self.unused_stick_ids.append(stick.id)
    
    #def get_stick_by_id(self, id: int) -> Optional[Stick]:
    #    for cam in self.cameras:
    #        for stick in cam.sticks:
    #            if stick.id == id:
    #                return stick
    #    return None

    def get_camera(self, id_or_path_str: Union[int, str]) -> Camera:
        if isinstance(id_or_path_str, str):
            return next(filter(lambda cam: str(cam.folder) == id_or_path_str, self.cameras))
        return next(filter(lambda cam: cam.id == id_or_path_str, self.cameras))

    # Reacts to a stick being deleted. It unlinks it from all other sticks and then emits signal that this stick
    # is no longer in use, so GUI should handle this by destroying any StickWidgets representing this stick.
    def handle_stick_removed(self, stick: Stick):
        self.unlink_stick_(stick)
        self.unused_stick_ids.append(stick.id)
        del self.stick_local_to_global_ids[stick.camera_id][stick.local_id]
        #self.stick_removed.emit(stick)

    def handle_sticks_removed(self, sticks: List[Stick]):
        for stick in sticks:
            self.handle_stick_removed(stick)

    def connect_camera_signals(self, camera: Camera):
        camera.stick_removed.connect(self.handle_stick_removed)
        camera.sticks_removed.connect(self.handle_sticks_removed)
        camera.sticks_added.connect(self.handle_camera_sticks_added)

    def disconnect_camera_signals(self, camera: Camera):
        camera.stick_removed.disconnect(self.handle_stick_removed)
        camera.sticks_removed.disconnect(self.handle_sticks_removed)

    def get_state(self) -> Dict[str, Any]:
        stick_views_list = []
        presence_set = set({})
        for stick_id, stick_views in self.stick_views_map.items():
            if stick_id in presence_set:
                continue
            stick_views = list(map(lambda stick: stick.id, stick_views))
            stick_views.append(stick_id)
            stick_views_list.append(stick_views)
            for stick_id_ in stick_views:
                presence_set.add(stick_id_)
        return {
            'next_camera_id': self.next_camera_id,
            'next_stick_id': self.next_stick_id,
            'path': str(self.path),
            #'camera_folders': list(map(lambda path: str(path), self.camera_folders)),
            #'cameras_ids': self.cameras_ids,
            'cameras_ids': self.make_camera_paths_relative(),
            'linked_cameras': list(self.linked_cameras),
            'stick_views_map': stick_views_list,
            'unused_stick_ids': self.unused_stick_ids,
            'stick_local_to_global_ids': self.stick_local_to_global_ids,
        }

    def make_camera_paths_relative(self):
        transformed: Dict[str, int] = {}
        for path_str, cam_id in self.cameras_ids.items():
            path = Path(path_str)
            if path.is_absolute():
                try:
                    path = path.relative_to(self.path.parent)
                except ValueError:
                    pass
            transformed[str(path)] = cam_id
        return self.cameras_ids

    def register_stick(self, stick: Stick, camera: Camera):
        stick.camera_id = camera.id
        if stick.camera_id not in self.stick_local_to_global_ids:
            self.stick_local_to_global_ids[stick.camera_id] = dict({})

        if stick.local_id not in self.stick_local_to_global_ids[stick.camera_id]:
            if len(self.unused_stick_ids) > 0:
                id_to_assign = self.unused_stick_ids.pop()
            else:
                id_to_assign = self.next_stick_id
                self.next_stick_id += 1
            self.stick_local_to_global_ids[stick.camera_id][stick.local_id] = id_to_assign
        else:
            id_to_assign = self.stick_local_to_global_ids[stick.camera_id][stick.local_id]
        stick.id = id_to_assign
        self.sticks[stick.id] = stick

    def get_stick_by_id(self, stick_id: int) -> Stick:
        return self.sticks.get(stick_id)

    def get_stick_views(self, stick: Stick) -> List[Stick]:
        return self.stick_views_map.get(stick.id, [])

    def get_stick_view_from_camera(self, stick: Stick, camera: Union[int, Camera]) -> Optional[Stick]:
        if isinstance(camera, Camera):
            camera_id = camera.id
        else:
            camera_id = camera

        stick_views = self.get_stick_views(stick)
        if len(stick_views) == 0:
            return None

        for stick in stick_views:
            if stick.camera_id == camera_id:
                return stick

        return None

    def handle_camera_sticks_added(self, sticks: List[Stick], camera: Camera):
        for stick in sticks:
            self.register_stick(stick, camera)

    def get_camera_groups(self) -> List[List[Camera]]:
        return list(self.camera_groups.values())

