from dataclasses import dataclass
from pathlib import PurePath, PosixPath, Path
from typing import Dict, List, Optional, Set, Tuple, Union, Any
from datetime import datetime, timedelta

import jsonpickle
import json

from numpy import zeros
from PyQt5.QtCore import QObject, QTimer, Qt
from PyQt5.QtCore import pyqtSignal as Signal
from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QWidget
from pandas import Timestamp, Timedelta, Series
import pandas as pd
import numpy as np
import pprofile

from camera import Camera, PD_DATE
from stick import Stick


class CameraSynchronization(QObject):

    synchronization_finished = Signal('PyQt_PyObject')

    def __init__(self, left_cam: Camera, right_cam: Camera, left_ts: Timestamp, right_ts: Timestamp, sync: Dict[Camera, List[str]] = {}):
        QObject.__init__(self)
        self.left_camera = left_cam
        self.right_camera = right_cam
        self.left_timestamp = left_ts
        self.right_timestamp = right_ts
        self.synchronized_photos: Dict[Camera, List[str]] = {}

    @staticmethod
    def get_image_id_for_timestamp(camera: Camera, timestamp: datetime) -> int:
        return camera.measurements.index.get_loc(timestamp)

    def get_reciprocal_image_by_date(self, source: Camera, timestamp: datetime) -> Optional[Tuple[str, datetime]]:
        target = self.right_camera if source == self.left_camera else self.left_camera
        image_id = self.get_image_id_for_timestamp(source, timestamp)
        rec_image_name = self.synchronized_photos[source][image_id]
        if rec_image_name == "":
            return None
        rec_image_id = target.image_names_ids[rec_image_name]
        rec_timestamp = target.measurements.iloc[rec_image_id]['date_time']
        return rec_image_name, rec_timestamp

    def get_reciprocal_image_by_name(self, source: Camera, image_name: str) -> Optional[Tuple[str, datetime]]:
        target = self.right_camera if source == self.left_camera else self.left_camera
        source_image_id = source.image_names_ids[image_name]
        target_image_name = self.synchronized_photos[source][source_image_id]
        if target_image_name == "":
            return None
        target_image_id = target.image_names_ids[target_image_name]
        target_timestamp = target.measurements.iloc[target_image_id]['date_time']
        return target_image_name, target_timestamp

    def get_synchronized_images(self) -> Dict[Camera, List[str]]:
        return self.synchronized_photos

    def get_reciprocal_date(self, source: Camera, timestamp: datetime) -> Optional[datetime]:
        target = self.right_camera if source == self.left_camera else self.left_camera
        source_image_id = self.get_image_id_for_timestamp(source, timestamp)
        target_image_name = self.synchronized_photos[source][source_image_id]
        if target_image_name == "":
            return None
        target_image_id = target.image_names_ids[target_image_name]
        return target.measurements.iloc[target_image_id]['date_time']

    def synchronize(self, left_ts: Optional[Timestamp] = None, right_ts: Optional[Timestamp] = None) -> 'CameraSynchronization':
        if left_ts is None or right_ts is None:
            left_ts = self.left_timestamp
            right_ts = self.right_timestamp
        common_time = pd.Timestamp(datetime(year=2000, month=1, day=1))

        cam1_image_id = self.left_camera.measurements.index.get_loc(left_ts)
        cam1_image_name = self.left_camera.image_list[cam1_image_id]
        cam1_offsets: Series = self.left_camera.measurements.loc[:, 'date_time'].sub(left_ts)
        cam1_offsets.loc[left_ts] = pd.Timedelta(days=0)

        #self.left_camera.measurements.loc[:, 'date_time'].diff().to_csv(f'/home/radoslav/{self.left_camera.folder.name}_off.csv')

        df1 = pd.DataFrame(data={'date_time':  [common_time] * cam1_offsets.shape[0],
                                 'offset': cam1_offsets,
                                 'cam': [0] * cam1_offsets.shape[0],
                                 'orig_dt': self.left_camera.measurements.loc[:, 'date_time'],
                                 'image_name': self.left_camera.measurements.loc[:, 'image_name']})
        df1.loc[:, 'date_time'] = df1.loc[:, 'date_time'].add(cam1_offsets)
        df1.set_index('image_name', drop=False, inplace=True)

        df1.iat[cam1_image_id, PD_DATE] = df1.iat[cam1_image_id, PD_DATE] + Timedelta(seconds=1)
        df1.set_index('date_time', drop=False, inplace=True)


        cam2_image_id = self.right_camera.measurements.index.get_loc(right_ts)
        cam2_image_name = self.right_camera.image_list[cam2_image_id]
        #self.right_camera.measurements.loc[:, 'date_time'].diff().to_csv(f'/home/radoslav/{self.right_camera.folder.name}_off.csv')
        #cam2_offsets: Series = self.right_camera.measurements.loc[:, 'date_time'].diff()
        cam2_offsets: Series = self.right_camera.measurements.loc[:, 'date_time'].sub(right_ts)
        cam2_offsets.loc[right_ts] = pd.Timedelta(days=0)
        df2 = pd.DataFrame(data={'date_time': [common_time] * cam2_offsets.shape[0],
                                 'offset': cam2_offsets,
                                 'cam': [1] * cam2_offsets.shape[0],
                                 'orig_dt': self.right_camera.measurements.loc[:, 'date_time'],
                                 'image_name': self.right_camera.measurements.loc[:, 'image_name']})
        #df2.loc[:, 'date_time'] = self.right_camera.measurements.loc[:, 'date_time']
        df2.loc[:, 'date_time'] = df2.loc[:, 'date_time'].add(cam2_offsets)
        df2.iat[cam2_image_id, 0] += Timedelta(seconds=1)

        df2.set_index('image_name', drop=False, inplace=True)

        df2.set_index('date_time', drop=False, inplace=True)

        df = df1.append(df2)
        df.set_index('date_time', drop=False, inplace=True)
        df.sort_index(inplace=True)

        source = 0 if cam1_offsets.shape[0] < cam2_offsets.shape[0] else 1
        target = 0 if source == 1 else 1

        source_df = df1 if source == 0 else df2
        target_df = df2 if target == 1 else df1

        dupli_dt = df.index.duplicated(keep='first')

        for i, is_dup in enumerate(dupli_dt):
            if not is_dup:
                continue
            dt = df.iloc[i]['date_time']
            #df['date_time'].iloc[i] = dt + Timedelta(seconds=1)
            df.iloc[i, PD_DATE] = dt + Timedelta(seconds=1)
            if df.iloc[i]['cam'] == source:
                source_df.loc[dt, 'date_time'] = source_df.loc[dt, 'date_time'] + Timedelta(seconds=1)
            else:
                target_df.loc[dt, 'date_time'] = target_df.loc[dt, 'date_time'] + Timedelta(seconds=1)

        #df.loc[dupli_dt, 'date_time'] = df.loc[dupli_dt, 'date_time'] + Timedelta(seconds=1)
        df.set_index('date_time', drop=False, inplace=True)
        source_df.set_index('date_time', drop=False, inplace=True)
        target_df.set_index('date_time', drop=False, inplace=True)

        source_ids = df.loc[df['cam'] == source].index
        target_ids = df.loc[df['cam'] == target].index
        #source_ids = enumerate(map(lambda t: t[0], filter(lambda t: t[1], enumerate(df['cam'] == source))))
        #target_ids = enumerate(map(lambda t: t[0], filter(lambda t: t[1], enumerate(df['cam'] == target))))

        source_idxs: Dict[int, int] = {int(df.index.get_loc(dt)): int(source_df.index.get_loc(dt)) for dt in source_ids}
        target_idxs: Dict[int, int] = {int(df.index.get_loc(dt)): int(target_df.index.get_loc(dt)) for dt in target_ids}

        #source_idxs: Dict[int, int] = {dt[1]: dt[0] for dt in source_ids}
        #target_idxs: Dict[int, int] = {dt[1]: dt[0] for dt in target_ids}

        target_to_source: List[Tuple[int, Timedelta]] = [(-1, Timedelta(days=9999))] * target_df.shape[0]
        source_to_target: List[Tuple[int, Timedelta]] = [(-1, Timedelta(days=9999))] * source_df.shape[0]

        for common_idx, local_idx in source_idxs.items():
            source_dt = df.iat[common_idx, 0]

            target_delta_next = Timedelta(days=9999)
            next_target_assignment = (-1, Timedelta(days=9999))
            target_delta_prev = Timedelta(days=9999)
            prev_target_assignment = (-1, Timedelta(days=9999))

            if common_idx - 1 >= 0 and df.iat[common_idx - 1, 2] == target:
                target_dt_prev = df.iat[common_idx - 1, 0]
                target_delta_prev = source_dt - target_dt_prev
                prev_target_idx = target_idxs[common_idx - 1]
                prev_target_assignment = target_to_source[prev_target_idx]
            if common_idx + 1 < df.shape[0]:
                if df.iat[common_idx + 1, 2] == target:
                    target_dt_next = df.iat[common_idx + 1, 0]
                    target_delta_next = target_dt_next - source_dt
                    next_target_idx = target_idxs[common_idx + 1]
                    next_target_assignment = target_to_source[next_target_idx]

            if target_delta_prev < target_delta_next and target_delta_prev < prev_target_assignment[1]:
                source_to_target[local_idx] = (prev_target_idx, target_delta_prev)
                previous_source = target_to_source[prev_target_idx][0]
                if previous_source >= 0:
                    source_to_target[previous_source] = (-1, Timedelta(days=9999))
                target_to_source[prev_target_idx] = (local_idx, target_delta_prev)
            elif target_delta_next < target_delta_prev and target_delta_next < next_target_assignment[1]:
                source_to_target[local_idx] = (next_target_idx, target_delta_next)
                previous_source = target_to_source[next_target_idx][0]
                if previous_source >= 0:
                    source_to_target[previous_source] = (-1, Timedelta(days=9999))
                target_to_source[next_target_idx] = (local_idx, target_delta_next)

        source_cam = self.left_camera if len(self.left_camera.image_list) < len(self.right_camera.image_list) else self.right_camera
        target_cam = self.right_camera if source_cam == self.left_camera else self.left_camera
        source_to_target = list(map(lambda idx_td: '' if idx_td[0] < 0 else target_cam.image_list[idx_td[0]],
                                    source_to_target))
        target_to_source = list(map(lambda idx_td: '' if idx_td[0] < 0 else source_cam.image_list[idx_td[0]],
                                    target_to_source))

        self.synchronized_photos = {source_cam: source_to_target, target_cam: target_to_source}
        #source_ = []
        #target_ = []
        #for source_dt in source_cam.measurements.loc[:, 'date_time']:
        #    target_dt = self.get_reciprocal_date(source_cam, source_dt)
        #    if target_dt is None:
        #        continue
        #    source_.append(source_dt)
        #    target_.append(target_dt)
        #mapp = pd.DataFrame(data={'s': source_, 't': target_})
        #mapp.to_csv('/home/radoslav/mapp.csv')
        self.left_timestamp = left_ts
        self.right_timestamp = right_ts
        self.synchronization_finished.emit(self)
        return self

    def __eq__(self, other: Union[Tuple[Camera, Camera], 'CameraSynchronization']):
        if isinstance(other, tuple):
            return (other[0] == self.left_camera and other[1] == self.right_camera) or (other[0] == self.right_camera and other[1] == self.left_camera)
        elif isinstance(other, CameraSynchronization):
            return other.left_camera == self.left_camera and other.right_camera == self.right_camera
        return False


class CameraSynchronizationEncoder(json.JSONEncoder):

    def default(self, o: Any) -> Any:
        if isinstance(o, CameraSynchronization):
            return {
                'left': o.left_camera.id,
                'right': o.right_camera.id,
                'left_timestamp': str(o.left_timestamp),
                'right_timestamp': str(o.right_timestamp),
                'synchronized_photos': {cam.id: image_names for cam, image_names in o.synchronized_photos.items()}
            }
        return json.JSONEncoder.default(self, o)


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
    cameras_linked = Signal([Camera, Camera, CameraSynchronization])
    cameras_unlinked = Signal([Camera, Camera])
    stick_removed = Signal([Stick])
    stick_created = Signal([Stick])
    all_sticks_removed = Signal([Camera])
    new_sticks_created = Signal('PyQt_PyObject')
    sticks_linked = Signal([Stick, Stick])
    sticks_unlinked = Signal([Stick, Stick])
    loading_finished = Signal()
    synchronization_started = Signal()
    synchronization_finished = Signal('PyQt_PyObject')

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
        #self.linked_cameras: Set[Tuple[int, int]] = set() #, int]] = set()
        #self.linked_cameras: List[Dict[str, Camera]] = []
        self.linked_cameras: List[CameraSynchronization] = []
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

        camera_links = list(filter(lambda link: link.left_camera == camera or link.right_camera == camera, self.linked_cameras))

        for link in camera_links:
            other_cam_id = link.left_camera if link.right_camera == camera.id else link.left_camera
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
                json.dump(state, output_file, indent=1, cls=CameraSynchronizationEncoder)
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
            #for link in linked_cameras:
            #    left_cam_id = link['left']
            #    right_cam_id = link['right']
            #    left_cam = self.get_camera(left_cam_id)
            #    right_cam = self.get_camera(right_cam_id)
            #    self.link_cameras(left_cam, right_cam)

            for sync in linked_cameras:
                sync_photos = sync['synchronized_photos']

                left_cam_id = sync['left']
                left_cam = self.get_camera(left_cam_id)
                left_timestamp = pd.Timestamp(sync['left_timestamp'])
                left_to_right_photos = sync_photos[str(left_cam_id)]

                right_cam_id = sync['right']
                right_cam = self.get_camera(right_cam_id)
                right_timestamp = pd.Timestamp(sync['right_timestamp'])
                right_to_left_photos = sync_photos[str(right_cam_id)]

                camera_sync = CameraSynchronization(left_cam, right_cam, left_ts=left_timestamp,
                                                    right_ts=right_timestamp)
                camera_sync.left_camera = left_cam
                camera_sync.left_timestamp = left_timestamp
                camera_sync.right_camera = right_cam
                camera_sync.right_timestamp = right_timestamp
                camera_sync.synchronized_photos = {left_cam: left_to_right_photos,
                                                   right_cam: right_to_left_photos}
                self.linked_cameras.append(camera_sync)
                self.cameras_linked.emit(left_cam, right_cam, camera_sync)

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
        #self.linked_cameras.add((cam1.id, cam2.id)) #, camera_group))
        #self.linked_cameras.append({'left': cam1, 'right': cam2})
        # synchronize cameras' timestamps
        synchronized_timestamps = self.find_closest_datetimes(cam1, cam2)
        synchronization = CameraSynchronization(left_cam=cam1, right_cam=cam2,
                                                left_ts=synchronized_timestamps[cam1],
                                                right_ts=synchronized_timestamps[cam2])
        #synchronization.left_camera = cam1
        #synchronization.right_camera = cam2
        #synchronization.left_timestamp = synchronized_timestamps[synchronization.left_camera]
        #synchronization.right_timestamp = synchronized_timestamps[synchronization.right_camera]
        synchronization.synchronization_finished.connect(self.synchronization_finished.emit)
        self.linked_cameras.append(synchronization)
        synchronization.synchronize()
        #self.synchronize_cameras(synchronization)
        #print(f'synchronized cameras: {cam1.folder.name} - {synchronized_timestamps[cam1]} <-> {cam2.folder.name} - {synchronized_timestamps[cam2]}')
        self.cameras_linked.emit(cam1, cam2, synchronization) #, camera_group)
    
    def unlink_cameras(self, cam1: Camera, cam2: Camera):
        link1 = (cam1.id, cam2.id)
        link2 = (cam2.id, cam1.id)
        link1 = (cam1, cam2)
        link2 = (cam2, cam1)
        #link = list(filter(lambda _link: _link == link1 or _link == link2, self.linked_cameras))[0]
        #self.available_link_groups.append(link[2])
        #self.available_link_groups.sort()
        self.linked_cameras = list(filter(lambda _link: _link != link1 and _link != link2, self.linked_cameras))
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
        stick1.primary = True
        stick2.primary = True
        stick1.alternative_view = stick2
        stick2.alternative_view = stick1
        if stick1.length_px > stick2.length_px:
            stick2.primary = False
            stick2.length_cm = stick1.length_cm
        else:
            stick1.primary = False
            stick1.length_cm = stick2.length_cm
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
        stick1.primary = True
        stick2.primary = True
        stick1.alternative_view = None
        stick2.alternative_view = None
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
            'linked_cameras': self.linked_cameras,
            #'linked_cameras': list(map(lambda link: {k: v.id for k, v in link.items()}, self.linked_cameras)),
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

    @staticmethod
    def find_closest_datetimes(cam1: Camera, cam2: Camera) -> Dict[Camera, Timestamp]:
        cam1_ts: Timestamp = cam1.measurements.iat[0, PD_DATE]
        offsets1 = cam2.measurements.loc[:, 'date_time'].sub(cam1_ts).abs()
        smallest_delta1 = offsets1.min()
        closest_timestamp_cam2 = cam2.measurements.iloc[offsets1.argmin()]['date_time']

        cam2_ts: Timestamp = cam2.measurements.iat[0, PD_DATE]
        offsets1 = cam1.measurements.loc[:, 'date_time'].sub(cam2_ts).abs()
        smallest_delta2 = offsets1.min()
        closest_timestamp_cam1 = cam1.measurements.iloc[offsets1.argmin()]['date_time']

        if smallest_delta2 < smallest_delta1:
            return {cam1: closest_timestamp_cam1, cam2: cam2_ts}
        return {cam1: cam1_ts, cam2: closest_timestamp_cam2}

    def synchronize_cameras(self, sync: CameraSynchronization):
        pass

