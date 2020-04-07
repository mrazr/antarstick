from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jsonpickle
from PySide2.QtCore import QObject, Signal

from camera import Camera


class Dataset(QObject):
    """Class for representing dataset, where a dataset is considered a set of cameras.
    ...

    Attributes
    ----------
    path : Path
        absolute path to the JSON file representing this Dataset
    stick_translation_table : Dict[int, Tuple[int, int]]
        a translation table to know how a particular stick
        on a camera relates to the same stick but seen on a different camera.
        Key is a global_stick_ID_1 and value is (global_stick_ID_2, camera_ID)
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

    Signals
    -------
    camera_added : Signal(Camera)
        Emitted when a Camera is added to Dataset. Argument is the Camera
        that was added.
    camera_removed : Signal(int)
        Emitted when a Camera is remove from Dataset.
        Argument is the id of the removed camera.
    """

    camera_added = Signal(Camera)
    camera_removed = Signal(int)

    def __init__(self, path: Optional[Path] = None):
        super(Dataset, self).__init__()
        if path is not None:
            if path.exists():
                self.load_from(path)
        else:
            self.path = Path(".")
        self.stick_translation_table: Dict[int, Tuple[int, int]] = dict({})
        self.cameras: List[Camera] = []
        self.next_camera_id = 0

    def add_camera(self, folder: Path):
        camera = Camera(folder, self.next_camera_id)
        if self.path:
            camera.measurements_path = self.path.parent / f"camera{camera.id}.csv"
        self.cameras.append(camera)
        self.next_camera_id += 1
        self.camera_added.emit(camera)

    def remove_camera(self, camera_id: int):
        old_camera_count = len(self.cameras)
        self.cameras = list(filter(lambda camera: camera.id != camera_id, self.cameras))
        if old_camera_count > len(self.cameras):
            self.camera_removed.emit(camera_id)

    def save(self) -> bool:
        if self.path == Path("."):
            return False
        try:
            with open(self.path, "w") as output_file:
                for camera in self.cameras:
                    path = self.path.parent / f"camera{camera.id}.csv"
                    camera.save_measurements(path)
                state = self.__dict__.copy()
                del state['camera_added']
                del state['camera_removed']
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
                self.stick_translation_table = decoded['stick_translation_table']
                self.next_camera_id = decoded['next_camera_id']
                self.cameras = [Camera.build_from_state(camera_state) for camera_state in decoded['cameras']]
                for camera in self.cameras:
                    self.camera_added.emit(camera)
            return True
        except OSError:  # TODO do actual handling
            return False
