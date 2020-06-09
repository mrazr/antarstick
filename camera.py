from os import listdir
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QObject, pyqtSignal

from stick import Stick

import json


class Camera(QObject):
    """Class for representing a particular photo folder
    comprised of photos from one camera.

    ...

    Attributes
    ----------
    folder : Path
        Path to the folder containing the photos.
    sticks : List[Stick]
        List of Stick-s that are seen in the photos of this Camera.
    id : int
        Integer identifier within the Dataset this Camera belongs to.
    measurements_path : Path
        Path to the CSV file where the measurements are stored.
    measurements : pandas.DataFrame
        DataFrame of measurements. Rows correspond to individual photos,
        columns correspond to individual sticks + maybe some other
        measurements.

    Methods
    -------
    save_measurements(path: Path)
        Saves this measurements to the file specified by `path` in CSV format.
    get_folder_name() -> str
        Returns the name of this Camera's photos folder.
    """

    stick_added = pyqtSignal([Stick])
    stick_removed = pyqtSignal(Stick)
    sticks_added = pyqtSignal(['PyQt_PyObject', 'PyQt_PyObject'])
    sticks_removed = pyqtSignal('PyQt_PyObject')
    stick_changed = pyqtSignal(Stick)

    def __init__(self, folder: Path, _id: int = -1, measurements_path: Optional[Path] = None):
        super(Camera, self).__init__()
        self.folder = Path(folder)
        self.sticks: List[Stick] = []
        self.id = _id
        self.next_stick_id = 0
        self.unused_stick_ids = []
        self.measurements_path = folder
        if measurements_path:
            self.measurements_path = measurements_path
            self.measurements = self.__load_measuremets()
        else:
            self.measurements = pd.DataFrame()
            self.measurements_path = None
        #self.rep_image_path = self.folder / Path(listdir(self.folder)[0]) #TODO listdir - filter out non image files
        #self.rep_image: np.ndarray = cv.resize(cv.imread(str(self.rep_image_path)), (0, 0), fx=0.25, fy=0.25,
        #                                      interpolation=cv.INTER_NEAREST)
        self.rep_image_path: Path = None
        self.rep_image = None
        self.image_list: List[str] = list(filter(lambda f: f[-4:].lower() == 'jpeg' or f[-3:].lower() == 'jpg',listdir(self.folder)))
        self.next_photo_id: int = 0
        self.next_photo: str = self.image_list[self.next_photo_id]

    def __load_measuremets(self) -> pd.DataFrame:
        try:
            with open(self.folder / '_results/results.csv') as meas_file:
                measurements = pd.read_csv(meas_file)
                if measurements.shape[0] > 0:
                    last_photo = measurements.iloc[-1][0]
                    self.next_photo_id = self.image_list.index(last_photo) + 1
                    self.next_photo = self.image_list[self.next_photo_id]
        except FileNotFoundError:
            pass
        return pd.DataFrame()

    def save_measurements(self, path: Path):
        self.measurements_path = path
        with open(path, "w") as output:
            self.measurements.to_csv(output, index=False)

    def get_state(self):
        state = self.__dict__.copy()
        del state['measurements']
        del state['rep_image']
        return state

    def get_folder_name(self) -> str:
        return self.folder.name
    
    def stick_count(self) -> int:
        return len(self.sticks)

    @staticmethod
    def build_from_state(state: Dict) -> 'Camera':
        path = state['folder']
        sticks = state['sticks']
        _id = state['id']
        measurements_path = state['measurements_path']
        camera = Camera(path, _id, measurements_path)
        camera.sticks = sticks
        camera.rep_image_path = state['rep_image_path']
        return camera

    def add_stick(self, stick: Stick):
        stick.camera_id = self.id
        self.sticks.append(stick)
        self.stick_added.emit(stick)

    def remove_stick(self, stick: Stick):
        if stick.camera_id != self.id:
            return
        self.sticks = list(filter(lambda s: s.local_id != stick.local_id, self.sticks))
        self.unused_stick_ids.append(stick.local_id)
        self.stick_removed.emit(stick)

    def add_sticks(self, sticks: List[Stick]):
        for stick in sticks:
            stick.camera_id = self.id
            self.sticks.append(stick)
        self.sticks_added.emit(sticks)

    def remove_sticks(self, sticks: Optional[List[Stick]] = None):
        if len(self.sticks) == 0:
            return
        if sticks is None:
            sticks = self.sticks.copy()
            self.sticks.clear()
        else:
            for stick in sticks:
                self.sticks.remove(stick)
        self.unused_stick_ids.extend(map(lambda stick: stick.local_id, sticks))
        self.sticks_removed.emit(sticks)

    def save(self):
        stick_states = list(map(lambda s: s.get_state(), self.sticks))
        state = {
            'folder': str(self.folder),
            'rep_image_path': str(self.rep_image_path),
            'sticks': stick_states,
            'measurements_path': str(self.measurements_path),
            'next_stick_id': self.next_stick_id,
            'unused_stick_ids': self.unused_stick_ids,
        }

        if len(self.measurements.columns) == 0 or (len(self.measurements.columns) - 1) / 4 != len(self.sticks):
            data = {
                'image_name': [],
            }

            for stick in self.sticks:
                data[stick.label + '_top'] = []
                data[stick.label + '_bottom'] = []
                data[stick.label + '_height_px'] = []
                data[stick.label + '_snow_height'] = []

            self.measurements = pd.DataFrame(data=data)
            self.save_measurements(self.folder / '_results/results.csv')

        with open(self.folder / 'camera.json', 'w') as f:
            json.dump(state, f, indent=1)

    @staticmethod
    def load_from_path(folder: Path) -> Optional['Camera']:
        camera_json = folder / 'camera.json'
        if not camera_json.exists():
            return None
        camera = None
        with open(str(camera_json), 'r') as f:
            state = json.load(f)
            camera = Camera(Path(state['folder']))
            camera.rep_image_path = Path(state['rep_image_path'])
            camera.measurements_path = Path(state['measurements_path'])
            camera.next_stick_id = state['next_stick_id']
            camera.unused_stick_ids = state['unused_stick_ids']
            sticks = list(map(lambda stick_state: Stick.build_from_state(stick_state), state['sticks']))
            camera.__load_measuremets()

            camera.sticks = sticks
        return camera

    def get_stick_by_id(self, stick_id: int) -> Stick:
        return next(filter(lambda s: s.local_id == stick_id, self.sticks))

    def create_new_sticks(self, lines: List[np.ndarray]) -> List[Stick]:
        count = len(lines)
        sticks = []
        for i in range(count):
            if len(self.unused_stick_ids) > 0:
                id_to_assign = self.unused_stick_ids.pop()
            else:
                id_to_assign = self.next_stick_id
                self.next_stick_id += 1
            line = lines[i]
            stick = Stick(local_id=id_to_assign, stick_views=[])
            stick.set_endpoints(line[0][0], line[0][1], line[1][0], line[1][1])
            sticks.append(stick)
        self.sticks.extend(sticks)
        self.sticks_added.emit(sticks, self)
        return sticks

    def get_batch(self, count: int = 50) -> List[str]:
        real_count = min(count, len(self.image_list) - self.next_photo_id + 1)
        return self.image_list[self.next_photo_id:self.next_photo_id + real_count]