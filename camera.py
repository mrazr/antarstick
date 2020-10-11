import os
from os import listdir, mkdir
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QObject, pyqtSignal

from stick import Stick

import json

PD_DAY = 1
PD_SNOW = 2

PD_STICK_TOP = 0
PD_STICK_BOTTOM = 1
PD_STICK_LENGTH_PX = 2
PD_STICK_SNOW_HEIGHT = 3
PD_STICK_COLS = 4

ENDPOINT_COLUMN_CONVERTER = lambda cell: np.array(list(map(int, cell.strip('[] ').split())))

MEASUREMENTS_FILE = 'results.csv'
IMAGE_STATS_FILE = 'img_stats.csv'

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
        self.measurements_path = Path('./_results')
        if not (self.folder / self.measurements_path).exists():
            mkdir(str(self.folder / self.measurements_path)) #TODO handle possible exceptions
        self.photo_daytime_snow = pd.DataFrame()
        #if measurements_path:
        #    self.measurements_path = measurements_path
        #    self.__load_measurements()
        #else:
        #    self.measurements = pd.DataFrame()
        #    self.measurements_path = None
        #self.rep_image_path = self.folder / Path(listdir(self.folder)[0]) #TODO listdir - filter out non image files
        #self.rep_image: np.ndarray = cv.resize(cv.imread(str(self.rep_image_path)), (0, 0), fx=0.25, fy=0.25,
        #                                      interpolation=cv.INTER_NEAREST)
        self.image_list: List[str] = list(sorted(filter(lambda f: f[-4:].lower() == 'jpeg' or f[-3:].lower() == 'jpg', listdir(self.folder))))
        self.rep_image_path: str = self.image_list[0]
        self.rep_image: Optional[np.ndarray] = None

        self.stick_labels_column_ids = dict({})

        if self.measurements_path is None:
            self.photo_daytime_snow = pd.DataFrame(data={
                'image': self.image_list,
                'day': ['-'] * len(self.image_list),
                'snow': ['-'] * len(self.image_list),
            })
            self.measurements = pd.DataFrame()
        else:
            self.__load_measurements()

        self.next_photo_daytime_snow = self.photo_daytime_snow.index[self.photo_daytime_snow['day'] == '-'].tolist()
        self.next_photo_daytime_snow = self.next_photo_daytime_snow[0] if len(self.next_photo_daytime_snow) > 0 else len(self.image_list)

        self.image_names_ids: Dict[str, int] = {image_name: image_id for image_id, image_name in enumerate(self.image_list)}
        self.next_photo_id: int = 0
        self.next_photo: str = self.image_list[self.next_photo_id]
        self.default_stick_length_cm: int = 60

    def __load_measurements(self) -> None:
        try:
            with open(str(self.folder / self.measurements_path / MEASUREMENTS_FILE)) as meas_file:
                measurements = pd.read_csv(meas_file)
                if measurements.shape[0] > 0:
                    last_photo = measurements.iloc[-1][0]
                    self.next_photo_id = self.image_list.index(last_photo) + 1
                    self.next_photo = self.image_list[self.next_photo_id] if self.next_photo_id < len(self.image_list) else None
                for i in range(2, len(measurements.columns.values), 4):
                    column: str = measurements.columns.values[i]
                    self.stick_labels_column_ids[column[:column.index('_')]] = i
                self.measurements = measurements.apply(lambda x: x.apply(ENDPOINT_COLUMN_CONVERTER) if x.name.endswith(('bottom', 'top')) else x)
                #for col in self.stick_labels_column_ids.values():
                #    self.measurements.iloc[:, col + PD_STICK_TOP] = self.measurements.iloc[:, col + PD_STICK_TOP].apply(ENDPOINT_COLUMN_CONVERTER)
                #    self.measurements.iloc[:, col + PD_STICK_BOTTOM] = self.measurements.iloc[:, col + PD_STICK_BOTTOM].apply(ENDPOINT_COLUMN_CONVERTER)
        except FileNotFoundError:
            self.measurements = pd.DataFrame()
            self._update_stick_labels_id()

        try:
            with open(str(self.measurements_path / IMAGE_STATS_FILE)) as meas_file:
                self.photo_daytime_snow = pd.read_csv(meas_file)
        except FileNotFoundError:
            self.photo_daytime_snow = pd.DataFrame(data={
                'image': self.image_list,
                'day': ['-'] * len(self.image_list),
                'snow': ['-'] * len(self.image_list),
            })

    def save_measurements(self, path: Optional[Path] = None, filename: str = 'results.csv'):
        if path is None:
            path = self.folder / self.measurements_path
        with open(str(path / filename), 'w') as output:
            self.measurements.to_csv(output, index=False)
        with open(str(path / IMAGE_STATS_FILE), 'w') as output:
            self.photo_daytime_snow.to_csv(output, index=False)

    def get_state(self):
        state = self.__dict__.copy()
        del state['measurements']
        del state['rep_image']
        return state

    def get_folder_name(self) -> str:
        return self.folder.name
    
    def stick_count(self) -> int:
        return len(self.sticks)

    #@staticmethod
    #def build_from_state(state: Dict) -> 'Camera':
    #    path = state['folder']
    #    sticks = state['sticks']
    #    _id = state['id']
    #    measurements_path = state['measurements_path']
    #    camera = Camera(path, _id, measurements_path)
    #    camera.sticks = sticks
    #    camera.rep_image_path = state['rep_image_path']
    #    return camera

    def add_stick(self, stick: Stick):
        stick.camera_id = self.id
        self.sticks.append(stick)
        self.stick_added.emit(stick)

    def remove_stick(self, stick: Stick):
        if stick.camera_id != self.id:
            return
        self.sticks = list(filter(lambda s: s.local_id != stick.local_id, self.sticks))
        self.unused_stick_ids.append(stick.local_id)
        self.unused_stick_ids.sort(reverse=True)
        self._update_stick_labels_id()
        self.stick_removed.emit(stick)

    #def add_sticks(self, sticks: List[Stick]):
    #    for stick in sticks:
    #        stick.camera_id = self.id
    #        self.sticks.append(stick)
    #    self.sticks_added.emit(sticks)

    def remove_sticks(self, sticks: Optional[List[Stick]] = None):
        if len(self.sticks) == 0:
            return
        if sticks is None:
            sticks = self.sticks.copy()
            self.sticks.clear()
            self._update_stick_labels_id()
        else:
            for stick in sticks:
                self.sticks.remove(stick)
        self.unused_stick_ids.extend(map(lambda stick: stick.local_id, sticks))
        self.unused_stick_ids.sort(reverse=True)
        self.sticks_removed.emit(sticks)

    def save(self):
        stick_states = list(map(lambda s: s.get_state(), self.sticks))
        state = {
            #'folder': str(self.folder),
            #'rep_image_path': str(self.rep_image_path),
            'rep_image': self.rep_image_path,
            'sticks': stick_states,
            'measurements_path': str(self.measurements_path),
            'next_stick_id': self.next_stick_id,
            'unused_stick_ids': self.unused_stick_ids,
            'default_stick_length_cm': self.default_stick_length_cm,
        }

        if len(self.measurements.columns) == 0: #or (len(self.measurements.columns) - 1) / 4 != len(self.sticks):
            data = {
                'image_name': [],
                'processed': [],
            }

            for stick in self.sticks:
                data[stick.label + '_top'] = []
                data[stick.label + '_bottom'] = []
                data[stick.label + '_height_px'] = []
                data[stick.label + '_snow_height'] = []

            self.measurements = pd.DataFrame(data=data)

        self.save_measurements()
        with open(self.folder / 'camera.json', 'w') as f:
            json.dump(state, f, indent=1)

    @staticmethod
    def load_from_path(folder: Path) -> Optional['Camera']:
        camera_json = folder / 'camera.json'
        if not camera_json.exists():
            return None
        camera = None
        with open(str(camera_json), 'r') as f:
            state: Dict[str, Any] = json.load(f)
            #camera = Camera(Path(state['folder']))
            camera = Camera(folder)
            #camera.rep_image_path = Path(state['rep_image_path'])
            camera.rep_image_path = state['rep_image']
            #camera.measurements_path = Path(state['measurements_path'])
            camera.measurements_path = Path(state['measurements_path'])
            camera.next_stick_id = state['next_stick_id']
            camera.unused_stick_ids = state['unused_stick_ids']
            camera.default_stick_length_cm = state.get('default_stick_length_cm', 60)
            sticks = list(map(lambda stick_state: Stick.build_from_state(stick_state), state['sticks']))
            camera.sticks = sticks
            camera.__load_measurements()

        return camera

    def get_stick_by_id(self, stick_id: int) -> Stick:
        return next(filter(lambda s: s.local_id == stick_id, self.sticks))

    def create_new_sticks(self, lines: List[np.ndarray], image: str = '') -> List[Stick]:
        """Creates Stick instance for given line coordinates and assigns them Camera-unique IDs."""
        count = len(lines)
        sticks = []
        for i in range(count):
            if len(self.unused_stick_ids) > 0:
                id_to_assign = self.unused_stick_ids.pop()
            else:
                id_to_assign = self.next_stick_id
                self.next_stick_id += 1
            line = lines[i]
            stick = Stick(local_id=id_to_assign, view=image)
            stick.set_endpoints(line[0][0], line[0][1], line[1][0], line[1][1])
            sticks.append(stick)
        self.sticks.extend(sticks)
        self.sticks_added.emit(sticks, self)
        self._update_stick_labels_id()
        return sticks

    def _update_stick_labels_id(self):
        self.stick_labels_column_ids.clear()
        self.sticks.sort(key=lambda stick: stick.local_id)
        for i, stick in enumerate(self.sticks):
            self.stick_labels_column_ids[stick.label] = PD_STICK_COLS * i + 2

    def get_batch(self, count: int = 50) -> List[str]:
        """Return count image names that are to be processed next."""
        real_count = min(count, len(self.image_list) - self.next_photo_id + 1)
        return self.image_list[self.next_photo_id:self.next_photo_id + real_count]

    def get_measurement_for(self, image_name: str) -> Dict[str, Dict[str, Any]]:
        """For a give image name of this Camera, returns info about all sticks in that image."""
        if self.measurements.shape[0] == 0:
            return None
        measurement_dict = dict({})
        image_id = self.image_names_ids.get(image_name, None)
        if image_id is None or image_id >= self.next_photo_id:
            return None
        measurement = self.measurements.iloc[image_id]
        for stick in self.sticks:
            stick_column_id = self.stick_labels_column_ids[stick.label]
            measurement_dict[stick.label] = {
                'top': measurement[stick_column_id + PD_STICK_TOP],
                'bottom': measurement[stick_column_id + PD_STICK_BOTTOM],
                'length_px': measurement[stick_column_id + PD_STICK_LENGTH_PX],
                'snow_height': measurement[stick_column_id + PD_STICK_SNOW_HEIGHT],
            }

        return measurement_dict

    def get_measurements_for_stick(self, stick: Stick) -> pd.DataFrame:
        """For a given `stick` returns a DataFrame containing image names, coordinates, length and snow height."""
        if self.measurements.shape[0] == 0:
            return None
        stick_column_id = self.stick_labels_column_ids[stick.label]
        return self.measurements.iloc[:, [0] + list(range(stick_column_id, stick_column_id + PD_STICK_SNOW_HEIGHT + 1))]

    def insert_measurements(self, df: pd.DataFrame):
        self.measurements = self.measurements.append(df)
        self.next_photo_id += df.shape[0]
        self.save_measurements(self.measurements_path)

    def get_processed_count(self) -> int:
        return self.next_photo_id

    def get_photo_count(self) -> int:
        return len(self.image_list)

    def photo_is_daytime(self, img_name: str) -> Optional[bool]:
        idx = self.image_names_ids[img_name]
        day = self.photo_daytime_snow.iloc[idx, PD_DAY]
        if day == '-':
            return None
        return day == 'y'

    def photo_is_snow(self, img_name: str) -> Optional[bool]:
        idx = self.image_names_ids[img_name]
        snow = self.photo_daytime_snow.iloc[idx, PD_SNOW]
        if snow == '-':
            return None
        return snow == 'y'

    def set_photo_daytime_snow(self, img_name: str, daytime: bool, snow: bool) -> int:
        idx = self.image_names_ids[img_name]
        self.photo_daytime_snow.iloc[idx, 1:] = ['y' if daytime else 'n', 'y' if snow else 'n']
        self.next_photo_daytime_snow += 1
        return idx

    def get_next_photo_daytime_snow(self, count: int = 2) -> List[Path]:
        if self.next_photo_daytime_snow >= len(self.image_list):
            return []
        until = min(self.next_photo_daytime_snow + count, len(self.image_list))
        return list(map(lambda img: self.folder / img, self.image_list[self.next_photo_daytime_snow:until]))

    def get_stick_in_image(self, stick: Stick, image: str) -> Optional[Stick]:
        image_id = self.image_names_ids.get(image, -1)
        if image_id < 0:
            return None
        stick_id = self.stick_labels_column_ids[stick.label]
        data = self.measurements.iloc[image_id, [stick_id + PD_STICK_TOP,
                                                 stick_id + PD_STICK_BOTTOM,
                                                 stick_id + PD_STICK_LENGTH_PX,
                                                 stick_id + PD_STICK_SNOW_HEIGHT,
                                                 ]]
        stick.top = data[0]
        stick.bottom = data[1]
        stick.length_px = data[2]
        stick.snow_height_cm = data[3]

    def get_sticks_in_image(self, image: str) -> List[Stick]:
        image_id = self.image_names_ids[image]

        if self.measurements.shape[0] == 0 or self.measurements.iat[image_id, 2][0] < 0:
            return self.sticks
        for stick in self.sticks:
            stick_id = self.stick_labels_column_ids[stick.label]
            stick.view = image
            stick.top = self.measurements.iat[image_id, stick_id + PD_STICK_TOP]
            stick.bottom = self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM]
            stick.length_px = float(self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX])
            stick.snow_height_cm = int(self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT])
            self.stick_changed.emit(stick)
        return self.sticks

    def initialize_measurements(self, save_immediately: bool = False):
        data = {
            'image_name': self.image_list,
            'processed': [False] * len(self.image_list)
        }

        for stick in self.sticks:
            data[stick.label + '_top'] = [np.array([-1, -1], np.int32)] * len(self.image_list)
            data[stick.label + '_bottom'] = [np.array([-1, -1], np.int32)] * len(self.image_list)
            data[stick.label + '_height_px'] = [-1] * len(self.image_list)
            data[stick.label + '_snow_height'] = [-1] * len(self.image_list)

        self.measurements = pd.DataFrame(data=data)
        self.sticks.sort(key=lambda stick: stick.local_id)

        image_id = self.image_names_ids[self.sticks[0].view]

        for stick in self.sticks:
            stick_id = self.stick_labels_column_ids[stick.label]
            self.measurements.iat[image_id, stick_id + PD_STICK_TOP] = stick.top
            self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM] = stick.bottom
            self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX] = stick.length_px
            self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT] = stick.snow_height_cm

        if save_immediately:
            self.save_measurements()

    def insert_measurements2(self, measurements: Dict[str, List[Stick]]):
        for img_name, sticks in measurements.items():
            image_id = self.image_names_ids[img_name]
            self.measurements.iat[image_id, 1] = True
            for stick in sticks:
                stick = stick.scale(1.0/stick.scale_)
                stick_id = self.stick_labels_column_ids[stick.label]
                self.measurements.iat[image_id, stick_id + PD_STICK_TOP] = stick.top
                self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM] = stick.bottom
                self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX] = stick.length_px
                self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT] = stick.snow_height_cm

