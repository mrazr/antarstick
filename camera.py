import os
from enum import IntEnum
from os import listdir, mkdir
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

import pandas as pd
import numpy as np
import cv2 as cv
from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5.QtWidgets import QMessageBox
import exifread

from stick import Stick

import json

PD_DATE = 0
PD_ORIG_DATE = 1
PD_IMAGE_NAME = 2
PD_IMAGE_STATE = 3
PD_IMAGE_QUALITY = 4
PD_IS_SNOWY = 5
PD_IS_DAY = 6

PD_DAY = 1
PD_SNOW = 2

PD_FIRST_STICK_COLUMN = PD_IS_DAY + 1

PD_STICK_TOP = 0
PD_STICK_BOTTOM = 1
PD_STICK_LENGTH_PX = 2
PD_STICK_SNOW_HEIGHT = 3
PD_STICK_VISIBLE = 4
PD_STICK_COLUMNS_COUNT = 5

ENDPOINT_COLUMN_CONVERTER = lambda cell: np.array(list(map(int, cell.strip('[] ').split())))

MEASUREMENTS_FILE = 'results.csv'
IMAGE_STATS_FILE = 'img_stats.csv'


class PhotoState(IntEnum):
    Skipped = -1,
    Unprocessed = 0,
    Processed = 1,


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
    non_increasingness = pyqtSignal('PyQt_PyObject')

    def __init__(self, folder: Path, _id: int = -1, measurements_path: Optional[Path] = None):
        super(Camera, self).__init__()
        self.folder = Path(folder)
        self.sticks: List[Stick] = []
        self.id = _id
        self.next_stick_id = 0
        self.unused_stick_ids = []
        self.measurements_path = Path('./_results')
        if not (self.folder / self.measurements_path).exists():
            mkdir(str(self.folder / self.measurements_path))  # TODO handle possible exceptions
        self.photo_daytime_snow = pd.DataFrame()
        # if measurements_path:
        #    self.measurements_path = measurements_path
        #    self.__load_measurements()
        # else:
        #    self.measurements = pd.DataFrame()
        #    self.measurements_path = None
        # self.rep_image_path = self.folder / Path(listdir(self.folder)[0]) #TODO listdir - filter out non image files
        # self.rep_image: np.ndarray = cv.resize(cv.imread(str(self.rep_image_path)), (0, 0), fx=0.25, fy=0.25,
        #                                      interpolation=cv.INTER_NEAREST)
        self.image_list: List[str] = list(
            sorted(filter(lambda f: f[-4:].lower() == 'jpeg' or f[-3:].lower() == 'jpg', listdir(self.folder))))
        self.rep_image_path: str = self.image_list[0]
        self.rep_image: Optional[np.ndarray] = None

        self.stick_labels_column_ids = dict({})
        self.measurements = pd.DataFrame()
        if self.measurements_path is None:
            self.photo_daytime_snow = pd.DataFrame(data={
                'image': self.image_list,
                'day': ['-'] * len(self.image_list),
                'snow': ['-'] * len(self.image_list),
            })
            self.initialize_results()
        else:
            self.__load_measurements()

        self.next_photo_daytime_snow = self.photo_daytime_snow.index[self.photo_daytime_snow['day'] == '-'].tolist()
        self.next_photo_daytime_snow = self.next_photo_daytime_snow[0] if len(
            self.next_photo_daytime_snow) > 0 else len(self.image_list)

        self.image_names_ids: Dict[str, int] = {image_name: image_id for image_id, image_name in
                                                enumerate(self.image_list)}
        self.photos_state: Dict[str, PhotoState] = {image_name: PhotoState.Unprocessed for image_name in self.image_list}
        self.next_photo_id: int = 0
        self.next_photo: str = self.image_list[self.next_photo_id]
        self.default_stick_length_cm: int = 60
        self.needs_to_save: bool = False
        self.stick_to_stick_vectors: Dict[Stick, Dict[Stick, np.ndarray]] = {}
        self.processed_photos_count = 0
        self.to_process: List[str] = []
        self.batches: List[List[str]] = []
        self.batches_prepared = False
        self.average_stick_length = 0.0
        self.quality_sticks: int = 0
        self.timestamps_available = False
        try:
            with open(self.folder / self.image_list[-1], 'rb') as f:
                tags = exifread.process_file(f, details=False)
            self.standard_image_size = (tags['EXIF ExifImageWidth'].values[0], tags['EXIF ExifImageLength'].values[0])
        except KeyError:
            img = cv.imread(str(self.folder / self.image_list[-1]), cv.IMREAD_GRAYSCALE)
            self.standard_image_size: Tuple[int, int] = (img.shape[1], img.shape[0])

    def __load_measurements(self) -> None:
        try:
            with open(str(self.folder / self.measurements_path / MEASUREMENTS_FILE)) as meas_file:
                measurements = pd.read_csv(meas_file, converters={'date_time': lambda d: pd.Timestamp(d)})
                if set(measurements.columns).intersection({'date_time', 'image_name', 'state'}) != {'date_time', 'image_name', 'state'}:
                    self.initialize_results()
                else:
                    measurements = measurements.set_index(keys=['date_time'], drop=False)
                    if measurements.shape[0] > 0:
                        last_photo = measurements.iloc[-1][PD_IMAGE_NAME]
                        self.next_photo_id = self.image_list.index(last_photo) + 1
                        self.next_photo = self.image_list[self.next_photo_id] if self.next_photo_id < len(
                            self.image_list) else None
                    for i in range(PD_FIRST_STICK_COLUMN, len(measurements.columns.values), PD_STICK_COLUMNS_COUNT):
                        column: str = measurements.columns.values[i]
                        self.stick_labels_column_ids[column[:column.index('_')]] = i
                    self.measurements = measurements.apply(
                        lambda x: x.apply(ENDPOINT_COLUMN_CONVERTER) if x.name.endswith(('bottom', 'top')) else x)
                    self.photos_state = {img_name: processed for img_name, processed in
                                         zip(self.measurements['image_name'], self.measurements['state'])}
        except FileNotFoundError:
            self.measurements = pd.DataFrame()
            self.initialize_results()
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
        self.needs_to_save = False

    def get_state(self):
        state = self.__dict__.copy()
        del state['measurements']
        del state['rep_image']
        return state

    def get_folder_name(self) -> str:
        return self.folder.name

    def stick_count(self) -> int:
        return len(self.sticks)

    # @staticmethod
    # def build_from_state(state: Dict) -> 'Camera':
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

    # def add_sticks(self, sticks: List[Stick]):
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
            # 'folder': str(self.folder),
            # 'rep_image_path': str(self.rep_image_path),
            'rep_image': self.rep_image_path,
            'sticks': stick_states,
            'measurements_path': str(self.measurements_path),
            'next_stick_id': self.next_stick_id,
            'unused_stick_ids': self.unused_stick_ids,
            'default_stick_length_cm': self.default_stick_length_cm,
            'timestamps_available': self.timestamps_available,
            'standard_image_size': self.standard_image_size,
        }

        # if len(self.measurements.columns) == 0: #or (len(self.measurements.columns) - 1) / 4 != len(self.sticks):
        #    data = {
        #        'image_name': [],
        #        'processed': [],
        #    }

        #    for stick in self.sticks:
        #        data[stick.label + '_top'] = []
        #        data[stick.label + '_bottom'] = []
        #        data[stick.label + '_height_px'] = []
        #        data[stick.label + '_snow_height'] = []

        #    self.measurements = pd.DataFrame(data=data)

        if self.needs_to_save:
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
            # camera = Camera(Path(state['folder']))
            camera = Camera(folder)
            # camera.rep_image_path = Path(state['rep_image_path'])
            camera.rep_image_path = state['rep_image']
            # camera.measurements_path = Path(state['measurements_path'])
            camera.measurements_path = Path(state['measurements_path'])
            camera.next_stick_id = state['next_stick_id']
            camera.unused_stick_ids = state['unused_stick_ids']
            camera.default_stick_length_cm = state.get('default_stick_length_cm', 60)
            camera.timestamps_available = state.get('timestamps_available', False)
            camera.standard_image_size = state.get('standard_image_size', None)
            if camera.standard_image_size is None:
                try:
                    with open(camera.folder / camera.image_list[-1], 'rb') as f:
                        tags = exifread.process_file(f, details=False)
                    camera.standard_image_size = (
                        tags['EXIF ExifImageWidth'].values[0], tags['EXIF ExifImageLength'].values[0])
                except KeyError:
                    img = cv.imread(str(camera.folder / camera.image_list[-1]), cv.IMREAD_GRAYSCALE)
                    camera.standard_image_size = (img.shape[1], img.shape[0])
            sticks = list(map(lambda stick_state: Stick.build_from_state(stick_state), state['sticks']))
            camera.sticks = sticks
            for s in camera.sticks:
                s.camera_folder = camera.folder
                camera.average_stick_length += s.length_cm
            camera.average_stick_length /= (len(camera.sticks) + 0.000001)
            camera.__load_measurements()

        return camera

    def get_stick_by_id(self, stick_id: int) -> Stick:
        return next(filter(lambda s: s.local_id == stick_id, self.sticks))

    def create_new_sticks(self, lines: List[Tuple[np.ndarray, int]], image: str = '') -> List[Stick]:
        """Creates Stick instance for given line coordinates and assigns them Camera-unique IDs."""
        count = len(lines)
        sticks = []
        for i in range(count):
            if len(self.unused_stick_ids) > 0:
                id_to_assign = self.unused_stick_ids.pop()
            else:
                id_to_assign = self.next_stick_id
                self.next_stick_id += 1
            line = lines[i][0]
            stick = Stick(local_id=id_to_assign, view=image, width=lines[i][1])
            stick.camera_folder = self.folder
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
            self.stick_labels_column_ids[stick.label] = PD_STICK_COLUMNS_COUNT * i + PD_FIRST_STICK_COLUMN

    def get_batch(self, batch_count: int = 2, batch_size: int = 50) -> Tuple[List[List[str]], int]:
        non_processed = self.measurements[self.measurements.iloc[:, PD_IMAGE_STATE] == PhotoState.Unprocessed]['image_name']
        if batch_size == 0:
            batch_size = min(int(np.ceil(non_processed.shape[0] / batch_count)), non_processed.shape[0])
        batches = []
        for i in range(batch_count):
            batch = non_processed[i * batch_size: i * batch_size + batch_size].tolist()
            batches.append(batch)
        return batches, len(non_processed)

    def get_processed_count(self) -> int:
        return self.processed_photos_count

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

    def get_sticks_in_image(self, image: str, output_sticks: Optional[List[Stick]] = None) -> List[Stick]:
        image_id = self.image_names_ids[image]

        if self.measurements.shape[0] == 0 or self.measurements.iloc[image_id]['state'] <= 0: #not self.measurements.iat[image_id, PD_IMAGE_PROCESSED]:
            return self.sticks
        sticks = output_sticks if output_sticks is not None else list(map(lambda s: s.copy(), self.sticks))
        for stick in sticks:
            stick_id = self.stick_labels_column_ids[stick.label]
            stick.view = image
            stick.top = self.measurements.iat[image_id, stick_id + PD_STICK_TOP]
            stick.bottom = self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM]
            stick.length_px = float(self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX])
            stick.set_snow_height_cm(int(self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT]))
            stick.is_visible = self.measurements.iat[image_id, stick_id + PD_STICK_VISIBLE]
            self.stick_changed.emit(stick)
        return sticks

    def initialize_measurements(self, save_immediately: bool = False):
        self._update_stick_labels_id()
        self.stick_to_stick_vectors.clear()
        average_length = np.mean(list(map(lambda stick: stick.length_px, self.sticks)))
        for i, stick_i in enumerate(self.sticks):
            stick_i.determines_quality = stick_i.length_px > 0.8 * average_length
            self.quality_sticks += 1
            stick_i_vecs = self.stick_to_stick_vectors.setdefault(stick_i, {})
            self.average_stick_length += stick_i.length_cm
            for stick_j in self.sticks[i + 1:]:
                vec = stick_j.bottom - stick_i.bottom
                stick_i_vecs[stick_j] = vec
                stick_j_vecs = self.stick_to_stick_vectors.setdefault(stick_j, {})
                stick_j_vecs[stick_i] = -1.0 * vec
            self.stick_changed.emit(stick_i)
        self.generate_bounding_boxes()

        stick_data = {}
        for stick in self.sticks:
            stick_data[stick.label + '_top'] = [np.array([-1, -1], np.int32)] * len(self.image_list) #pd.Series(data=[np.array([-1, -1], np.int32)] * len(self.image_list))
            stick_data[stick.label + '_bottom'] = [np.array([-1, -1], np.int32)] * len(self.image_list) #pd.Series(data=[np.array([-1, -1], np.int32)] * len(self.image_list))
            stick_data[stick.label + '_height_px'] = [-1] * len(self.image_list) #pd.Series(data=[-1] * len(self.image_list))
            stick_data[stick.label + '_snow_height'] = [-1] * len(self.image_list) #pd.Series(data=[-1] * len(self.image_list))
            stick_data[stick.label + '_visible'] = [False] * len(self.image_list) #pd.Series(data=[-1] * len(self.image_list))

        for col, val in stick_data.items():
            self.measurements.insert(self.measurements.shape[1], col, val)

        #self.measurements = pd.concat((self.measurements, pd.DataFrame(data=stick_data)), axis=1)
        self.sticks.sort(key=lambda stick: stick.local_id)

        image_id = self.image_names_ids[self.sticks[0].view]

        for stick in self.sticks:
            stick_id = self.stick_labels_column_ids[stick.label]
            self.measurements.iat[image_id, stick_id + PD_STICK_TOP] = stick.top
            self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM] = stick.bottom
            self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX] = stick.length_px
            self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT] = stick.snow_height_cm
            self.measurements.iat[image_id, stick_id + PD_STICK_VISIBLE] = stick.is_visible

        if save_immediately:
            self.save_measurements()
        self.needs_to_save = not save_immediately

    def initialize_results(self):
        self.measurements = pd.DataFrame(data=
        {
            'image_name': self.image_list,
            'state': [PhotoState.Unprocessed] * len(self.image_list),
            'image_quality': [-1.0] * len(self.image_list),
            'snowy': [False] * len(self.image_list),
            'is_day': [True] * len(self.image_list),
        })
        self.timestamps_available = False

    def insert_measurements2(self, measurements: Dict[str, Dict[str, Union[List[Stick], float, PhotoState]]]):
        for img_name, sticks_photoinfo in measurements.items():
            self.processed_photos_count += 1
            sticks = sticks_photoinfo['sticks']
            image_id = self.image_names_ids[img_name]
            state = sticks_photoinfo['state']
            self.measurements.iat[image_id, PD_IMAGE_STATE] = state
            self.measurements.iat[image_id, PD_IMAGE_QUALITY] = sticks_photoinfo['image_quality']
            self.measurements.iat[image_id, PD_IS_SNOWY] = sticks_photoinfo.get('is_snowy', False)
            self.measurements.iat[image_id, PD_IS_DAY] = sticks_photoinfo['is_day']
            self.photos_state[img_name] = state
            for stick in sticks:
                stick = stick.scale(1.0 / stick.scale_)
                stick_id = self.stick_labels_column_ids[stick.label]
                self.measurements.iat[image_id, stick_id + PD_STICK_TOP] = stick.top
                self.measurements.iat[image_id, stick_id + PD_STICK_BOTTOM] = stick.bottom
                self.measurements.iat[image_id, stick_id + PD_STICK_LENGTH_PX] = stick.length_px
                self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT] = stick.snow_height_cm
                self.measurements.iat[image_id, stick_id + PD_STICK_VISIBLE] = stick.is_visible

    def get_measurements(self, img: str, output_sticks: Optional[List[Stick]] = None) -> Dict[
        str, Union[List[Stick], float]]:
        image_id = self.image_names_ids[img]
        sticks = self.get_sticks_in_image(img, output_sticks)
        row = self.measurements.iloc[image_id]
        quality = row.iloc[PD_IMAGE_QUALITY]
        return {'sticks': sticks, 'image_quality': quality, 'state': row['state'], 'is_day': row['is_day']}

    def get_default_stick_length(self) -> int:
        return self.default_stick_length_cm

    def set_default_stick_length(self, length: int):
        self.default_stick_length_cm = length

    def is_label_available(self, stick: Stick, label: str) -> bool:
        if label == '':
            return False
        if stick.label == label:
            return True
        if len(list(filter(lambda s: s != stick and s.label == label, self.sticks))) > 0:
            return False
        stick.label = label
        return True

    def image_quality(self, image: str) -> float:
        return self.measurements.iat[self.image_names_ids[image], PD_IMAGE_QUALITY]

    def is_snowy(self, image: str) -> bool:
        if self.measurements.shape[1] > 5:
            return self.measurements.iat[self.image_names_ids[image], PD_IS_SNOWY]
        return False

    def average_snow_height(self, image: str) -> int:
        image_id = self.image_names_ids[image]

        if self.measurements.shape[0] == 0 or self.measurements.iloc[0]['state'] <= 0: #self.measurements.iat[image_id, 3] < 0:
            return 0
        snow_height = []
        for stick in self.sticks:
            stick_id = self.stick_labels_column_ids[stick.label]
            if not self.measurements.iat[image_id, stick_id + PD_STICK_VISIBLE]:
                continue
            qu = stick.length_px * 0.25
            quarter_id = int(round(max(self.measurements.iat[image_id, stick_id + PD_STICK_SNOW_HEIGHT], 0) / qu))
            snow_height.append(quarter_id)

        return 0 if len(snow_height) == 0 else int(np.median(snow_height))

    def generate_bounding_boxes(self):
        sticks = self.sticks.copy()
        sticks.sort(key=lambda s: s.bottom[0])
        for i, stick in enumerate(sticks):
            left = min(stick.top[0], stick.bottom[0])
            right = max(stick.top[0], stick.bottom[0])
            if i > 0:
                left_neigh = sticks[i - 1]
                left_dist = left - max(left_neigh.top[0], left_neigh.bottom[0])
                x1 = min(int(np.round(0.5 * left_dist)), 17)  # max(int(0.3 * left_dist), 17)
            else:
                x1 = 17
            if i < len(sticks) - 1:
                right_neigh = sticks[i + 1]
                right_dist = min(right_neigh.top[0], right_neigh.bottom[0]) - right
                x2 = min(int(np.round(0.5 * right_dist)), 17)
            else:
                x2 = 17
            stick.bbox = np.array([
                [left - x1, stick.top[1] - 17],
                [right + x2, stick.bottom[1] + 17]
            ])
            stick.bbox_left_range = x1
            stick.bbox_right_range = x2

    def insert_timestamps(self, timestamps: pd.Series):
        if 'date_time' in self.measurements.columns:
            self.measurements['date_time'] = timestamps
        else:
            self.measurements.insert(0, 'date_time', timestamps)
        self.measurements.insert(1, 'orig_date_time', timestamps)
        self.measurements = self.measurements.set_index('date_time', drop=False)
        self.timestamps_available = True
        self.check_for_temporal_monotonicity()
        self.save_measurements()

    def skipped_image(self, img: str):
        img_id = self.image_names_ids[img]
        self.measurements.iat[img_id, PD_IMAGE_STATE] = PhotoState.Skipped

    def check_for_temporal_monotonicity(self):
        if not self.measurements['date_time'].is_monotonic:
            diff = self.measurements['date_time'].diff()
            problem_id = diff.argmin()
            last_correct_img = self.image_list[problem_id - 1]
            last_correct_date = self.measurements['date_time'].iloc[problem_id - 1]
            problem_img = self.image_list[problem_id]
            problem_date = self.measurements['date_time'].iloc[problem_id]
            proposed_date = last_correct_date + (
                    self.measurements['date_time'].iloc[problem_id - 1] - self.measurements['date_time'].iloc[problem_id - 2])
            self.non_increasingness.emit((last_correct_img, last_correct_date, problem_img, problem_date, proposed_date))
        else:
            self.save_measurements()

    def repair_timestamps(self):
        diff = self.measurements['date_time'].diff()
        problem_id = diff.argmin()
        last_correct_img = self.image_list[problem_id - 1]
        last_correct_date = self.measurements['date_time'].iloc[problem_id - 1]
        problem_img = self.image_list[problem_id]
        problem_date = self.measurements['date_time'].iloc[problem_id]
        proposed_date = last_correct_date + (
                self.measurements['date_time'].iloc[problem_id - 1] - self.measurements['date_time'].iloc[
            problem_id - 2])

        self.measurements.iat[problem_id, PD_DATE] = proposed_date
        offsets = self.measurements['orig_date_time'].iloc[problem_id:].diff()[1:]
        indices = self.measurements.index[problem_id+1:]
        # self.measurements['date_time'].iloc[problem_id+1:] = offsets.cumsum().add(proposed_date)
        #self.measurements.iat[problem_id + 1:, PD_DATE] = offsets.cumsum().add(proposed_date)
        self.measurements.loc[indices, 'date_time'] = offsets.cumsum().add(proposed_date)
        self.check_for_temporal_monotonicity()

    def __hash__(self):
        return self.folder.__hash__()

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()
