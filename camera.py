from os import listdir
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import cv2 as cv

from stick import Stick


class Camera:
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
    def __init__(self, folder: Path, _id: int, measurements_path: Optional[Path] = None):
        self.folder = Path(folder)
        self.sticks: List[Stick] = []
        self.id = _id
        if measurements_path:
            self.measurements_path = measurements_path
            self.measurements = self.__load_measuremets()
        else:
            self.measurements = pd.DataFrame()
            self.measurements_path = None
        self.rep_image_path = self.folder / Path(listdir(self.folder)[0]) #TODO listdir - filter out non image files
        self.rep_image: np.ndarray = cv.resize(cv.imread(str(self.rep_image_path)), (0, 0), fx=0.25, fy=0.25,
                                               interpolation=cv.INTER_NEAREST)

    def __load_measuremets(self) -> pd.DataFrame:
        try:
            with open(self.measurements_path) as meas_file:
                return pd.read_csv(meas_file)
        except FileNotFoundError:
            pass
        return pd.DataFrame()

    def save_measurements(self, path: Path):
        self.measurements_path = path
        with open(path, "w") as output:
            self.measurements.to_csv(output)

    def get_state(self):
        state = self.__dict__.copy()
        del state['measurements']
        del state['rep_image']
        del state['rep_image_path']
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

        return camera
