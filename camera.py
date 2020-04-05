from pathlib import Path
from typing import List

import jsonpickle
import pandas as pd

from stick import Stick


class Camera:
    def __init__(self, folder: Path, id: int, measurements_path: Path):
        self.folder = Path(folder)
        self.sticks: List[Stick] = []
        self.id = id
        if measurements_path:
            self.measurements_path = measurements_path
            self.measurements = self.__load_measuremets()
        else:
            self.measurement = pd.DataFrame()
            self.measurements_path = None

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
        return state

    def build_from_state(state):
        path = state['folder']
        sticks = state['sticks']
        id = state['id']
        measurements_path = state['measurements_path']
        camera = Camera(path, id, measurements_path)
        camera.sticks = sticks

        return camera
