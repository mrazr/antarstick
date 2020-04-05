from pathlib import Path
from typing import Dict, List, Tuple

import simplejson as json
import jsonpickle

from camera import Camera


class Dataset:
    def __init__(self, path: Path = None):
        self.path = path
        self.stick_translation_table: Dict[int, Tuple[int, int]] = dict({})
        self.cameras: List[Camera] = []
        self.next_camera_id = 0

    def add_camera(self, folder: Path):
        camera = Camera(folder, self.next_camera_id)
        if self.path:
                camera.measurements_path = self.path.parent / f"camera{camera.id}.csv"
        self.cameras.append(camera)
        self.next_camera_id += 1

    def save(self) -> bool:
        if not self.path:
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

    def load_from(path: Path) -> bool:
        with open(path, "r") as dataset_file:
            decoded = jsonpickle.decode(dataset_file.read())
            dataset = Dataset(decoded['path'])
            dataset.stick_translation_table = decoded['stick_translation_table']
            dataset.next_camera_id = decoded['next_camera_id']
            dataset.cameras = [Camera.build_from_state(camera_state) for camera_state in decoded['cameras']]

            return dataset
