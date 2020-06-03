from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numpy import ndarray

import json


@dataclass
class Stick:
    """
    A class for representing a stick

    ...

    Attributes
    ----------
    id : int
        a stick id unique within a Dataset(meaning unique within a number of cameras)
    label : str
        custom label for displaying purposes
    top : ndarray
        a (1, 2)-shaped numpy array representing top endpoint of the stick
    bottom : ndarray
        a (1, 2)-shaped numpy array representing bottom endpoint of the stick
    length_px: float
        length of the stick in pixels
    scale_ : float
        current scale of the stick. There might be situations when we want to
        down/upscale the stick for whatever reasons

    Methods
    -------
    scale(factor: float)
        returns a new Stick object constructed by scaling `self` by the factor `factor`
    """

    local_id: int
    top: ndarray
    bottom: ndarray
    camera_id: int = -1
    id: int = -1
    length_px: float = field(init=False)
    length_cm: int = 60
    label: str = "stick"
    scale_: float = 1.0

    def __post_init__(self):
        self.length_px = np.linalg.norm(self.top - self.bottom)
        self.label = str(self.id)

    def scale(self, factor: float):
        """
        Parameters
        ----------
        factor : float
            factor to scale the stick by

        Returns
        -------
        Stick
            a scaled version of the original stick
        """
        return Stick(self.local_id, factor * self.top, factor * self.bottom)
    
    def set_endpoints(self, y1: int, x1: int, y2: int, x2: int):
        self.top = np.array([x1, y1])
        self.bottom = np.array([x2, y2])
        self.__post_init__()

    def get_state(self) -> Dict[str, Any]:
        top = [int(self.top[0]), int(self.top[1])]
        bottom = [int(self.bottom[0]), int(self.bottom[1])]

        return {
            'id': self.local_id,
            #'camera_id': self.camera_id,
            'top': top,
            'bottom': bottom,
            'length_px': self.length_px,
            'length_cm': self.length_cm,
            'label': self.label,
            'scale_': self.scale_
        }

    @staticmethod
    def build_from_state(state: Dict[str, Any]) -> 'Stick':
        local_id = state['id']
        #camera_id = state['camera_id']
        top = np.array(state['top'])
        bottom = np.array(state['bottom'])

        stick = Stick(local_id, top, bottom)
        stick.length_px = state['length_px']
        stick.length_cm = state['length_cm']
        stick.label = state['label']
        stick.scale_ = state['scale_']

        return stick
