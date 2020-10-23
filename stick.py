from dataclasses import dataclass, field
from typing import Any, Dict, List

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
    #stick_views: List['Stick']
    view: str
    top: ndarray = np.zeros((2,), np.int32)
    bottom: ndarray = np.zeros((2,), np.int32)
    camera_id: int = -1
    id: int = -1
    length_px: float = field(init=False)
    length_cm: int = 60
    snow_height_cm: int = 0
    snow_height_px: int = 0
    label: str = "stick"
    scale_: float = 1.0
    width: int = 3

    def __post_init__(self):
        self.length_px = np.linalg.norm(self.top - self.bottom)
        self.label = "s" + str(self.local_id)
        self.stick_views = []

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
        return Stick(self.local_id, self.view, (factor * self.top).astype(np.int32),
                     (factor * self.bottom).astype(np.int32), label=self.label, scale_=factor*self.scale_)
    
    def set_endpoints(self, x1: int, y1: int, x2: int, y2: int):
        self.top = np.array([x1, y1], np.int32)
        self.bottom = np.array([x2, y2], np.int32)
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
            'scale_': self.scale_,
            'view': self.view,
            'width': self.width
        }

    def line(self) -> np.ndarray:
        return np.array([self.top, self.bottom])

    def __eq__(self, other: 'Stick') -> bool:
        return self.camera_id == other.camera_id and self.local_id == other.local_id

    def __hash__(self) -> int:
        return (str(self.camera_id) + str(self.local_id)).__hash__()

    @staticmethod
    def build_from_state(state: Dict[str, Any]) -> 'Stick':
        local_id = state['id']
        #camera_id = state['camera_id']
        top = np.array(state['top'])
        bottom = np.array(state['bottom'])

        stick = Stick(local_id, view='', top=top, bottom=bottom)
        stick.length_px = state['length_px']
        stick.length_cm = state['length_cm']
        stick.label = state['label']
        stick.scale_ = state['scale_']
        stick.view = state['view']
        stick.width = state['width']

        return stick
