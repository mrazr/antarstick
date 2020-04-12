from dataclasses import dataclass, field

import numpy as np
from numpy import ndarray


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

    id: int
    top: ndarray
    bottom: ndarray
    length_px: float = field(init=False)
    label: str = "stick"
    scale_: float = 1.0

    def __post_init__(self):
        self.length_px = np.linalg.norm(self.top - self.bottom)

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
        return Stick(self.id, factor * self.top, factor * self.bottom)
