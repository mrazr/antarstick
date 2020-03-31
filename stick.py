from dataclasses import dataclass
from numpy import ndarray
from copy import deepcopy

@dataclass
class Stick:
    """
    A class for representing a stick

    ...

    Attributes
    ----------
    id : int
        a stick id unique only within a dataset folder
    label : str
        custom label for displaying purposes
    top : ndarray
        a (1, 2)-shaped numpy array representing top endpoint of the stick
    bottom : ndarray
        a (1, 2)-shaped numpy array representing bottom endpoint of the stick
    scale_ : float
        current scale of the stick. There might be situations when we want to
        down/upscale the stick for whatever reasons
    
    Methods
    -------
    scale(factor: float)
        returns a new Stick object constructed by scaling `self` by the factor `factor`
    """

    id: int
    label: str
    top: ndarray
    bottom: ndarray
    scale_: float = 1.0

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
        stick2 = deepcopy(self)
        stick2.top *= factor
        stick2.bottom *= factor
        stick2.scale_ = factor
        return stick2
    
