from enum import Enum


class LPS(Enum):
    """
    Enum defining standard direction vectors in LPS (Left-Posterior-Superior) coordinate system.

    Each direction is represented as a 3-element tuple (x, y, z) in physical SimpleITK coordinates.
    """

    LEFT_TO_RIGHT = (-1, 0, 0)
    RIGHT_TO_LEFT = (1, 0, 0)
    POSTERIOR_TO_ANTERIOR = (0, -1, 0)
    ANTERIOR_TO_POSTERIOR = (0, 1, 0)
    INFERIOR_TO_SUPERIOR = (0, 0, 1)
    SUPERIOR_TO_INFERIOR = (0, 0, -1)
