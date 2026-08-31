"""Control and simulation tools for the SERPUGA robot."""

from .config import MPCParameters, RobotParameters, SimulationParameters
from .kinematics import KinematicModel
from .robot import RobotDescription

__all__ = [
    "KinematicModel",
    "MPCParameters",
    "RobotDescription",
    "RobotParameters",
    "SimulationParameters",
]

__version__ = "0.4.0"
