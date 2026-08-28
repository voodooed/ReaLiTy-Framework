"""Geometric degradation plugins. Importing this package registers them."""

from reality.degradation.base import (
    LABEL_COLUMN,
    PHYSICS_COLUMN,
    DegradationError,
    GeometricDegradation,
)
from reality.degradation.learned import LearnedDegradation
from reality.degradation.physics_weather import WeatherModelUnavailable, PhysicsWeatherDegradation

__all__ = [
    "GeometricDegradation", "DegradationError", "PHYSICS_COLUMN", "LABEL_COLUMN",
    "PhysicsWeatherDegradation", "WeatherModelUnavailable", "LearnedDegradation",
]
