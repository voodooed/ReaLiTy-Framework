"""PhysicsWeatherDegradation — physics-based weather degradation.

Degrades a clear-weather point cloud before projection: Mie-scattering
attenuation, random scattering and point drop for a given precipitation rate,
emitting the physics-based intensity (``ref_new``) that the intensity model's
physics loss consumes.

The scattering model itself is bundled at ``reality/structure/weather_model`` and
is **derived from LISA (Kilic et al., https://github.com/velatkilic/LISA),
GPL-3.0** — see that directory's ``README.md`` and ``LICENSE.md``. This module is
only the wrapper: it locates a model directory, finds the model class by
capability rather than by name, and adapts it to the framework's Sample contract,
so a different scattering implementation can be substituted without touching the
pipeline (``docs/weather_model.md``).

Three properties of the bundled model are handled here rather than by editing it:

* **The Mie tables are loaded with a relative path**, so the model's directory is
  made the working directory for the duration of construction and the load
  succeeds from any CWD. A missing table is refused rather than silently
  recomputed, which is slow and does not reproduce the published coefficients.
* **Construction is the expensive part**, so one instance is built and reused
  across frames.
* **Lost points are moved to the origin** (range and reflectivity zero, label 0).
  They are not real returns: the projection's zero-range rule drops them, so they
  own no pixel and receive no generated intensity.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Iterator, Optional, Union

import numpy as np

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.registry import DEGRADATIONS
from reality.degradation.base import DegradationError, GeometricDegradation

#: Weather types the weather model's Monte-Carlo augmentation supports, mapped to atm_model.
WEATHER_TO_ATM_MODEL = {"rain": "rain", "snow": "snow"}
#: Environment fallback when the config does not carry a path.
WEATHER_MODEL_PATH_ENV = "REALITY_WEATHER_MODEL_PATH"
#: Vendoring slots, searched in order: the package slot, then a top-level weather_model/
#: directory beside the package. Dropping atmos_models.py and mie_q.npz in either
#: makes the weather path run real physics with no configuration at all.
VENDORED_MODEL = Path(__file__).resolve().parents[1] / "structure" / "weather_model"
REPO_MODEL = Path(__file__).resolve().parents[2] / "weather_model"
#: Default implementation: the vectorised torch one. ``atmos_models_cpu`` is the
#: original per-point CPU version, selectable via geometric_degradation.weather_model_module.
WEATHER_MODEL_MODULE = "atmos_models"
MIE_TABLE = "mie_q.npz"


class WeatherModelUnavailable(DegradationError):
    """Raised when the weather model cannot be located or imported."""


def find_weather_model(explicit: Union[str, Path, None] = None,
              module: str = WEATHER_MODEL_MODULE) -> Optional[Path]:
    """Locate the directory holding the weather model.

    ``reality/structure/weather_model`` is the canonical home; a config ``weather_model_path`` or
    ``$REALITY_WEATHER_MODEL_PATH`` overrides it, and a top-level ``weather_model/`` beside the
    package is accepted last so an unmoved checkout still works.
    """
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get(WEATHER_MODEL_PATH_ENV):
        candidates.append(Path(os.environ[WEATHER_MODEL_PATH_ENV]))
    candidates.extend((VENDORED_MODEL, REPO_MODEL))
    for candidate in candidates:
        directory = candidate if candidate.is_dir() else candidate.parent
        if (directory / f"{module}.py").is_file():
            return directory
    return None


def _install_scipy_compat() -> None:
    """Restore ``scipy.integrate.trapz`` for SciPy >= 1.14.

    SciPy renamed ``trapz`` to ``trapezoid`` and removed the old alias in 1.14.
    the weather model and its PyMieScatt dependency both still import the old name, so without
    this neither module imports at all. The two functions are the same routine
    under two names, so nothing about the physics changes -- and doing it here
    leaves the weather model's own source untouched.
    """
    import scipy.integrate

    if not hasattr(scipy.integrate, "trapz"):
        scipy.integrate.trapz = scipy.integrate.trapezoid


@contextlib.contextmanager
def _contained_import(directory: Path) -> Iterator[None]:
    """Import with ``directory`` on sys.path and as the CWD, then restore both."""
    _install_scipy_compat()
    previous_cwd = Path.cwd()
    preexisting = sys.modules.get(WEATHER_MODEL_MODULE)
    sys.path.insert(0, str(directory))
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(directory))
        if preexisting is not None:
            sys.modules[WEATHER_MODEL_MODULE] = preexisting
        else:
            sys.modules.pop(WEATHER_MODEL_MODULE, None)


def load_weather_model_class(directory: Path, module: str = WEATHER_MODEL_MODULE):
    """Import the weather model's class from ``directory``."""
    if not (directory / MIE_TABLE).is_file():
        # Not fatal upstream -- it recomputes the tables -- but it costs minutes
        # and the result is not the published table, so say so loudly.
        raise WeatherModelUnavailable(
            f"{directory} has no {MIE_TABLE}. the model would silently recompute the Mie "
            f"coefficients, which is slow and not the published table. Provide the "
            f"file shipped with the model."
        )
    with _contained_import(directory):
        loaded = importlib.import_module(module)
        if loaded.__file__ and not loaded.__file__.startswith(str(directory)):
            loaded = importlib.reload(loaded)  # a different model was already imported
        return _model_class(loaded, directory, module)


def _model_class(loaded, directory: Path, module: str):
    """Find the scattering model's class in a supplied module.

    The class is located by capability rather than by name, so any implementation
    exposing the documented ``augment(cloud, rate)`` interface can be dropped in
    (see docs/weather_model.md). A class literally named ``Model`` is preferred
    when several qualify.
    """
    # `augment` is commonly bound per-instance in __init__ (it dispatches on the
    # atmospheric model), so a class attribute check alone would miss it. Accept
    # any of the augmentation entry points.
    capabilities = ("augment", "augment_mc", "augment_avg")
    candidates = [obj for obj in vars(loaded).values()
                  if isinstance(obj, type) and obj.__module__ == loaded.__name__
                  and any(hasattr(obj, name) for name in capabilities)]
    if not candidates:
        raise WeatherModelUnavailable(
            f"{directory / module}.py exposes no class with an augment() method; "
            f"see docs/weather_model.md for the expected interface"
        )
    for candidate in candidates:
        if candidate.__name__ == "Model":
            return candidate
    return candidates[0]


@DEGRADATIONS.register("physics")
class PhysicsWeatherDegradation(GeometricDegradation):
    """Applies the weather model's weather model and emits its ``ref_new`` as the physics intensity."""

    name = "physics_weather"

    def __init__(self, config: Optional[Config] = None, *,
                 weather_model_path: Union[str, Path, None] = None, weather_model=None) -> None:
        super().__init__(config)
        spec = self.spec
        if spec is None or not spec.enabled:
            raise DegradationError(
                "physics degradation selected but geometric_degradation is not enabled"
            )
        weather = (spec.weather or "").lower()
        if weather not in WEATHER_TO_ATM_MODEL:
            raise DegradationError(
                f"geometric_degradation.weather: the weather model's Monte-Carlo augmentation "
                f"supports {sorted(WEATHER_TO_ATM_MODEL)}, got {spec.weather!r}"
            )
        self.atm_model = WEATHER_TO_ATM_MODEL[weather]
        self.precipitation_rate = float(spec.precipitation_rate)
        self.weather_model_module = getattr(spec, "weather_model_module", WEATHER_MODEL_MODULE)
        self.weather_model_path = find_weather_model(weather_model_path or getattr(spec, "weather_model_path", None),
                                   module=self.weather_model_module)
        self._weather_model = weather_model  # injectable, so the wiring is testable without the model present
        self._seeded = False

    # -- the model ------------------------------------------------------------- #

    @property
    def available(self) -> bool:
        """True when the weather model can be loaded; checked before it is needed lazily."""
        return self._weather_model is not None or self.weather_model_path is not None

    @property
    def weather_model(self):
        """The weather-model instance, built once and reused (Mie setup is the cost)."""
        if self._weather_model is None:
            if self.weather_model_path is None:
                raise WeatherModelUnavailable(
                    f"weather model not found. Expected {self.weather_model_module}.py and {MIE_TABLE} in "
                    f"{VENDORED_MODEL}, or set geometric_degradation.weather_model_path / "
                    f"{WEATHER_MODEL_PATH_ENV}."
                )
            spec = self.spec
            model_cls = load_weather_model_class(self.weather_model_path, self.weather_model_module)
            with _contained_import(self.weather_model_path):
                self._weather_model = model_cls(
                    atm_model=self.atm_model, mode=spec.mode,
                    rmax=spec.rmax, rmin=spec.rmin, bdiv=spec.bdiv,
                    saved_model=True,
                )
        return self._weather_model

    # -- the stage --------------------------------------------------------------- #

    def apply(self, sample: Sample) -> Sample:
        """Degrade the cloud and attach the physics intensity the model computed."""
        if sample.range_image is not None:
            raise DegradationError(
                "the weather model degrades the 3D cloud and must run before projection; this "
                "sample is already projected."
            )
        cloud = sample.raw_cloud().astype(np.float64)
        intensity = cloud[:, 3]
        if intensity.size and (intensity.min() < 0.0 or intensity.max() > 1.0):
            raise DegradationError(
                f"the weather model requires reflectivity normalised to [0, 1] and range in metres; "
                f"this cloud's intensity spans [{intensity.min():.3f}, {intensity.max():.3f}]"
            )

        degraded = np.asarray(self.weather_model.augment(cloud, self.precipitation_rate))
        if degraded.shape != (cloud.shape[0], 5):
            raise DegradationError(
                f"the weather model.augment returned {degraded.shape}, expected "
                f"({cloud.shape[0]}, 5) as [x, y, z, ref_new, label]"
            )

        physics_intensity = degraded[:, 3].astype(np.float32)
        labels = degraded[:, 4].astype(np.float32)
        out = self.attach(sample, degraded[:, :3].astype(np.float32),
                          physics_intensity, labels)
        out.meta.extra["degradation"] = {
            "type": self.name,
            "weather": self.atm_model,
            "precipitation_rate": self.precipitation_rate,
            "n_lost": int((labels == 0).sum()),
            "n_scattered": int((labels == 1).sum()),
            "n_unchanged": int((labels == 2).sum()),
        }
        return out
