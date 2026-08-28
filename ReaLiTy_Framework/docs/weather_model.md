# The physics weather model

The weather path degrades a clear-weather point cloud before projection: it
applies Mie-scattering attenuation, random scattering and point drop for a given
precipitation rate, and emits the physics-based intensity (`ref_new`) that
PICGAN's physics loss consumes.

The model **ships with the repository** at `reality/structure/weather_model/`, so
the weather path runs with no setup. ReaLiTy calls it through a thin,
model-agnostic wrapper (`reality/degradation/physics_weather.py`), so the
scattering implementation remains a replaceable component.

It is derived from LISA (Kilic et al.), GPL-3.0 — see
`reality/structure/weather_model/README.md` for provenance and
[THIRD_PARTY.md](../THIRD_PARTY.md) for the licence position.

## Substituting your own

The wrapper looks for a directory containing the model module and `mie_q.npz`,
in this order:

1. `geometric_degradation.weather_model_path` in the run config
2. `$REALITY_WEATHER_MODEL_PATH`
3. `reality/structure/weather_model/` inside the package (the bundled default)

```bash
export REALITY_WEATHER_MODEL_PATH=/path/to/weather_model
```

The model is expected to expose a class with:

```python
Model(atm_model='snow', mode='strongest', rmax=200, rmin=1.5, bdiv=3e-3,
      saved_model=True)
model.augment(cloud, precipitation_rate) -> (N, 5)  # x, y, z, ref_new, label
#   label 0 = lost   1 = randomly scattered   2 = unscattered
```

`mie_q.npz` matters: without it the model recomputes the Mie coefficients on every
construction, which takes minutes and does not reproduce the published table. The
wrapper refuses to run rather than let that happen silently.

Two compatibility issues are handled in the wrapper so the model's own source
stays untouched: `scipy.integrate.trapz` was removed in SciPy 1.14 and is
re-aliased before import, and `PyMieScatt` must be installed even when the
precomputed table means its routines are never called.

## Writing your own

Subclass `GeometricDegradation`, populate the physics-intensity column, and
register it:

```python
from reality.core.registry import DEGRADATIONS
from reality.degradation.base import GeometricDegradation

@DEGRADATIONS.register("my_model")
class MyWeatherModel(GeometricDegradation):
    def apply(self, sample):
        ...  # return a Sample carrying physics_intensity
```

Select it with `geometric_degradation: {type: my_model}`. Because it returns the
same `Sample` shape, projection, the intensity model and back-projection are
unchanged.
