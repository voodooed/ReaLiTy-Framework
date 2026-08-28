# Physics weather-degradation model

The scattering model behind the weather path. It applies Mie-scattering
attenuation, random scattering and point drop for a given precipitation rate, and
emits the physics-based intensity (`ref_new`) that the intensity model's physics
loss consumes.

| file | provenance |
|---|---|
| `atmos_models.py` | **Modified** from upstream: `augment_mc` rewritten as a vectorised, chunked PyTorch/GPU implementation. Scattering physics unchanged. Used by default. |
| `atmos_models_cpu.py` | **Unmodified** upstream implementation, retained as a CPU fallback. Select with `geometric_degradation: {weather_model_module: atmos_models_cpu}`. |
| `mie_q.npz` | Precomputed Mie coefficients. Without it the model recomputes them on every construction — slow, and not the published table. |
| `mie_plots.py` | Plotting helper from upstream; not used by the pipeline. |
| `LICENSE.md` | GPL-3.0, the licence this code is distributed under. |

## Attribution

Derived from **LISA (Lidar Light Scattering Augmentation)**, Kilic et al. —
<https://github.com/velatkilic/LISA>, GPL-3.0.

> V. Kilic, D. Hegde, V. Sindagi, A. B. Cooper, M. A. Foster and V. M. Patel,
> *Lidar Light Scattering Augmentation (LISA): Physics-based Simulation of Adverse
> Weather Conditions for 3D Object Detection*, arXiv:2107.07004.

Because this code is bundled here, the combined work is distributed under GPL-3.0.

## Using a different model

ReaLiTy calls this through a model-agnostic wrapper
(`reality/degradation/physics_weather.py`) that locates a model by directory and
finds its class by capability. To substitute your own, point
`geometric_degradation.weather_model_path` or `$REALITY_WEATHER_MODEL_PATH` at a
directory containing a module that exposes:

```python
Model(atm_model='snow', mode='strongest', rmax=200, rmin=1.5, bdiv=3e-3,
      saved_model=True)
model.augment(cloud, precipitation_rate) -> (N, 5)   # x, y, z, ref_new, label
```

See `docs/weather_model.md`.
