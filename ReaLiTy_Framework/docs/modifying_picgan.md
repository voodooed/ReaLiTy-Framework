# Modifying the intensity model

`reality/models/PICGAN/` holds the intensity-transfer network. Its **research
method** — the cycle-consistent GAN structure, the adversarial + cycle + physics
loss design, and the physics-informed formulation — is what the framework is built
around, and changing it changes the science.

Practical guidance for contributors:

**Safe to change.** Normalization constants and transform logic, data-loading
details, path handling and device handling. These are implementation plumbing; the
framework already drives them from config.

**Change deliberately.** The generator or discriminator architecture, the loss
terms or their weights, and the training objective. These define the method. If
you change them, say so in your results — a number produced by a different
objective is not comparable to one produced by this repository's.

**Extension points that need no edits at all:**

| you want to | do this instead |
|---|---|
| a different intensity model | implement `IntensityModel`, register it, select with `model: {type: ...}` |
| a different weather model | implement `GeometricDegradation` (see `docs/weather_model.md`) |
| a different dataset | add an adapter, or use `generic` with declared columns |
| a different metric | subclass `Evaluator` |

Several research options already ship behind flags, defaulting off:
`model.output_activation` (`tanh` | `sigmoid`), `model.lambda_wasserstein` (a
distributional generator term), `model.retro_channel`, and the training-stability
knobs `training.label_smoothing_real` and `training.disc_learning_rate`.
