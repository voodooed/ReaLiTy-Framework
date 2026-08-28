"""The single bridge between a ReaLiTy :class:`Sample` and the frozen PICGAN.

Nothing under ``models/PICGAN/`` is modified. This adapter reproduces exactly what
PICGAN's own ``dataset.py`` produces, drives the model through runtime config
injection, and delegates the optimisation step to PICGAN's own ``train_fn`` so the
loss formulation is never restated here.

The contract, read off the frozen source rather than assumed:

``dataset.py.__getitem__`` indexes a 4-channel simulated stack and a 2-channel
real stack, and returns a 3-tuple::

    data_sim  = (4, H, W)   [0] range  [1] incidence  [2] reflectance  [3] phy
    data_real = (2, H, W)   [0] range (unused)        [1] intensity

    sim  = cat(lidar_transform(range),
               incidence_transform(incidence),
               reflectance_transform(reflectance))          -> (3, H, W)
    real = intensity_real_transform(data_real[1])           -> (1, H, W)
    phy  = intensity_sim_transform(data_sim[3])             -> (1, H, W)

    return sim, real, phy

``train.py.train_fn`` unpacks that tuple as ``(sim, real, phy)`` and never indexes
channels again, so the source stack's width is free: ``gen_R`` is built as
``Generator(img_channels=IN_CHANNELS_S, out_channels=IN_CHANNELS_R)`` and
``disc_S`` as ``Discriminator(in_channels=IN_CHANNELS_S)``. Both are constructor
arguments, so the no-labels 2-channel source needs no edit to PICGAN — see
``build_batch`` for how the reflectance channel is dropped.

Normaweather_modeltion constants are taken from PICGAN's ``transform_utils.py`` by using
those transform objects directly; they are never restated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.registry import MODELS
from reality.models.base import IntensityModel
from reality.preprocessing.statistics import NormalizationStats
from reality.models.picgan_runtime import (
    Picgan,
    inject_config,
    inject_paths,
    load_picgan,
    select_device,
)

#: Channel order of the simulated stack, exactly as ``dataset.py`` indexes it.
SIM_CHANNELS_WITH_REFLECTANCE = ("range", "incidence", "reflectance", "phy")
#: No-labels variant: reflectance is dropped, phy stays last.
SIM_CHANNELS_NO_REFLECTANCE = ("range", "incidence", "phy")
#: Channel order of the real stack; ``dataset.py`` reads index 1 and ignores 0.
REAL_CHANNELS = ("range", "intensity")


def wasserstein_1d(a: torch.Tensor, b: torch.Tensor, quantiles: int = 512
                   ) -> torch.Tensor:
    """Differentiable 1-D Wasserstein distance between two value sets.

    In one dimension W1 is the mean absolute difference between the two quantile
    functions. Both sides are sorted and read at a common grid of quantiles, with
    linear interpolation so the two sets need not be the same size -- which they
    are not, since occupancy differs per frame. Sorting is a permutation, so
    gradients flow back to the generated values.
    """
    a_sorted, _ = torch.sort(a.reshape(-1))
    b_sorted, _ = torch.sort(b.reshape(-1))

    def at_quantiles(values: torch.Tensor) -> torch.Tensor:
        positions = torch.linspace(0, values.numel() - 1, quantiles,
                                   device=values.device)
        low = positions.floor().long()
        high = positions.ceil().long()
        weight = (positions - low.to(positions.dtype)).to(values.dtype)
        return values[low] * (1 - weight) + values[high] * weight

    return (at_quantiles(a_sorted) - at_quantiles(b_sorted)).abs().mean()


class _TransformSet:
    """Attribute access over the transform dict PICGAN builds."""

    def __init__(self, transforms: Dict[str, Any]) -> None:
        self.__dict__.update(transforms)


class PicganAdapterError(RuntimeError):
    """Raised when a Sample cannot be expressed in PICGAN's input contract."""


@MODELS.register("picgan")
class PicganAdapter(IntensityModel):
    """Drives the frozen PICGAN from ReaLiTy Samples."""

    name = "picgan"

    def __init__(self, config: Optional[Config] = None, *,
                 workspace: Union[str, Path, None] = None,
                 device: Union[str, None] = None,
                 stats: Optional[NormalizationStats] = None) -> None:
        super().__init__(config)
        checkpoint_dir = (Path(config.output.checkpoint_dir)
                          if config is not None and config.output else Path("checkpoints/picgan"))
        self.workspace = Path(workspace) if workspace else checkpoint_dir
        self.picgan: Picgan = load_picgan(self.workspace)
        inject_paths(self.picgan, self.workspace)
        self.device = torch.device(select_device(device))
        inject_config(self.picgan, DEVICE=str(self.device))
        if config is not None:
            self.apply_training_config(config)

        self.stats = stats or NormalizationStats.picgan_default()
        self.output_activation = (config.model.output_activation
                                  if config is not None else "tanh")
        self.lambda_wasserstein = (config.model.lambda_wasserstein
                                   if config is not None else 0.0)
        self._transforms = None
        self.gen_R = self.gen_S = self.disc_R = self.disc_S = None
        self.opt_gen = self.opt_disc = None
        self.in_channels_s: Optional[int] = None
        self.in_channels_r: int = 1

    # -- configuration ------------------------------------------------------- #

    def apply_training_config(self, config: Config) -> None:
        """Push ReaLiTy's resolved training config into PICGAN's config module."""
        training = config.training
        inject_config(
            self.picgan,
            LEARNING_RATE=training.learning_rate,
            LAMBDA_CYCLE=training.lambda_cycle,
            LAMBDA_Physics=training.lambda_physics,
            BATCH_SIZE=training.batch_size,
            NUM_EPOCHS=training.epochs,
            NUM_WORKERS=training.num_workers,
        )

    def use_statistics(self, stats: NormalizationStats) -> None:
        """Rebuild the transforms from measured statistics."""
        self.stats = stats
        self._transforms = None

    @property
    def transforms(self):
        """The channel transforms, built by PICGAN from this run's statistics.

        Construction stays in PICGAN's ``transform_utils.build_transforms`` -- the
        normaweather_modeltion method lives there, only its constants come from here.
        """
        if self._transforms is None:
            pairs = dict(self.stats.as_pairs())
            if self.output_activation == "sigmoid":
                # gen_R now emits [0, 1] data units, and intensity is already on
                # that scale, so the target transform is the identity. Every
                # operand the discriminator and the cycle terms see moves with it.
                pairs["intensity"] = (0.0, 1.0)
            self._transforms = _TransformSet(
                self.picgan.transform_utils.build_transforms(pairs)
            )
        return self._transforms

    def normaweather_modeltion(self, transform_name: str) -> Tuple[float, float]:
        """Read (mean, std) out of one of the Compose transforms in use."""
        normalize = getattr(self.transforms, transform_name).transforms[1]
        return float(normalize.mean[0]), float(normalize.std[0])

    @property
    def distributional_loss(self):
        """Wasserstein term over occupied pixels, or None when disabled.

        Empty pixels carry no return, so both sides are restricted to occupied
        ones. Occupancy is recovered from the sentinel each channel takes where a
        pixel is empty: projection writes 0 there, so after normaweather_modeltion the
        empty value is ``(0 - mean) / std`` for that channel.
        """
        if self.lambda_wasserstein <= 0:
            return None
        range_mean, range_std = self.normaweather_modeltion("lidar_transform")
        intensity_mean, intensity_std = self.normaweather_modeltion("intensity_real_transform")
        range_empty = (0.0 - range_mean) / range_std
        intensity_empty = (0.0 - intensity_mean) / intensity_std

        def loss(fake_real, real, sim, phy):
            source_mask = sim[:, :1] > range_empty + 1e-4
            target_mask = real > intensity_empty + 1e-6
            generated = fake_real[source_mask]
            reference = real[target_mask]
            if generated.numel() < 2 or reference.numel() < 2:
                return fake_real.sum() * 0.0
            return self.lambda_wasserstein * wasserstein_1d(generated, reference)

        return loss

    @property
    def physics_transform(self):
        """Put ``fake_real`` back into z-space for the physics comparison only.

        The physics term compares the generated intensity's position within the
        target distribution against phy's position within the physics
        distribution; both operands must therefore be z-scored. With a tanh head
        the generator already emits z-scores and nothing is needed. With a sigmoid
        head it emits [0, 1] data units, so this applies the measured target
        statistics to that one operand, leaving the term's meaning identical.
        """
        if self.output_activation != "sigmoid":
            return None
        mean, std = self.stats.pair("intensity")

        def to_z(tensor: torch.Tensor) -> torch.Tensor:
            return (tensor - mean) / std

        return to_z

    def denormalize_real_intensity(self, tensor: torch.Tensor) -> torch.Tensor:
        """Return generated intensity to data units.

        Uses the *target* domain's intensity statistics -- the same ones the real
        intensity was normalised with -- rather than a constant fitted elsewhere.
        """
        mean, std = self.normaweather_modeltion("intensity_real_transform")
        return tensor * std + mean

    # -- Sample -> PICGAN arrays --------------------------------------------- #

    @staticmethod
    def sim_channels(has_reflectance: bool) -> Tuple[str, ...]:
        return SIM_CHANNELS_WITH_REFLECTANCE if has_reflectance else SIM_CHANNELS_NO_REFLECTANCE

    def source_stack(self, sample: Sample) -> np.ndarray:
        """Build the simulated stack in PICGAN's channel order.

        Shape is ``(4, H, W)`` with reflectance and ``(3, H, W)`` without; phy is
        always the last channel, as ``dataset.py`` reads it.
        """
        if sample.range_image is None:
            raise PicganAdapterError(
                f"{sample.meta.dataset}: no range_image; project the sample first"
            )
        if sample.phy is None:
            raise PicganAdapterError(
                f"{sample.meta.dataset}: phy is not set. PICGAN never computes the "
                f"physics intensity; it must be produced upstream (source simulator "
                f"for sensor transfer, the weather model for weather transfer)."
            )
        planes: List[np.ndarray] = []
        for channel in self.sim_channels(sample.meta.has_reflectance):
            if channel == "phy":
                planes.append(sample.phy[0])
            else:
                planes.append(sample.channel(channel))
        return np.stack(planes).astype(np.float32)

    def real_stack(self, sample: Sample) -> np.ndarray:
        """Build the ``(2, H, W)`` real stack: range then intensity."""
        if sample.range_image is None:
            raise PicganAdapterError(
                f"{sample.meta.dataset}: no range_image; project the sample first"
            )
        return np.stack([sample.channel(c) for c in REAL_CHANNELS]).astype(np.float32)

    # -- PICGAN arrays -> tensors (the parity-critical path) ------------------ #

    def to_tensors(self, sim_stack: np.ndarray, real_stack: np.ndarray
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reproduce ``LidarDataset.__getitem__`` on in-memory arrays.

        Mirrors the frozen code step for step: cast to float32, replace non-finite
        values, index the channels, apply PICGAN's own transforms, concatenate the
        source stack in range/incidence/[reflectance] order, and return
        ``(sim, real, phy)`` without a batch dimension.
        """
        tu = self.transforms
        data_sim = np.nan_to_num(np.asarray(sim_stack).astype(np.float32),
                                 nan=0.0, posinf=0.0, neginf=0.0)
        data_real = np.nan_to_num(np.asarray(real_stack).astype(np.float32),
                                  nan=0.0, posinf=0.0, neginf=0.0)
        if data_real.shape[0] < 2:
            raise PicganAdapterError(
                f"real stack must be (2, H, W) [range, intensity], got {data_real.shape}"
            )
        has_reflectance = data_sim.shape[0] == len(SIM_CHANNELS_WITH_REFLECTANCE)
        if data_sim.shape[0] not in (len(SIM_CHANNELS_NO_REFLECTANCE),
                                     len(SIM_CHANNELS_WITH_REFLECTANCE)):
            raise PicganAdapterError(
                f"sim stack must have {len(SIM_CHANNELS_NO_REFLECTANCE)} or "
                f"{len(SIM_CHANNELS_WITH_REFLECTANCE)} channels, got {data_sim.shape[0]}"
            )

        planes = [tu.lidar_transform(data_sim[0]), tu.incidence_transform(data_sim[1])]
        if has_reflectance:
            planes.append(tu.reflectance_transform(data_sim[2]))
        sim = torch.cat(tuple(planes), dim=0)

        phy = tu.intensity_sim_transform(data_sim[-1])
        real = tu.intensity_real_transform(data_real[1])
        return sim, real, phy

    def sample_to_tensors(self, source: Sample, target: Sample
                          ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(source, target)`` Samples -> the ``(sim, real, phy)`` tuple PICGAN consumes."""
        return self.to_tensors(self.source_stack(source), self.real_stack(target))

    def build_batch(self, sources: Sequence[Sample], targets: Sequence[Sample]
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Stack paired Samples into a batch on the model's device."""
        if len(sources) != len(targets):
            raise PicganAdapterError(
                f"paired batch expected, got {len(sources)} sources and {len(targets)} targets"
            )
        widths = {self.channels_for(s) for s in sources}
        if len(widths) > 1:
            raise PicganAdapterError(
                f"a batch mixes source stacks of {sorted(widths)} channels. Labelled and "
                f"unlabelled frames cannot share a batch: restrict the source to "
                f"sequences with consistent labelling (source.sequences)."
            )
        if self._built and widths and widths != {self.in_channels_s}:
            raise PicganAdapterError(
                f"model is built for in_channels_s={self.in_channels_s} but this batch "
                f"has {widths.pop()}-channel sources"
            )
        triples = [self.sample_to_tensors(s, t) for s, t in zip(sources, targets)]
        return tuple(  # type: ignore[return-value]
            torch.stack([t[i] for t in triples]).to(self.device) for i in range(3)
        )

    # -- model --------------------------------------------------------------- #

    def channels_for(self, sample: Sample) -> int:
        """Source-stack width implied by a Sample: 3 with reflectance, 2 without."""
        return 3 if sample.meta.has_reflectance else 2

    def build_model(self, in_channels_s: int, in_channels_r: int = 1) -> None:
        """Construct PICGAN's generators, discriminators and optimisers.

        Channel counts are injected into PICGAN's config and passed as constructor
        arguments, which is what lets ``in_channels_s=2`` work unmodified.
        """
        if in_channels_s not in (2, 3, 4):
            raise PicganAdapterError(
                f"in_channels_s must be 2 (no reflectance), 3 (with reflectance) or "
                f"4 (with reflectance and retro channel), got {in_channels_s}"
            )
        inject_config(self.picgan, IN_CHANNELS_S=in_channels_s, IN_CHANNELS_R=in_channels_r)
        cfg = self.picgan.config
        Generator, Discriminator = self.picgan.Generator, self.picgan.Discriminator

        # gen_R's head is selectable; gen_S always keeps tanh because it emits
        # z-scored source channels that are legitimately negative.
        self.gen_R = Generator(img_channels=in_channels_s, out_channels=in_channels_r,
                               num_residuals=9,
                               output_activation=self.output_activation).to(self.device)
        self.gen_S = Generator(img_channels=in_channels_r, out_channels=in_channels_s,
                               num_residuals=9, output_activation="tanh").to(self.device)
        self.disc_R = Discriminator(in_channels=in_channels_r).to(self.device)
        self.disc_S = Discriminator(in_channels=in_channels_s).to(self.device)

        # The discriminator may run at its own learning rate; None follows the
        # generator's, which is the original behaviour.
        disc_lr = (self.config.training.disc_learning_rate
                   if self.config is not None
                   and self.config.training.disc_learning_rate else cfg.LEARNING_RATE)
        self.disc_learning_rate = disc_lr
        self.opt_disc = torch.optim.Adam(
            list(self.disc_S.parameters()) + list(self.disc_R.parameters()),
            lr=disc_lr, betas=(0.5, 0.999),
        )
        self.opt_gen = torch.optim.Adam(
            list(self.gen_R.parameters()) + list(self.gen_S.parameters()),
            lr=cfg.LEARNING_RATE, betas=(0.5, 0.999),
        )
        self.l1 = torch.nn.L1Loss()
        self.mse = torch.nn.MSELoss()
        use_amp = self.device.type == "cuda"
        self.g_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.d_scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        self.in_channels_s, self.in_channels_r = in_channels_s, in_channels_r
        self._built = True

    def build_for(self, sample: Sample) -> None:
        """Build with the channel count this Sample implies."""
        self.build_model(self.channels_for(sample))

    # -- training and inference ---------------------------------------------- #

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                   ) -> Dict[str, float]:
        """Delegate one step to PICGAN's own ``train_fn``.

        The loss formulation stays where it is; ``train_fn`` iterates a loader, so a
        single batch is handed to it as a one-element sequence. Nothing about the
        adversarial, cycle or physics terms is restated in ReaLiTy.
        """
        self._require_built()
        sim, real, phy = batch
        terms = self.picgan.train_fn(
            self.disc_S, self.disc_R, self.gen_R, self.gen_S,
            [(sim, real, phy)], self.opt_disc, self.opt_gen,
            self.l1, self.mse, self.d_scaler, self.g_scaler,
            phy_transform=self.physics_transform,
            extra_generator_loss=self.distributional_loss,
            real_label=(self.config.training.label_smoothing_real
                        if self.config is not None else 1.0),
        ) or {}
        return {"batches": 1, "source_channels": float(self.in_channels_s), **terms}

    @torch.no_grad()
    def generate(self, source: Union[Sample, torch.Tensor],
                 target: Optional[Sample] = None) -> torch.Tensor:
        """Map a source stack to target-domain intensity with ``gen_R``.

        ``gen_R`` is the generator used at inference (README -> *PICGAN's role*).
        """
        self._require_built()
        if isinstance(source, Sample):
            sim_stack = self.source_stack(source)
            # A real stack is only needed for the loss; feed zeros for inference.
            real_stack = np.zeros((2,) + sim_stack.shape[1:], dtype=np.float32)
            sim, _, _ = self.to_tensors(sim_stack, real_stack)
            sim = sim.unsqueeze(0)
        else:
            sim = source if source.dim() == 4 else source.unsqueeze(0)
        self.gen_R.eval()
        return self.gen_R(sim.to(self.device))

    # -- weights -------------------------------------------------------------- #

    def state_dict(self) -> Dict[str, Any]:
        self._require_built()
        return {
            "gen_R": self.gen_R.state_dict(), "gen_S": self.gen_S.state_dict(),
            "disc_R": self.disc_R.state_dict(), "disc_S": self.disc_S.state_dict(),
            "in_channels_s": self.in_channels_s, "in_channels_r": self.in_channels_r,
            "output_activation": self.output_activation,
        }

    def save_weights(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        return path

    def load_weights(self, path: Union[str, Path]) -> None:
        """Load weights, building the networks for the checkpoint's channel count."""
        state = torch.load(str(path), map_location=self.device, weights_only=False)
        if not self._built:
            self.build_model(int(state.get("in_channels_s", 3)),
                             int(state.get("in_channels_r", 1)))
        elif int(state.get("in_channels_s", self.in_channels_s)) != self.in_channels_s:
            raise PicganAdapterError(
                f"checkpoint was trained with in_channels_s="
                f"{state['in_channels_s']}, model is built for {self.in_channels_s}"
            )
        for key in ("gen_R", "gen_S", "disc_R", "disc_S"):
            getattr(self, key).load_state_dict(state[key])

    def __repr__(self) -> str:
        built = (f"in_channels_s={self.in_channels_s}" if self._built else "not built")
        return f"PicganAdapter({built}, device={self.device})"
