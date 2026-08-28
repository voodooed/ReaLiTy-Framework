"""Assemble and run the stage sequence described by a config.

The stage order comes from the config alone (README -> *Pipeline*)::

    projection -> [geometric degradation] -> model -> back-projection

Sensor transfer skips the degradation stage; weather transfer includes it,
because that stage is what produces the physics intensity the model consumes.
Stages are resolved through the registries, so adding or swapping a stage never
requires editing this module.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from reality.core.config import Config
from reality.core.context import Sample
from reality.core.registry import DEGRADATIONS, MODELS, TRANSFORMS, Registry


class Stage(ABC):
    """One step of the pipeline. Takes a :class:`Sample`, returns a :class:`Sample`."""

    #: Human-readable stage name, used in logs and pipeline descriptions.
    name: str = "stage"

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config

    @abstractmethod
    def apply(self, sample: Sample) -> Sample:
        """Transform ``sample`` and return the result."""

    def __call__(self, sample: Sample) -> Sample:
        return self.apply(sample)


@dataclass(frozen=True)
class StageSpec:
    """A stage the config asks for, before it is resolved to a class."""

    #: What the stage does in the pipeline: projection, degradation, model, backprojection.
    role: str
    #: Registry namespace the implementation lives in.
    namespace: str
    #: Registered name of the implementation.
    name: str

    def __str__(self) -> str:
        return f"{self.role}({self.namespace}:{self.name})"


def plan_stages(config: Config) -> List[StageSpec]:
    """Return the ordered stage plan for ``config``, without resolving classes."""
    plan = []
    if config.geometric_degradation.enabled:
        # Degradation runs on the 3D cloud, before projection: the weather model models
        # scattering and point drop along each beam and consumes an (N, 4) cloud,
        # so it cannot operate on a range image. Its physics intensity rides along
        # as a point column and becomes Sample.phy when the degraded cloud is
        # projected. (Type is validated non-None whenever the stage is enabled.)
        plan.append(
            StageSpec("degradation", DEGRADATIONS.namespace, str(config.geometric_degradation.type))
        )
    plan.append(StageSpec("projection", TRANSFORMS.namespace, "projection"))
    plan.append(StageSpec("model", MODELS.namespace, config.model.type))
    plan.append(StageSpec("backprojection", TRANSFORMS.namespace, "backprojection"))
    return plan


class Pipeline:
    """An ordered sequence of stages applied to a sample."""

    def __init__(self, stages: Sequence[Stage] = ()) -> None:
        self.stages: List[Stage] = list(stages)

    @classmethod
    def from_config(
        cls, config: Config, registries: Optional[Dict[str, Registry]] = None
    ) -> "Pipeline":
        """Build a pipeline by resolving the config's stage plan through the registries.

        Raises :class:`~reality.core.registry.RegistryError` naming the stage that
        has no implementation registered yet.
        """
        from reality.core.registry import REGISTRIES, RegistryError

        table = REGISTRIES if registries is None else registries
        stages: List[Stage] = []
        for spec in plan_stages(config):
            registry = table.get(spec.namespace)
            if registry is None:
                raise RegistryError(
                    f"{spec}: no registry for namespace '{spec.namespace}'"
                )
            stages.append(registry.create(spec.name, config))
        return cls(stages)

    @property
    def stage_names(self) -> List[str]:
        return [getattr(s, "name", type(s).__name__) for s in self.stages]

    def run(self, sample: Sample) -> Sample:
        """Apply every stage in order. A pipeline with no stages is a no-op."""
        for stage in self.stages:
            sample = stage.apply(sample)
            if not isinstance(sample, Sample):
                raise TypeError(
                    f"stage {getattr(stage, 'name', type(stage).__name__)!r} returned "
                    f"{type(sample).__name__}, expected Sample"
                )
        return sample

    def run_all(self, samples: Iterable[Sample]) -> List[Sample]:
        """Run the pipeline over an iterable of samples."""
        return [self.run(s) for s in samples]

    def __len__(self) -> int:
        return len(self.stages)

    def __repr__(self) -> str:
        return f"Pipeline({' -> '.join(self.stage_names) or '<empty>'})"
