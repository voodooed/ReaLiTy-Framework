"""Load the material reflectance LUT and map semantic labels to reflectance.

The LUT (``reflectance_lut.yaml``) is two layers: a generic material table and
per-dataset class maps. Labels resolve class id -> material -> reflectance, so a
new labelled dataset needs a class map, not new reflectance values.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import yaml

DEFAULT_LUT_PATH = Path(__file__).with_name("reflectance_lut.yaml")


class ReflectanceLUTError(ValueError):
    """Raised when the LUT file is malformed."""


@dataclass(frozen=True)
class Material:
    """One material's reflectance and its provenance."""

    name: str
    reflectance: float
    source: str
    justification: str = ""
    measured_pct: Optional[float] = None
    anchor: Optional[str] = None


class ReflectanceLUT:
    """Material reflectance at ~905 nm, keyed by dataset class id."""

    def __init__(self, materials: Dict[str, Material], class_maps: Dict[str, Dict[int, str]],
                 class_names: Dict[str, Dict[int, str]], default_material: str = "unknown") -> None:
        self.materials = materials
        self.class_maps = class_maps
        self.class_names = class_names
        self.default_material = default_material
        if default_material not in materials:
            raise ReflectanceLUTError(
                f"default_material '{default_material}' is not defined in materials"
            )

    # -- loading ------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Union[str, Path] = DEFAULT_LUT_PATH) -> "ReflectanceLUT":
        """Parse and validate the LUT YAML."""
        path = Path(path)
        if not path.is_file():
            raise ReflectanceLUTError(f"reflectance LUT not found: {path}")
        raw = yaml.safe_load(path.read_text()) or {}
        if not isinstance(raw, dict) or "materials" not in raw:
            raise ReflectanceLUTError(f"{path}: expected a mapping with a 'materials' section")

        materials: Dict[str, Material] = {}
        for name, entry in (raw["materials"] or {}).items():
            if not isinstance(entry, dict):
                raise ReflectanceLUTError(f"materials.{name}: expected a mapping")
            if "reflectance" not in entry:
                raise ReflectanceLUTError(f"materials.{name}: missing 'reflectance'")
            value = entry["reflectance"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ReflectanceLUTError(f"materials.{name}.reflectance: must be a number")
            if not 0.0 <= float(value) <= 1.0:
                raise ReflectanceLUTError(
                    f"materials.{name}.reflectance: {value} is outside [0, 1]"
                )
            if not str(entry.get("source", "")).strip():
                raise ReflectanceLUTError(f"materials.{name}: 'source' must be non-empty")
            materials[name] = Material(
                name=name, reflectance=float(value), source=str(entry["source"]),
                justification=str(entry.get("justification", "")).strip(),
                measured_pct=entry.get("measured_pct"), anchor=entry.get("anchor"),
            )

        class_maps: Dict[str, Dict[int, str]] = {}
        class_names: Dict[str, Dict[int, str]] = {}
        for dataset, mapping in (raw.get("class_maps") or {}).items():
            ids, names = {}, {}
            for class_id, entry in (mapping or {}).items():
                material = entry["material"] if isinstance(entry, dict) else entry
                if material not in materials:
                    raise ReflectanceLUTError(
                        f"class_maps.{dataset}.{class_id}: unknown material '{material}'"
                    )
                ids[int(class_id)] = material
                if isinstance(entry, dict) and "name" in entry:
                    names[int(class_id)] = str(entry["name"])
            class_maps[dataset] = ids
            class_names[dataset] = names

        return cls(materials, class_maps, class_names,
                   str(raw.get("default_material", "unknown")))

    # -- lookup -------------------------------------------------------------- #

    def material_for(self, class_id: int, dataset: str = "semantickitti") -> Material:
        """Resolve one class id to its material, falling back to the default."""
        mapping = self.class_maps.get(dataset)
        if mapping is None:
            raise ReflectanceLUTError(
                f"no class map for dataset '{dataset}'; available: {sorted(self.class_maps)}"
            )
        name = mapping.get(int(class_id))
        if name is None:
            warnings.warn(
                f"class id {class_id} is not in the '{dataset}' class map; "
                f"falling back to '{self.default_material}' "
                f"({self.materials[self.default_material].reflectance})",
                RuntimeWarning, stacklevel=2,
            )
            name = self.default_material
        return self.materials[name]

    def reflectance_for(self, class_id: int, dataset: str = "semantickitti") -> float:
        """Reflectance of a single class id."""
        return self.material_for(class_id, dataset).reflectance

    def lookup(self, labels: np.ndarray, dataset: str = "semantickitti") -> np.ndarray:
        """Vectorised class id -> reflectance for an array of labels.

        Ids missing from the class map fall back to the default material, with one
        warning naming the offending ids rather than one per point.
        """
        mapping = self.class_maps.get(dataset)
        if mapping is None:
            raise ReflectanceLUTError(
                f"no class map for dataset '{dataset}'; available: {sorted(self.class_maps)}"
            )
        labels = np.asarray(labels)
        default = self.materials[self.default_material].reflectance
        out = np.full(labels.shape, default, dtype=np.float32)

        present = np.unique(labels)
        missing = [int(i) for i in present if int(i) not in mapping]
        if missing:
            warnings.warn(
                f"class id(s) {missing} not in the '{dataset}' class map; "
                f"using '{self.default_material}' ({default})",
                RuntimeWarning, stacklevel=2,
            )
        for class_id in present:
            name = mapping.get(int(class_id))
            if name is not None:
                out[labels == class_id] = self.materials[name].reflectance
        return out

    def class_name(self, class_id: int, dataset: str = "semantickitti") -> Optional[str]:
        """Human-readable class name, when the class map records one."""
        return self.class_names.get(dataset, {}).get(int(class_id))

    def to_dict(self) -> Dict:
        """Plain-data view; round-trips through :meth:`load`."""
        return {
            "default_material": self.default_material,
            "materials": {
                name: {k: v for k, v in {
                    "reflectance": m.reflectance, "measured_pct": m.measured_pct,
                    "source": m.source, "anchor": m.anchor,
                    "justification": m.justification,
                }.items() if v not in (None, "")}
                for name, m in self.materials.items()
            },
            "class_maps": {
                dataset: {
                    class_id: {"name": self.class_names[dataset].get(class_id),
                               "material": material}
                    if self.class_names[dataset].get(class_id) else {"material": material}
                    for class_id, material in mapping.items()
                }
                for dataset, mapping in self.class_maps.items()
            },
        }

    def __len__(self) -> int:
        return len(self.materials)

    def __repr__(self) -> str:
        return (f"ReflectanceLUT({len(self.materials)} materials, "
                f"class maps: {sorted(self.class_maps)})")


@lru_cache(maxsize=4)
def default_lut(path: Union[str, Path] = DEFAULT_LUT_PATH) -> ReflectanceLUT:
    """Load (and cache) the shipped LUT."""
    return ReflectanceLUT.load(path)
