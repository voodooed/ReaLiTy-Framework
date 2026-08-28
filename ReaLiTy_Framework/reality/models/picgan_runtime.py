"""Import the frozen PICGAN package without inheriting its side effects.

PICGAN is imported, never edited (see the contributor guide). Two properties of the frozen code
have to be contained by the caller rather than fixed at the source:

1. ``config.py`` runs ``os.makedirs`` at import time against the relative path
   ``Trial/Output``, so importing it from an arbitrary working directory litters
   that directory. We import it with the process CWD temporarily moved to a
   caller-chosen root, so those directories land where ReaLiTy decides.
2. PICGAN's modules import each other by bare name (``import config``,
   ``from dataset import LidarDataset``), which needs its directory on
   ``sys.path`` and would otherwise leave generic names like ``config`` and
   ``dataset`` occupying ``sys.modules`` for the whole process. We add the path
   and remove those entries again once the imports have resolved; the module
   objects stay reachable through the references PICGAN's own modules hold.

Everything after import is done by setting attributes on the imported ``config``
module, which the contributor guide names as the approved way to drive PICGAN.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, Optional, Union

#: The frozen model, vendored unchanged.
PICGAN_DIR = Path(__file__).resolve().parent / "PICGAN"

#: Modules ReaLiTy drives. ``utils`` is deliberately absent: it pulls in
#: matplotlib and writes images, and ReaLiTy owns checkpointing and outputs.
PICGAN_MODULES = ("config", "transform_utils", "generator", "discriminator", "dataset", "train")


class PicganImportError(RuntimeError):
    """Raised when the frozen PICGAN package cannot be imported."""


@dataclass(frozen=True)
class Picgan:
    """Handles on the imported PICGAN modules."""

    config: ModuleType
    transform_utils: ModuleType
    generator: ModuleType
    discriminator: ModuleType
    dataset: ModuleType
    train: ModuleType

    @property
    def Generator(self):
        return self.generator.Generator

    @property
    def Discriminator(self):
        return self.discriminator.Discriminator

    @property
    def LidarDataset(self):
        return self.dataset.LidarDataset

    @property
    def train_fn(self):
        return self.train.train_fn


@contextlib.contextmanager
def _contained_import(cwd: Path) -> Iterator[None]:
    """Run imports with PICGAN on sys.path and the CWD at ``cwd``, then restore."""
    previous_cwd = Path.cwd()
    preexisting = {name: sys.modules[name] for name in PICGAN_MODULES if name in sys.modules}
    cwd.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(PICGAN_DIR))
    os.chdir(cwd)
    try:
        yield
    finally:
        os.chdir(previous_cwd)
        with contextlib.suppress(ValueError):
            sys.path.remove(str(PICGAN_DIR))
        for name in PICGAN_MODULES:
            # Drop the bare names again so 'config'/'dataset'/'train' do not stay
            # claimed process-wide; PICGAN's modules keep their own references.
            if name in preexisting:
                sys.modules[name] = preexisting[name]
            else:
                sys.modules.pop(name, None)


_LOADED: Optional[Picgan] = None


def load_picgan(output_root: Union[Path, str]) -> Picgan:
    """Import PICGAN, containing its import-time directory creation under ``output_root``.

    The import happens once per process; later calls reuse it. Paths are not
    configured here -- call :func:`inject_paths` for that.
    """
    global _LOADED
    if _LOADED is not None:
        return _LOADED
    if not PICGAN_DIR.is_dir():
        raise PicganImportError(f"frozen PICGAN package not found at {PICGAN_DIR}")

    modules: Dict[str, ModuleType] = {}
    with _contained_import(Path(output_root).resolve()):
        for name in PICGAN_MODULES:
            try:
                modules[name] = importlib.import_module(name)
            except ImportError as exc:  # e.g. torchvision absent
                raise PicganImportError(
                    f"could not import PICGAN's {name}.py: {exc}. PICGAN's dependencies "
                    f"(torch, torchvision) must be installed; PICGAN itself is frozen."
                ) from exc
    _LOADED = Picgan(**modules)
    return _LOADED


def inject_config(picgan: Picgan, **values) -> Dict[str, object]:
    """Set attributes on PICGAN's ``config`` module and return the previous values.

    This is configuration injection, not an edit to PICGAN: the frozen source is
    untouched and only the imported module object is rebound.
    """
    previous = {}
    for key, value in values.items():
        previous[key] = getattr(picgan.config, key, None)
        setattr(picgan.config, key, value)
    return previous


def inject_paths(picgan: Picgan, checkpoint_dir: Union[Path, str],
                 output_dir: Union[Path, str, None] = None) -> None:
    """Point every PICGAN output path at absolute, ReaLiTy-resolved locations.

    PICGAN's defaults are relative (``Trial/Output/...``), so a run driven from a
    different working directory would write somewhere else. Absolute paths make
    the model's file behaviour independent of the process CWD.
    """
    checkpoint_dir = Path(checkpoint_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir else checkpoint_dir / "outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    trial = getattr(picgan.config, "Trial_Num", "T1")
    inject_config(
        picgan,
        Trial_Path=checkpoint_dir,
        OUTPUT_FOLDER=output_dir,
        CHECKPOINT_GEN_S=str(checkpoint_dir / f"gen_s.pth.tar_{trial}"),
        CHECKPOINT_GEN_R=str(checkpoint_dir / f"gen_r.pth.tar_{trial}"),
        CHECKPOINT_DISC_S=str(checkpoint_dir / f"disc_s.pth.tar_{trial}"),
        CHECKPOINT_DISC_R=str(checkpoint_dir / f"disc_r.pth.tar_{trial}"),
    )


def select_device(requested: Optional[str] = None) -> str:
    """Pick a device, refusing a CUDA that reports available but cannot run kernels.

    ``torch.cuda.is_available()`` is True whenever a driver and device are present,
    even when the installed torch build ships no kernels for that GPU's compute
    capability -- every CUDA op then fails with ``no kernel image is available``.
    Checking the capability against the build's arch list catches that up front.
    """
    import torch

    if requested:
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    try:
        major, minor = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        supported = {int(a.removeprefix("sm_")) for a in arch_list if a.startswith("sm_")}
        if not any(code // 10 == major and code % 10 <= minor for code in supported):
            warnings.warn(
                f"CUDA reports available but this torch build ({torch.__version__}) has "
                f"no kernels for compute capability {major}.{minor} "
                f"(built for {sorted(arch_list)}); falling back to CPU.",
                RuntimeWarning, stacklevel=2,
            )
            return "cpu"
    except Exception:  # pragma: no cover - a probe must never be the thing that fails
        return "cpu"
    return "cuda"


def reset_for_tests() -> None:
    """Forget the cached import so a test can re-import under a fresh root."""
    global _LOADED
    _LOADED = None
