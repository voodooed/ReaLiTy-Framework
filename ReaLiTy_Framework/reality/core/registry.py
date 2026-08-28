"""Name -> class plugin registry.

Every extension point (datasets, models, degradations, evaluators, transforms)
owns a :class:`Registry`. Adding an implementation means registering it in a new
module, never editing core pipeline code.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Type, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for duplicate registrations or unknown names."""


class Registry:
    """A namespaced mapping of registered names to classes."""

    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self._entries: Dict[str, type] = {}

    def register(
        self, name: str, cls: type | None = None, *, override: bool = False
    ) -> Callable[[Type[T]], Type[T]] | type:
        """Register ``cls`` under ``name``.

        Usable as a decorator (``@REGISTRY.register("kitti")``) or called
        directly (``REGISTRY.register("kitti", KittiAdapter)``).
        """
        if not name or not isinstance(name, str):
            raise RegistryError(f"{self.namespace}: registration name must be a non-empty string")

        def _do(target: type) -> type:
            if name in self._entries and not override:
                existing = self._entries[name]
                raise RegistryError(
                    f"{self.namespace}: '{name}' is already registered to "
                    f"{existing.__module__}.{existing.__qualname__}; "
                    f"pass override=True to replace it"
                )
            self._entries[name] = target
            return target

        if cls is not None:
            return _do(cls)
        return _do

    def get(self, name: str) -> type:
        """Resolve a registered name, with the available names in the error."""
        try:
            return self._entries[name]
        except KeyError:
            raise RegistryError(
                f"{self.namespace}: unknown name '{name}'. Available: {self.names() or ['<none>']}"
            ) from None

    def create(self, name: str, *args, **kwargs):
        """Resolve ``name`` and instantiate it with the given arguments."""
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        """Registered names, sorted."""
        return sorted(self._entries)

    def unregister(self, name: str) -> None:
        """Remove a registration (mainly for tests)."""
        self._entries.pop(name, None)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry({self.namespace!r}, entries={self.names()})"


#: Extension points. Implementations register themselves on import.
DATASETS = Registry("datasets")
MODELS = Registry("models")
DEGRADATIONS = Registry("degradations")
TRANSFORMS = Registry("transforms")
EVALUATORS = Registry("evaluators")

REGISTRIES: Dict[str, Registry] = {
    r.namespace: r for r in (DATASETS, MODELS, DEGRADATIONS, TRANSFORMS, EVALUATORS)
}


def get_registry(namespace: str) -> Registry:
    """Look up a registry by namespace name."""
    try:
        return REGISTRIES[namespace]
    except KeyError:
        raise RegistryError(
            f"unknown registry namespace '{namespace}'. Available: {sorted(REGISTRIES)}"
        ) from None
