"""Registry resolution semantics."""

import pytest

from reality.core.registry import (
    DATASETS,
    DEGRADATIONS,
    EVALUATORS,
    MODELS,
    REGISTRIES,
    TRANSFORMS,
    Registry,
    RegistryError,
    get_registry,
)


@pytest.fixture
def registry():
    return Registry("things")


class Thing:
    def __init__(self, *args):
        self.args = args


def test_register_and_get(registry):
    registry.register("thing", Thing)
    assert registry.get("thing") is Thing


def test_register_as_decorator(registry):
    @registry.register("decorated")
    class Decorated(Thing):
        pass

    assert registry.get("decorated") is Decorated
    # The decorator returns the class unchanged.
    assert Decorated.__name__ == "Decorated"


def test_create_instantiates(registry):
    registry.register("thing", Thing)
    obj = registry.create("thing", 1, 2)
    assert isinstance(obj, Thing) and obj.args == (1, 2)


def test_unknown_name_lists_available(registry):
    registry.register("kitti", Thing)
    with pytest.raises(RegistryError) as exc:
        registry.get("kittii")
    message = str(exc.value)
    assert "things" in message and "kittii" in message and "kitti" in message


def test_duplicate_registration_rejected(registry):
    registry.register("thing", Thing)
    with pytest.raises(RegistryError, match="already registered"):
        registry.register("thing", Thing)


def test_duplicate_allowed_with_override(registry):
    class Other(Thing):
        pass

    registry.register("thing", Thing)
    registry.register("thing", Other, override=True)
    assert registry.get("thing") is Other


def test_empty_name_rejected(registry):
    with pytest.raises(RegistryError, match="non-empty string"):
        registry.register("", Thing)


def test_container_protocol(registry):
    registry.register("b", Thing)
    registry.register("a", Thing)
    assert "a" in registry and "z" not in registry
    assert len(registry) == 2
    assert registry.names() == ["a", "b"] == list(registry)


def test_unregister(registry):
    registry.register("thing", Thing)
    registry.unregister("thing")
    assert "thing" not in registry
    registry.unregister("thing")  # idempotent


def test_repr(registry):
    registry.register("thing", Thing)
    assert "things" in repr(registry) and "thing" in repr(registry)


def test_framework_registries_are_namespaced():
    expected = {"datasets", "models", "degradations", "transforms", "evaluators"}
    assert set(REGISTRIES) == expected
    for namespace, reg in REGISTRIES.items():
        assert reg.namespace == namespace
        assert get_registry(namespace) is reg
    assert (DATASETS, MODELS, DEGRADATIONS, TRANSFORMS, EVALUATORS) == tuple(
        get_registry(n) for n in ("datasets", "models", "degradations", "transforms", "evaluators")
    )


def test_unknown_namespace():
    with pytest.raises(RegistryError, match="unknown registry namespace"):
        get_registry("sensors")
