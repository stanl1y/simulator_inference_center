"""Backend registry -- maps simulator name strings to backend classes."""

from __future__ import annotations

from typing import Type

from simulator_inference_center.backend import SimulatorBackend

_REGISTRY: dict[str, Type[SimulatorBackend]] = {}


def register_backend(name: str, cls: Type[SimulatorBackend]) -> None:
    """Register a backend class under the given name."""
    _REGISTRY[name] = cls


def get_backend_class(name: str) -> Type[SimulatorBackend]:
    """Look up a backend class by name. Raises KeyError if not found."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(
            f"Unknown backend {name!r}. Available backends: {available}"
        )
    return _REGISTRY[name]


def list_backends() -> list[str]:
    """Return the names of all registered backends."""
    return sorted(_REGISTRY.keys())


def _discover_backends() -> None:
    """Import built-in backend modules so they self-register."""
    try:
        from simulator_inference_center.backends import libero  # noqa: F401
    except ImportError:
        pass


_discover_backends()
