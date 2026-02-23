from __future__ import annotations

from typing import Callable, Dict, Type

# ---- Registry: name -> class
_REDUCTOR_REGISTRY: Dict[str, Type["BaseIPReductor"]] = {}

def register_reductor(name: str) -> Callable[[Type["BaseIPReductor"]], Type["BaseIPReductor"]]:
    """
    Decorator to register a reductor implementation under a string key.
    Typically used in problem packages via import side-effects.
    """
    def deco(cls: Type["BaseIPReductor"]) -> Type["BaseIPReductor"]:
        if name in _REDUCTOR_REGISTRY and _REDUCTOR_REGISTRY[name] is not cls:
            raise ValueError(f"Reductor '{name}' already registered with {_REDUCTOR_REGISTRY[name]}")
        _REDUCTOR_REGISTRY[name] = cls
        cls.reductor_name = name  # optional convenience
        return cls
    return deco

def get_reductor_class(name: str) -> Type["BaseIPReductor"]:
    try:
        return _REDUCTOR_REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REDUCTOR_REGISTRY.keys()))
        raise KeyError(f"Unknown reductor '{name}'. Available: [{available}]")

def list_reductors() -> Dict[str, Type["BaseIPReductor"]]:
    return dict(_REDUCTOR_REGISTRY)