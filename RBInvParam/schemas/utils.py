from __future__ import annotations

from typing import Mapping, Iterable, Any



def unknown_keys(data: Mapping[str, Any], allowed: Iterable[str], *, where: str) -> None:
    unknown = set(data.keys()) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}")


def require(data: Mapping[str, Any], keys: Iterable[str], *, where: str) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"Missing keys in {where}: {missing}")