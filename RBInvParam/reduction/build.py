from __future__ import annotations
from typing import Any, List

from RBInvParam.reduction.base import BaseIPReductor
from RBInvParam.reduction.registry import get_reductor_class

def build_reductor(*, FOM: Any, active_bases: List[str], config: Any) -> BaseIPReductor:
    """
    Build reductor based on config.type and config.ctor_kwargs.
    """
    reductor_cls = get_reductor_class(config.type)
    return reductor_cls(
        FOM,
        active_bases=active_bases,
        config=config,
        **getattr(config, "ctor_kwargs", {}),
    )