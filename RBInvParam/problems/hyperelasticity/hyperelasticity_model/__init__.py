
"""
Python bindings for the deal.II-based hyperelasticity material model.

Notes
-----
The C++ bindings accept an optional parameter vector `q_np` in
`HyperElasticityModel.assemble_A_q`. Passing `None` (or an empty 1D array)
signals that the parameter is unused and avoids converting/copying it into a
deal.II `Vector`.
"""

import RBInvParam.problems.shared.material_model

from . import hyperelasticity_model as _mod
globals().update(vars(_mod))
__all__ = getattr(_mod, "__all__", dir(_mod))
