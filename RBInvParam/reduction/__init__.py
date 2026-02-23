from .registry import register_reductor, get_reductor_class, list_reductors  # optional re-export

# Import for side effects: this runs @register_reductor("default")
from .default import DefaultIPReductor  # noqa: F401