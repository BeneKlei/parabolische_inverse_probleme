import numpy as np
import scipy.sparse.linalg as spla

from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP

class LinearGradientScipyOperator(spla.LinearOperator):
    def __init__(self,
                 model: InstationaryModelIP,
                 q: VectorArray,
                 alpha : float,
                 b: VectorArray,
                 use_cached_operators: bool):

        assert isinstance(model, InstationaryModelIP)
        assert isinstance(q, VectorArray)
        assert q in model.Q
        assert alpha > 0

        self.model = model
        self.q = q
        self.alpha = alpha
        self.use_cached_operators = use_cached_operators
        self.b = b

        self.shape = (
            self.model.Q.dim, 
            self.model.Q.dim
        )
        self.dtype = np.dtype(np.float64)    

    def _matvec(self, x) -> None:
        return (self.model.compute_linearized_gradient(
            q=self.q, 
            d=self.model.Q.make_array(x), 
            alpha=self.alpha, 
            use_cached_operators=self.use_cached_operators
        ).to_numpy()[0] - self.b)

class IterationCounter:
    def __init__(self):
        self.count = 0
    def __call__(self, xk):
        self.count += 1