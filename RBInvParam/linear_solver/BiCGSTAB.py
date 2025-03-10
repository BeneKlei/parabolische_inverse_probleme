import numpy as np
import scipy.sparse.linalg as spla

from pymor.vectorarrays.interface import VectorArray

from RBInvParam.model import InstationaryModelIP

class LinearGradientScipyOperator(spla.LinearOperator):
    def __init__(self,
                 model: InstationaryModelIP,
                 q: VectorArray,
                 alpha : float,
                 use_cached_operators: bool):

        assert isinstance(model, InstationaryModelIP)
        assert isinstance(q, VectorArray)
        assert q in model.Q
        assert alpha > 0

        self.model = model
        self.q = q
        self.alpha = alpha
        self.use_cached_operators = use_cached_operators

        self.shape = (
            self.model.Q.dim, 
            self.model.Q.dim
        )
        self.dtype = np.dtype(np.float64)

        self.num_iter = 0
    
    def set_b(self, b : np.ndarray):
        self.b = b
        self.num_iter = 0

    def _matvec(self, x):
        self.num_iter += 1
        return (self.model.compute_linearized_gradient(
            q=self.q, 
            d=self.model.Q.make_array(x), 
            alpha=self.alpha, 
            use_cached_operators=self.use_cached_operators
        ).to_numpy()[0] - self.b)