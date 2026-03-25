import numpy as np
import logging
import itertools

from pymor.vectorarrays.numpy import NumpyVectorArray

from RBInvParam.model import InstationaryModelIP
from RBInvParam.reduction.base import BaseIPReductor
from RBInvParam.utils.logger import get_default_logger
from pymor.algorithms.basic import almost_equal

class ProjectionMismatchError(Exception):
    """Raised when a projection changes a VectorArray unexpectedly."""

    def __init__(self, message, diff_norm=None):
        super().__init__(message)
        self.diff_norm = diff_norm

class DomainProjector():
    id_iter = itertools.count()

    def __init__(self, 
                 logger: logging.Logger = None):

        logging.basicConfig()
        if logger:
            self.logger = logger
        else:
            self.logger = get_default_logger(
                logger_name=self.__class__.__name__ + str(next(InstationaryModelIP.id_iter))
            )
            self.logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")
    
    def pre_compute(self) -> None:
        raise NotImplementedError

    def project_domain(self) -> None:
        raise NotImplementedError

class SimpleBoundDomainProjector(DomainProjector):

    def __init__(self,
                 model: InstationaryModelIP,             
                 bounds: np.ndarray,
                 reductor: BaseIPReductor = None,
                 use_sufficient_condition: bool = True,
                 logger: logging.Logger = None):
     
        super().__init__(
            logger = logger
        )

        self.model = model
        self.bounds = bounds
        self.reductor = reductor
        self.use_sufficient_condition = use_sufficient_condition

        assert isinstance(self.model, InstationaryModelIP)
        assert isinstance(self.bounds, np.ndarray)
        if self.reductor:
            assert isinstance(self.reductor, BaseIPReductor)
        assert isinstance(self.use_sufficient_condition, bool)

        if self.use_sufficient_condition:
            self.logger.debug('Using sufficient condition.')
            assert self.reductor is not None
        else:
            self.logger.debug('NOT using sufficient condition.')

        if self.reductor:
            self.FOM_Q_dim = self.reductor.FOM.Q.dim
        else:
            self.FOM_Q_dim = self.model.Q.dim
        
        if self.model.q_time_dep:
            assert self.bounds.shape == (self.model.nt * self.FOM_Q_dim, 2)
        else:
            assert self.bounds.shape == (self.FOM_Q_dim , 2)
        assert np.all(self.bounds[:,0] < self.bounds[:,1])
    
    def check_q(self, q: NumpyVectorArray) -> bool:
        if self.reductor:
            q_recon = self.reductor.reconstruct(q, basis='parameter_basis')
            q_recon = q_recon.to_numpy().flatten()
        else:
            q_recon = q.to_numpy().flatten()

        mask_lb = q_recon < self.bounds[:,0]
        mask_ub = q_recon > self.bounds[:,1]

        if np.any(mask_lb) or np.any(mask_ub):
            return False
        else:
            return True

    def pre_compute(self,
                    center: NumpyVectorArray) -> None:

        self.logger.debug('Precomputing domain projector.')
        if self.use_sufficient_condition:
            center_recon = self.reductor.reconstruct(center, basis='parameter_basis')
            center_recon = center_recon.to_numpy().flatten()

            mask_lb = center_recon < self.bounds[:,0]
            mask_ub = center_recon > self.bounds[:,1]

            if np.any(mask_lb) or np.any(mask_ub):
                center_recon[mask_lb] = self.bounds[mask_lb,0]
                center_recon[mask_ub] = self.bounds[mask_ub,1]

            if self.model.q_time_dep:  
                center_recon = center_recon.reshape((self.model.nt, self.FOM_Q_dim))
            else:
                center_recon = center_recon.reshape((1, self.FOM_Q_dim))   
                 

            b = np.linalg.norm(
                self.reductor.bases['parameter_basis'].to_numpy(), 
                axis=0
            )
            assert np.all(b > 0)

            if self.model.q_time_dep:
                dim_Q_h = self.reductor.FOM.Q.dim
                self.r = np.zeros(self.model.nt)

                for i in range(self.model.nt):
                    _bounds = self.bounds[(i * dim_Q_h):((i+1) * dim_Q_h),:]
                    l = np.min((_bounds[:,1] - center_recon[i]) * (1 / b), axis = 0)
                    u = np.min((center_recon[i] - _bounds[:,0]) * (1 / b), axis = 0)
                    self.r[i] = np.min(np.stack([l,u]), axis=0)
            else:
                self.r = np.zeros(1)
                _bounds = self.bounds
                l = np.min((_bounds[:,1] - center_recon[0]) * (1 / b), axis = 0)
                u = np.min((center_recon[0] - _bounds[:,0]) * (1 / b), axis = 0)
                self.r[0] = np.min(np.stack([l,u]), axis=0)
            
            assert np.all(self.r + 1e-16 > 0)
            
    def project_domain(self,
                       center: NumpyVectorArray,
                       direction: NumpyVectorArray = None) -> NumpyVectorArray:

        if direction:
            update = center + direction
        else:
            update = center

        if self.use_sufficient_condition and direction:
            assert hasattr(self, 'r')
            suff_cond = np.linalg.norm(direction.to_numpy(), axis=1) <= self.r
            suff_cond = np.all(suff_cond)
            
            if suff_cond:
                return update
        
        if self.reductor:
            update_recon = self.reductor.reconstruct(update, basis='parameter_basis')
            update_recon = update_recon.to_numpy().flatten()
        else:
            update_recon = update.to_numpy().flatten()
        
        mask_lb = update_recon < self.bounds[:,0]
        mask_ub = update_recon > self.bounds[:,1]

        if np.any(mask_lb) or np.any(mask_ub):        
            update_recon[mask_lb] = self.bounds[mask_lb,0]
            update_recon[mask_ub] = self.bounds[mask_ub,1]
        else:
            return update    
        
        if self.model.q_time_dep:  
            update_recon = update_recon.reshape((self.model.nt, self.FOM_Q_dim))
        else:
            update_recon = update_recon.reshape((1, self.FOM_Q_dim))

        if self.reductor:
            update_recon = self.reductor.FOM.Q.make_array(update_recon)
            projected = self.reductor.project_vectorarray(
                update_recon, basis='parameter_basis'
            )
            projected = self.model.Q.make_array(projected)

            if not almost_equal(update, projected, rtol=1e-12, atol=1e-14).all():
                diff = (update - projected).norm()[0]
                raise ProjectionMismatchError(
                    f"Projection changed the vector. Norm difference(s): {diff:3.4e}",
                    diff_norm=diff
                )

            return update_recon
        else:
            return self.model.Q.make_array(update_recon)  
        
        
        
