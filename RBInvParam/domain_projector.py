import numpy as np
import logging
import itertools

from pymor.vectorarrays.numpy import NumpyVectorArray

from RBInvParam.model import InstationaryModelIP
from RBInvParam.reductor import InstationaryModelIPReductor
from RBInvParam.utils.logger import get_default_logger

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
                 reductor: InstationaryModelIPReductor = None,
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
        assert isinstance(self.reductor, InstationaryModelIPReductor)
        assert isinstance(self.use_sufficient_condition, bool)

        if self.use_sufficient_condition:
            self.logger.debug('Using sufficient condition.')
            assert self.reductor is not None
        else:
            self.logger.debug('NOT using sufficient condition.')
                    
        if self.model.q_time_dep:
            assert self.bounds.shape == (self.model.nt * self.reductor.FOM.Q.dim , 2)
        else:
            assert self.bounds.shape == (self.reductor.FOM.Q.dim , 2)
        assert np.all(self.bounds[:,0] < self.bounds[:,1])

    def pre_compute(self,
                    center: NumpyVectorArray) -> None:

        self.logger.debug('Procomputing domain projector.')
        if self.use_sufficient_condition:
            center_recon = self.reductor.reconstruct(center, basis='parameter_basis')
            center_recon = center_recon.to_numpy()

            b = np.linalg.norm(
                self.reductor.bases['parameter_basis'].to_numpy(), 
                axis=0
            )
            assert np.all(b > 0)

            if self.model.setup['model_parameter']['q_time_dep']:
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
            assert np.all(self.r > 0)
            
    def project_domain(self,
                       center: NumpyVectorArray,
                       direction: NumpyVectorArray) -> NumpyVectorArray:

        if self.use_sufficient_condition:
            assert hasattr(self, 'r')
            suff_cond = np.linalg.norm(direction.to_numpy(), axis=1) <= self.r
            suff_cond = np.all(suff_cond)
            
            if suff_cond:
                return center + direction
        
        update = center + direction
        update_recon = self.reductor.reconstruct(update, basis='parameter_basis')
        update_recon = update_recon.to_numpy().flatten()
            
        mask_lb = update_recon < self.bounds[:,0]
        mask_ub = update_recon > self.bounds[:,1]
        
        if np.any(mask_lb) or np.any(mask_ub):
            update_recon[mask_lb] = self.bounds[mask_lb,0]
            update_recon[mask_ub] = self.bounds[mask_ub,1]
            
            if self.model.setup['model_parameter']['q_time_dep']:  
                update_recon = update_recon.reshape((self.reductor.FOM.nt, self.reductor.FOM.Q.dim))
            else:
                update_recon = update_recon.reshape((1, self.reductor.FOM.Q.dim))

            update_recon = self.reductor.FOM.Q.make_array(update_recon)  
            update_recon = self.reductor.project_vectorarray(update_recon, basis='parameter_basis')
            return self.model.Q.make_array(update_recon)
        else:
            return update        
