import logging
import itertools
import numpy as np

from pymor.vectorarrays.interface import VectorSpace, VectorArray
from pymor.discretizers.builtin.cg import L2ProductQ1
from pymor.discretizers.builtin.grids.boundaryinfos import EmptyBoundaryInfo
from RBInvParam.model import InstationaryModelIP
from RBInvParam.reductor import InstationaryModelIPReductor

class ProximalOperator():
    id_iter = itertools.count()

    def __init__(self, 
                 source: VectorSpace,
                 range: VectorSpace,
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

        self.source = source
        self.range = range
    
    def apply(self, 
              center: VectorArray,
              direction: VectorArray,
              step_size: float) -> VectorArray:

        raise NotImplementedError
    
    def pre_compute(self)-> None:
        raise NotImplementedError


class SimpleBoundDomainL1ProximalOperator(ProximalOperator):
    def __init__(self,
                 model: InstationaryModelIP,   
                 bounds: np.ndarray,
                 reductor: InstationaryModelIPReductor = None,
                 use_sufficient_condition: bool = True,
                 logger: logging.Logger = None):
     
        super().__init__(
            logger = logger,
            source = model.Q,
            range = model.Q,
        )

        self.model = model
        self.bounds = bounds
        self.reductor = reductor
        self.use_sufficient_condition = use_sufficient_condition

        assert isinstance(self.model, InstationaryModelIP)
        assert isinstance(self.bounds, np.ndarray)

        if self.reductor is not None:
            assert isinstance(self.reductor, InstationaryModelIPReductor)

        assert isinstance(self.use_sufficient_condition, bool)

        assert self.model.Q == self.source
        assert self.model.Q == self.range

        if self.use_sufficient_condition:
            self.logger.debug('Using sufficient condition.')
            assert self.reductor is not None
        else:
            self.logger.debug('NOT using sufficient condition.')

        if self.reductor is not None:
            self.FOM_dim = self.reductor.FOM.Q.dim
            self.q_circ = self.reductor.FOM.setup['model_parameter']['q_circ'].flatten()
            grid = self.reductor.FOM.A.grid
        else:
            self.FOM_dim = self.model.Q.dim
            self.q_circ = self.model.setup['model_parameter']['q_circ'].flatten()
            grid = self.model.A.grid
            
        W = L2ProductQ1(grid, 
                        boundary_info = EmptyBoundaryInfo(grid))
                        
        self.W = W.apply(W.source.ones())
                        
        if self.model.q_time_dep:
            assert self.bounds.shape == (self.model.nt * self.FOM_dim , 2)
        else:
            assert self.bounds.shape == (self.FOM_dim , 2)
        assert np.all(self.bounds[:,0] < self.bounds[:,1])

        
    
    def pre_compute(self,
                    center: VectorArray)-> None:

        if not self.use_sufficient_condition:
            return

    def apply(self, 
              center: VectorArray,
              direction: VectorArray,
              step_size: float,
              alpha: float = 0) -> VectorArray:
        
        assert alpha >= 0
        if self.use_sufficient_condition:
            raise NotImplementedError

        if self.reductor:
            center_recon = self.reductor.reconstruct(center, basis='parameter_basis')
            direction_recon = self.reductor.reconstruct(direction, basis='parameter_basis')
        else:
            center_recon = center
            direction_recon = direction

        center_recon = center_recon.to_numpy().flatten()
        direction_recon = direction_recon.to_numpy().flatten()

        C_2 = step_size * alpha * self.W.to_numpy()
        C_2 = C_2[0]
        if self.model.q_time_dep:
           C_2 = np.stack([C_2] * self.reductor.FOM.nt, axis=0).flatten()
           
      
        C_1 = center_recon - self.q_circ
        buf = np.abs(direction_recon - C_1) - C_2
        buf = np.where(buf >= 0, buf, 0)
        direction_recon = C_1 + np.sign(direction_recon - C_1) * buf
                         
        update_recon = center_recon + direction_recon

        mask_lb = update_recon < self.bounds[:,0]
        mask_ub = update_recon > self.bounds[:,1]

        if np.any(mask_lb) or np.any(mask_ub):
            update_recon[mask_lb] = self.bounds[mask_lb,0]
            update_recon[mask_ub] = self.bounds[mask_ub,1]
            

        if self.model.setup['model_parameter']['q_time_dep']:  
            update_recon = update_recon.reshape((self.reductor.FOM.nt, self.FOM_dim))
        else:
            update_recon = update_recon.reshape((1, self.FOM_dim))
        
        if self.reductor: 
            update_recon = self.reductor.FOM.Q.make_array(update_recon)  
            update_recon = self.reductor.project_vectorarray(update_recon, basis='parameter_basis')
            
        return self.model.Q.make_array(update_recon)
        