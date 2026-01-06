import logging

from typing import Dict, Tuple, List

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.core.base import BasicObject

from RBInvParam.utils.logger import get_default_logger
from RBInvParam.model import InstationaryModelIP

class SnapshotPreprocessor(BasicObject):

    def __init__(self,
                 logger: logging.Logger = None,
                 FOM : InstationaryModelIP = None) -> None:
        
        logging.basicConfig()

        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")
    
        self.FOM = FOM
        if self.FOM is not None:
            self.krylov_directions = self.FOM.Q.empty()
            self.krylov_sensitivites = self.FOM.V.empty()
    
    def _HaPOD(self,
               snapshots: VectorArray,
               product: Operator,
               config: Dict) -> Tuple[VectorArray, List[float], int]:
    
        assert isinstance(snapshots, VectorArray) 
        assert product.source == product.range == snapshots.space

        return \
        inc_vectorarray_hapod(steps=len(snapshots) / 2, 
                              U=snapshots, 
                              eps=config['eps'],
                              omega=config['omega'],  
                              product=product)

        

        # from pymor.algorithms.pod import pod
        # snapshots, svals = pod(
        #     snapshots,
        #     product=product,
        #     l2_err = config['eps'],
        #     atol=1e-16,
        #     rtol=1e-16,
        #     orth_tol = 1e-16
        # )

        return snapshots, svals, None         

    def _compute_krylov(self,
                        config: Dict,
                        q : VectorArray,
                        u : VectorArray,
                        use_cached_operators: bool,
                        nabla_J: VectorArray = None) -> None: 
                        

        n = config['n']
        self.logger.debug(f'Include krylov directions, with n = {n}') 

        if config['inital_direction'] == 'gradient':
            krylov_direction = nabla_J
        elif config['inital_direction'] == 'ones':
            krylov_direction = self.FOM.Q.ones()
        else:
            raise ValueError

        self.krylov_directions.append(krylov_direction)

        for j in range(n):
            lin_u = self.FOM.solve_linearized_state(q=q, 
                                                    d=krylov_direction,
                                                    u=u, 
                                                    use_cached_operators=use_cached_operators)
            self.krylov_sensitivites.append(lin_u)
            z = self.FOM.solve_second_adjoint(q=q, 
                                                lin_u=lin_u, 
                                                use_cached_operators=use_cached_operators)



            krylov_direction = self.FOM.gauss_newton_hessian(u = u, 
                                                                z = z, 
                                                                q = q,
                                                                use_cached_operators=use_cached_operators)
            
            self.krylov_directions.append(krylov_direction)
            

        lin_u = self.FOM.solve_linearized_state(q=q, 
                                                d=krylov_direction,
                                                u=u, 
                                                use_cached_operators=use_cached_operators)
        self.krylov_sensitivites.append(lin_u)

    def _additional_parameter_snapshots(self,
                                        config: Dict,
                                        q : VectorArray = None,
                                        u : VectorArray = None,
                                        nabla_J : VectorArray = None,
                                        nabla_lin_J: VectorArray = None,
                                        time_steps_nabla_J: VectorArray = None,
                                        time_steps_nabla_lin_J: VectorArray = None,
                                        use_cached_operators: bool = None) -> VectorArray:

        assert self.FOM is not None
        parameter_snapshots = self.FOM.Q.empty()
        
        if config['include_each_nabla_J_time_step'] and not self.FOM.q_time_dep:
            assert time_steps_nabla_J is not None
            self.logger.debug('Include gradients for each time step as snapshots')
            parameter_snapshots.append(time_steps_nabla_J)
        
        if config['include_each_nabla_lin_J_time_step'] and not self.FOM.q_time_dep:
            assert time_steps_nabla_lin_J is not None
            self.logger.debug('Include linearized gradients for each time step as snapshots')
            parameter_snapshots.append(time_steps_nabla_lin_J)
        
        if config['include_lin_grad']:
            assert nabla_lin_J is not None
            self.logger.debug('Include nabla_lin_J')
            parameter_snapshots.append(nabla_lin_J)
        
        if config['include_krylov_directions']:
            assert q is not None
            assert u is not None
            assert use_cached_operators is not None
            assert nabla_J is not None

            self._compute_krylov(
                config = config['include_krylov_directions'],
                q = q,
                u = u,
                use_cached_operators = use_cached_operators,
                nabla_J = nabla_J,
            )
            parameter_snapshots.append(self.krylov_directions)
        
        return parameter_snapshots

    def _additional_state_snapshots(self,
                                    config: Dict,
                                    lin_u : VectorArray = None,
                                    lin_p : VectorArray = None) -> VectorArray:

        assert self.FOM is not None
        state_snapshots = self.FOM.V.empty()
        
        if config['include_lins']:
            assert lin_u is not None
            assert lin_p is not None

            self.logger.debug('Include lins')
            state_snapshots.append(lin_u)       
            state_snapshots.append(lin_p)
        
        if config['include_krylov_sensitivites']:
            self.logger.debug('Include krylov sensitivites') 
            state_snapshots.append(self.krylov_sensitivites)
        
        return state_snapshots

    def additional_snapshots(self,
                             config: Dict,
                             bases: list = None,
                             q : VectorArray = None,
                             u : VectorArray = None,
                             lin_u : VectorArray = None,
                             lin_p : VectorArray = None,
                             nabla_J : VectorArray = None,
                             nabla_lin_J : VectorArray = None,
                             time_steps_nabla_J : VectorArray = None,
                             time_steps_nabla_lin_J : VectorArray = None,
                             use_cached_operators : bool = None) -> Tuple[VectorArray,VectorArray, VectorArray]:
        
        assert self.FOM is not None

        parameter_snapshots = self.FOM.Q.empty()
        state_snapshots = self.FOM.V.empty()
        adjoint_snapshots = self.FOM.V.empty()

        for basis in bases:
            if basis == 'parameter_basis':
                parameter_snapshots = self._additional_parameter_snapshots(
                    config = config['parameter_basis']['additional_snapshots'],
                    q = q,
                    u = u,
                    nabla_J = nabla_J,
                    nabla_lin_J = nabla_lin_J,
                    time_steps_nabla_J = time_steps_nabla_J,
                    time_steps_nabla_lin_J = time_steps_nabla_lin_J,
                    use_cached_operators = use_cached_operators,
                )
            elif basis == 'state_basis':
                state_snapshots = self._additional_state_snapshots(
                    config = config['state_basis']['additional_snapshots'],
                    lin_u = lin_u,
                    lin_p = lin_p
                )
            elif basis == 'adjoint_basis':
                adjoint_snapshots = self.FOM.V.empty()
            else:
                raise ValueError
            
        #adjoint_snapshots = state_snapshots
        
        return (parameter_snapshots, state_snapshots, adjoint_snapshots)


    def preprocess(self,
                   snapshots: VectorArray,
                   product: Operator,
                   config: Dict) -> VectorArray:
        
        assert isinstance(snapshots, VectorArray) 

        self._logger.debug("Starting snapshot preprocesssing")
                    
        if config['normalize']:
            self._logger.debug(f"    Applying 'normalize'")
            norms = snapshots.norm(product)
            norms[norms <= 1e-16] = 1
            snapshots.scal(1/norms)

        if config['HaPOD']:
            self._logger.debug(f"    Applying 'HaPOD' with eps = {config['HaPOD']['eps']} and omega = {config['HaPOD']['omega']}")
            snapshots, svals, snap_count = self._HaPOD(
                snapshots = snapshots,
                product = product,
                config = config['HaPOD']
            )

            self._logger.debug(f"    HaPOD returned {len(snapshots)} modes from {snap_count}, with singular values = {svals}")

        return snapshots



