import logging

from typing import Dict, Tuple, List

from pymor.operators.interface import Operator
from pymor.vectorarrays.interface import VectorArray
from pymor.algorithms.hapod import inc_vectorarray_hapod
from pymor.core.base import BasicObject

from RBInvParam.utils.logger import get_default_logger

class SnapshotPreprocessor(BasicObject):

    def __init__(self,
                 logger: logging.Logger = None) -> None:
        
        logging.basicConfig()

        if logger:
            self._logger = logger
        else:
            self._logger = get_default_logger(self.__class__.__name__)
            self._logger.setLevel(logging.DEBUG)
        self.logger.debug(f"Setting up {self.__class__.__name__}")
    
    def _HaPOD(self,
               snapshots: VectorArray,
               product: Operator,
               HaPOD_tol: float = 1e-16) -> Tuple[VectorArray, List[float], int]:
    
        assert isinstance(snapshots, VectorArray) 
        assert HaPOD_tol > 0
        assert product.source == product.range == snapshots.space

        return \
        inc_vectorarray_hapod(steps=len(snapshots)/2, 
                              U=snapshots, 
                              eps=HaPOD_tol,
                              omega=0.1,                
                              product=product)

    def preprocess(self,
                   snapshots: VectorArray,
                   product: Operator,
                   config: Dict) -> VectorArray:
        
        assert isinstance(snapshots, VectorArray) 

        self._logger.debug("Starting snapshot preprocesssing")

        if config['sample_every_n_th']:
            self._logger.debug(f"    Apply 'sample_every_n_th' with n = {config['sample_every_n_th']}")
            snapshots = snapshots[::config['sample_every_n_th']]
                    
        if config['normalize']:
            self._logger.debug(f"    Apply 'normalize'")
            norms = snapshots.norm(product)
            norms[norms <= 1e-16] = 1
            snapshots.scal(1/norms)

        if config['HaPOD']:
            self._logger.debug(f"    Apply 'HaPOD'")

            snapshots, svals, snap_count  = self._HaPOD(
                snapshots = snapshots,
                product = product,
                **config['HaPOD']
            )

            self._logger.debug(f"    HaPOD returned {len(snapshots)} modes from {snap_count}, with singular values = {svals}")

        return snapshots



