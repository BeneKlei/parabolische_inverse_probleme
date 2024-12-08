from typing import Dict

from pymor.reductors.basic import ProjectionBasedReductor
from pymor.vectorarrays.interface import VectorArray
#from pymor.operators.interface import Operator

from model import InstationaryModelIP

class InstationaryModelIPReductor(ProjectionBasedReductor):
    def __init__(self, 
                 fom: InstationaryModelIP, 
                 products: Dict, 
                 check_orthonormality: bool =False, 
                 check_tol: float =1e-3):

        bases = {
            'state_basis' : None,
            'parameter_basis' : None
        }

        _products = {
            'state_basis' : products['prod_V'],
            'parameter_basis' : products['prod_Q']
        }

        super().__init__(fom, 
                         bases, 
                         _products,
                         check_orthonormality=check_orthonormality, 
                         check_tol=check_tol)


    def project_vectorarray(self, 
                            x : VectorArray,
                            basis: str) -> VectorArray:

        _basis = self.bases[basis]
        assert isinstance(x, VectorArray)
        # assert product.source == product.range
        # assert basis in product.source
        # assert basis in product.range

        if len(_basis) == 0:
            return x
        else:
            return _basis.lincomb(x.inner(_basis, self.products[basis]))



    def project_operators(self) -> Dict:
        state_basis = self.bases['state_basis']
        parameter_basis = self.bases['parameter_basis']

        # projected_operators = {
        #     'u_0' : 
        #     'M' : 
        #     'A' :
        #     'f' :
        #     'B' :
        #     'constant_cost_term' :
        #     'linear_cost_term'
        #     'bilinear_cost_term'
        #     'Q' :
        #     'q_circ' :
        #     'constant_reg_term'
        #     'linear_reg_term'
        #     'bilinear_reg_term'
        #     'products' : 

        # }

        return projected_operators 

    def build_rom(self):
        pass

    def assemble_error_estimator(self):
        pass

    # TODO write subbasis variants for project_operator and assemble_error_estimator
    def for_subbasis(self):
        pass
