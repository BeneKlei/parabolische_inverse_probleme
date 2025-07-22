import numpy as np

#%% grid function

def hatfunction(start, peak, end):
    """get a hat function

    1             /\
    |            /  \
    |           /    \
    |          /      \
    0----------        -------------
    |
               |   |   |
           start peak  end

    :param start: value where function starts to rise, may be -float('inf')
    :param peak: value where functions has its maximum
    :param end: value where functions ends to fall, may be float('inf')

    :returns: python function
    """
    start = float(start)
    peak = float(peak)
    end = float(end)
    assert start < peak < end
    assert np.isfinite(peak)

    def f(x):
        x = np.array(x)
        return np.maximum(0,
                   np.minimum(
                       (x - peak) / (peak - start) + 1,
                       (x - peak) / (peak - end) + 1
                   )
                )
    return f


def ndhatfunction(definitions):
    """get an n dimensional hat function

    :param definitions: list of triples (start,peak,end)

    :returns: python function
    """
    onedfunctions = [hatfunction(*d) for d in definitions]

    def ndf(x):
        x = np.array(x)
        result = np.ones(x.shape[:-1])
        for i, f in enumerate(onedfunctions):
            result *= f(x[..., i])
        return result
    return ndf

def twodhatfunction(domains):
    """get an 2 dimensional hat function

    :param definitions: list of triples (start,peak,end)

    :returns: python function
    """
    onedfunctions = [hatfunction(*domain) for domain in domains]

    def twodf(x):
        x = np.array(x)
        result = np.ones(x.shape[:-1])
        for i, f in enumerate(onedfunctions):
            result *= f(x[..., i])
        return result
    return twodf

def gen_definitions(boundaries, outer_constant):
    """generates hat function definitions from domain boundaries

    examples:

    (0,1,2,3,4) becomes with outer_constant=False
    [(-float('inf'),0,1),
     (0,1,2),
     (1,2,3),
     (3,4,float('inf'))]

    (0,1,2,3,4) becomes with outer_constant=True
    [(-float('inf'),1,2),
     (2,3,float('inf'))]


    """
    boundaries = list(boundaries)  # create a copy
    if outer_constant:
        boundaries = boundaries[1:-1]

    boundaries = [-float('inf')] + boundaries + [float('inf')]

    if len(boundaries) == 2:
        # we need a constant one function
        return [(-float('inf'), 0, float('inf'))]

    return [(boundaries[i], boundaries[i + 1], boundaries[i + 2])
            for i in range(len(boundaries) - 2)]

#%% partition of unity

# Note:
# The code for the partition of unity was taken from Andreas Buhr and can be 
# found at https://doi.org/10.5281/zenodo.2555624 .

def partition_of_unity(boundary_lists, outer_constant=True):
    """get a partition of unity

    :param boundary_lists: a list of lists. For each dimension, a
                           list of domain boundaries
    :param outer_constant: whether the outmost function should be
                           one at the boundary and then go to zero within
                           the first domain or
                           constant one in the first domain and go to zero
                           within the second domain
    :returns: numpy array of python functions
    """

    definition_lists = [gen_definitions(b, outer_constant)
                        for b in boundary_lists]
    resultshape = [len(d) for d in definition_lists]
    alldefs = [[definition] for definition in definition_lists[0]]
    # outer product
    for deflist in definition_lists[1:]:
        alldefs = [oldlist + [newelem]
                   for oldlist in alldefs for newelem in deflist]

    allfuns = map(ndhatfunction, alldefs)
    result = np.array(list(allfuns))
    result = result.reshape(resultshape)
    return result

def localized_pou(coarse_grid_resolution):
    boundaries_x = np.linspace(0., 1., coarse_grid_resolution[0]+1)
    boundaries_y = np.linspace(0., 1., coarse_grid_resolution[1]+1)
    pou = partition_of_unity((boundaries_x, boundaries_y), False)
    return pou

#%% H1 thermal block
from itertools import product
from pymor.analyticalproblems.domaindescriptions import RectDomain
from pymor.analyticalproblems.elliptic import StationaryProblem
from pymor.analyticalproblems.functions import ConstantFunction, LincombFunction, GenericFunction
from pymor.parameters.functionals import ProjectionParameterFunctional

def thermal_block_problem_h1(num_blocks=(3, 3), parameter_range=(0.1, 1)):
    """Analytical description of a 2D 'thermal block' diffusion problem.

    The problem is to solve the elliptic equation ::

      - ∇ ⋅ [ d(x, μ) ∇ u(x, μ) ] = 1

    on the domain [0,1]^2 with Dirichlet zero boundary values. The domain is
    partitioned into nx x ny blocks and the diffusion function d(x, μ) has
    value μ_i on node i of of the blocks::

                   μ_8      μ_9
      μ_7  *--------*--------*
           |        |        |
           |        |        |
           |        |        |
      μ_4  *--------*--------* μ_6
           |    μ_5 |        |
           |        |        |
           |        |        |
      μ_1  *--------*--------*
                   μ_2      μ_3

    Parameters
    ----------
    num_blocks
        The tuple `(nx, ny)`
    parameter_range
        A tuple `(μ_min, μ_max)`. Each |Parameter| component μ_i is allowed
        to lie in the interval [μ_min, μ_max].
    """

    def parameter_functional_factory(ix, iy):
        return ProjectionParameterFunctional('diffusion',
                                             size=(num_blocks[0]+1)*(num_blocks[1]+1),
                                             index=ix + iy*(num_blocks[0]+1),
                                             name=f'diffusion_{ix}_{iy}')

    pou = localized_pou(num_blocks)

    def diffusion_function_factory(ix, iy):
        pou_func = GenericFunction(pou[ix, iy], dim_domain=2)
        return pou_func

    return StationaryProblem(
        domain=RectDomain(),
        rhs=ConstantFunction(dim_domain=2, value=1.),
        diffusion=LincombFunction([diffusion_function_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]+1), range(num_blocks[0]+1))],
                                  [parameter_functional_factory(ix, iy)
                                   for iy, ix in product(range(num_blocks[1]+1), range(num_blocks[0]+1))],
                                  name='diffusion'),
        parameter_ranges=parameter_range,
        name=f'ThermalBlock({num_blocks})'
    )


# TODO Move to test
# if __name__ == "__main__":
#     # most simple test
#     myfun = hatfunction(3, 4, 5)
#     assert myfun(2) == 0
#     assert myfun(3) == 0
#     assert myfun(3.5) == 0.5
#     assert myfun(4) == 1
#     assert myfun(4.5) == 0.5
#     assert myfun(5) == 0
#     assert myfun(6) == 0
#     # vectorized test
#     assert np.all(myfun((3.5, 4, 4.5)) == (0.5, 1, 0.5))

#     # 'inf' test
#     myfun = hatfunction(3, 4, float('inf'))
#     assert myfun(2) == 0
#     assert myfun(3) == 0
#     assert myfun(3.5) == 0.5
#     assert myfun(4) == 1
#     assert myfun(4.5) == 1
#     assert myfun(5) == 1
#     assert myfun(6) == 1

#     myfun = hatfunction(-float('inf'), 4, 6)
#     assert myfun(2) == 1
#     assert myfun(3) == 1
#     assert myfun(3.5) == 1
#     assert myfun(4) == 1
#     assert myfun(4.5) == 0.75
#     assert myfun(5) == 0.5
#     assert myfun(6) == 0

#     # n dimensional test
#     myfun = ndhatfunction(((3, 4, 5), (4, 5, float('inf'))))
#     assert myfun((3, 5)) == 0
#     assert myfun((3.5, 5)) == 0.5
#     assert myfun((4, 4.5)) == 0.5
#     # vectorized test
#     assert np.all(myfun( ( (3, 5), (3.5, 5), (4, 4.5) ) ) == (0, 0.5, 0.5) )

#     # gen_definitions test
#     # 3 domains, constant
#     refresult = [(-float('inf'), 1, 2), (1, 2, float('inf'))]
#     actualresult = gen_definitions((0, 1, 2, 3), outer_constant=True)
#     assert refresult == actualresult

#     # 3 domains, not constant
#     refresult = [(-float('inf'), 0, 1),
#                  (0, 1, 2),
#                  (1, 2, 3),
#                  (2, 3, float('inf'))]
#     actualresult = gen_definitions((0, 1, 2, 3), outer_constant=False)
#     assert refresult == actualresult

#     # 2 domains, constant
#     refresult = [(-float('inf'), 1, float('inf'))]
#     actualresult = gen_definitions((0, 1, 2), outer_constant=True)
#     assert refresult == actualresult

#     # 2 domains, not constant
#     refresult = [(-float('inf'), 0, 1),
#                  (0, 1, 2),
#                  (1, 2, float('inf'))]
#     actualresult = gen_definitions((0, 1, 2), outer_constant=False)
#     assert refresult == actualresult

#     # 1 domain, constant
#     refresult = [(-float('inf'), 0, float('inf'))]
#     actualresult = gen_definitions((0, 1), outer_constant=True)
#     assert refresult == actualresult

#     # 1 domain, not constant
#     refresult = [(-float('inf'), 0, 1),
#                  (0, 1, float('inf'))]
#     actualresult = gen_definitions((0, 1), outer_constant=False)
#     assert refresult == actualresult

#     # partition of unity tests
#     pou = partition_of_unity((
#         (1, 2, 3),
#     ), outer_constant=False)
#     assert pou.shape == (3,)

#     pou = partition_of_unity((
#         (1, 2, 3),
#         (4, 5, 6),
#     ), outer_constant=False)
#     assert pou.shape == (3, 3)
#     myfun = pou[1, 1]
#     assert myfun((2, 5)) == 1
#     assert myfun((2, 5.5)) == 0.5
#     assert myfun((2.5, 5.5)) == 0.25
#     myfun = pou[1, 2]
#     assert myfun((2, 5)) == 0
#     assert myfun((2, 6)) == 1

#     pou = partition_of_unity((
#         (1, 2, 3),
#         (4, 5, 6),
#     ), outer_constant=True)
#     assert pou.shape == (1, 1)
#     myfun = pou[0, 0]
#     assert myfun((3, 4)) == 1

#     pou = partition_of_unity((
#         (1, 2),
#         (4, 5),
#     ), outer_constant=True)
#     assert pou.shape == (1, 1)
#     myfun = pou[0, 0]
#     assert myfun((3, 4)) == 1

#     pou = partition_of_unity((
#         (1, 2, 3, 5),
#         (4, 5, 6, 7),
#     ), outer_constant=False)
#     assert pou.shape == (4, 4)
#     myfun = pou[1, 1]
#     assert myfun((2, 5)) == 1