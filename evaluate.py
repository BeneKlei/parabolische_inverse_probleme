import argparse
import numpy as np
from typing import Union
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "cm",
    "font.size": 10,
    'text.latex.preamble': r'\usepackage{amsfonts}',
    'figure.dpi': 400
})

from pymor.basic import *
from pymor.core.pickle import load

from model import InstationaryModelIP
from problems import whole_problem
from discretizer import discretize_instationary_IP

set_log_levels({
    'pymor' : 'WARN'
})

parser = argparse.ArgumentParser()
parser.add_argument("data_path", help="Path to dumped optimization statistics.")
args = parser.parse_args()

def evaluate_optmizer_run(path: Union[str, Path]) -> None:
    path = Path(path)
    assert path.suffix in ['.pkl', 'pickle']

    with open(path, 'rb') as file:
        data = load(file)

    if 1:
        N = data['dims']['N']
        dims = data['dims']
        model_parameter = data['model_parameter']

        analytical_problem, _, _, problem_type, _, _ = whole_problem(N = N,
                                                                     parameter_location = 'reaction',
                                                                     boundary_conditions = 'dirichlet',
                                                                     exact_parameter = 'Kirchner')

        building_blocks = discretize_instationary_IP(analytical_problem,
                                                     model_parameter,
                                                     dims, 
                                                     problem_type) 

        FOM = InstationaryModelIP(                 
            *building_blocks,
            dims = dims,
            model_parameter = model_parameter
        )

        q_estim = data['optimizer_statistics']['q'][-1]
        q_exact = FOM.Q.make_array(model_parameter['q_exact'])
        abs_delta_q = FOM.Q.make_array(np.abs(q_estim.to_numpy() - q_exact.to_numpy()))

        FOM.visualizer.visualize(q_estim, title=r'Final parameter: q_h^I')
        FOM.visualizer.visualize(q_exact, title=r'Exact parameter: q_exact')
        FOM.visualizer.visualize(abs_delta_q, title=r'Difference: \|q_h^I - q_exact\|')

        print("Q-Norm") 
        norm_delta_q = np.sqrt(FOM.products['bochner_prod_Q'](abs_delta_q, abs_delta_q))
        norm_q_exact = np.sqrt(FOM.products['bochner_prod_Q'](q_exact, q_exact))
        print(f"Absolute error: {norm_delta_q:3.4e}")
        print(f"Relative error: {norm_delta_q / norm_q_exact * 100:3.4}%.")



    if 0:
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6,3))
        cmap = plt.get_cmap('viridis')

        Js = np.array(data['optimizer_statistics']['J'])
        norm_nabla_Js = np.array(data['optimizer_statistics']['norm_nabla_J'])
        assert len(Js) == len(norm_nabla_Js)
        
        ax[0].set_xlabel(r'$\textrm{Main loop index } i$')
        ax[0].set_ylabel(r'$J_h(q^i_h)$')
        ax[0].set_yscale('log')
        ax[0].grid()
        ax[0].plot(Js, marker='x',color=cmap(0))

        ax[1].set_xlabel(r'$\textrm{Main loop index } i$')
        ax[1].set_ylabel(r'$\|\nabla_q J_h(q^i_h)\|_{\mathbb{R}^{KN_h}}$')
        ax[1].set_yscale('log')
        ax[1].grid()
        ax[1].plot(norm_nabla_Js, marker='x',color=cmap(100))
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    evaluate_optmizer_run(args.data_path)
