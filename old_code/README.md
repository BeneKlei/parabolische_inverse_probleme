# Adaptive Trust Region Reduced Basis Methods for Inverse Parameter Identification Problems

```
# ~~~
# This file is part of the paper:
#   
#           "Adaptive Trust Region Reduced Basis for Inverse Parameter Identification Problems"
#
#
# Copyright 2023 all developers. All rights reserved.
# License: Licensed as BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)
# Authors:
#   Michael Kartmann, Tim Keil
# ~~~
```

In this repository, we provide the code for the numerical experiments of the paper "Adaptive Trust Region Reduced Basis Methods for Inverse Parameter Identification Problems". The implementation is based on Python and includes both the reduction of the infinite-dimensional parameter space and the reduction of the state space in an error-aware trust region framework.

## Organization of the repository

The repository contains the directories [`Code/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Code) and [`Setup/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Setup). If you want to run the code on your machine, you can use `requirements.txt`  or `environment.yml`  in the directory [`Setup/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Setup) to set up your Python environment (via pip or conda), in which you can run the code. The directory [`Code/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Code) contains the source code written in the framework of the package [pyMor](https://github.com/pymor/pymor). The code consists of the main files

* `Main_Reaction.py`: the main file for the reconstruction of the reaction coefficient in Section 4.2,
* `Main_Diffusion.py`: the main file for the reconstruction of the diffusion coefficient in Section 4.3,

the files realizing the modeling of the problem in a [pyMORish way](https://docs.pymor.org/2023-1-0/technical_overview.html)

* `problems.py`: contains the implementation of the analytical problems,
* `discretizer.py`: discretizes the analytical problem to obtain a full-order model (FOM),
* `model.py`: contains the implementation of the full-order or reduced-order model (ROM),
* `reductor.py`: reduces the full-order model to obtain a reduced-order model,

and the files containing the code for the variants of the iteratively regularized gauß newton methods (IRGNM)

* `IRGNM.py:` contains the implementation of the FOM IRGNM (see Section 2.2) and the parameter space reduced Qr IRGNM (see Section 3.1),
* `Qr_Vr_TR_IRGNM.py:` contains the implementation of the parameter and state space reduced Qr-Vr TR IRGNM (see Section 3.3).

Further, there is the file  `helpers.py` for everything else and the empty directories [`Code/Diffusion_Plots/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Code/Diffusion_Plots) and [`Code/Reaction_Plots/`](https://github.com/michikartmann/adaptive_tr_rb_methods_for_inverse_problems/tree/main/Code/Reaction_Plots), where the data is saved after calling the corresponding main files.

## Questions

If there are questions of any kind, don't hesitate to get in touch with us at <michael.kartmann@uni-konstanz.de> or <tim.keil@wwu.de>.
