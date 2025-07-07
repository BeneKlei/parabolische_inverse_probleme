# Parabolische Inverse Probleme

This repository contains the code for the numerical experiments presented in the paper *"Adaptive Reduced Basis Trust Region Methods for Parabolic Inverse Problems"* by Michael Kartmann, Benedikt Klein, Mario Ohlberger, Thomas Schuster, and Stefan Volkwein. The implementation builds upon [previous work](https://zenodo.org/records/8328835) by Michael Kartmann and Tim Keil.

The experiments in the paper were conducted on the PALMA II HPC cluster at the University of Münster, funded by the DFG (INST 211/667-1), using compute nodes with Intel Skylake Gold 6140 CPUs @ 2.30 GHz and 92 GB RAM. You can rerun these experiments on your local Linux machine or use the code to explore and experiment further.


## Setup

Ensure you have Git and Python 3.9 installed on your Linux system (later versions may also work). Then, clone the repository:

```bash
git clone https://github.com/BeneKlei/parabolische_inverse_probleme.git
```

Navigate to the project directory and create a virtual environment:

```bash
cd parabolische_inverse_probleme
python -m venv ./venv
source ./venv/bin/activate
```

Install the dependencies and install the project as a module:

```bash
pip install -r requirements.txt
pip install -e .
```

## Running the Experiments

Tested on **Ubuntu 24.04 LTS** with **Intel® Core™ i7-7700** and **32 GiB RAM**

TODO: Test all configs

To solve a single inverse problem, first follow the setup steps. Then navigate to either `./examples/reaction` or `./examples/diffusion`, and run `FOM_IRGNM.py` or `TR_IRGNM.py` using your local Python environment.

You can modify the behavior of the optimization algorithms by editing the values in the `setup` and `optimizer_parameter` dictionaries.

To run the full set of experiments from the paper as a batch:

```bash
cd /path/to/parabolische_inverse_probleme/experiments/paper_POD_tol_compare
python ../../RBInvParam/deployment/run_experiment_batch.py ./paper_POD_tol_compare.py local
```

⚠️ **Warning:** Running the full experiment set may take several days to complete.



