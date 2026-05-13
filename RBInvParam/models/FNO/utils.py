import json
import torch
import numpy as np
from pathlib import Path

from RBInvParam.utils.create_q_exact import *

from torch.utils.data import TensorDataset, DataLoader, random_split


def generate_random_q(
    n_samples,
    param_y_res,
    param_z_res,
    y_bounds,
    z_bounds,
    nt,
    base_value=1.0,
    value_range=(1.5, 4.0),
    n_patches_range=(1, 3),
    half_size_range=(0.5, 2.0),
    seed=0,
):
    rng = np.random.default_rng(seed)
    par_dim = (param_y_res + 1) * (param_z_res + 1)

    Q = np.zeros((n_samples, nt + 1, par_dim), dtype=np.float32)

    for i in range(n_samples):
        q = base_value * np.ones((param_y_res + 1, param_z_res + 1), dtype=np.float32)

        n_patches = rng.integers(n_patches_range[0], n_patches_range[1] + 1)

        for _ in range(n_patches):
            center_coords = (
                rng.integers(y_bounds[0], y_bounds[1]),
                rng.integers(z_bounds[0], z_bounds[1]),
            )
            value = rng.uniform(value_range[0], value_range[1])
            #half_size = rng.uniform(half_size_range[0], half_size_range[1])
            half_size = 1
            
            add_constant_square_patch_from_center_coords(
                q,
                center_coords=center_coords,
                value=value,
                half_size=half_size,
                interpolated=True,
                distance="square",
                y_bounds=y_bounds,
                z_bounds=z_bounds,
            )
        Q[i] = np.tile(q.flatten(), (nt + 1, 1))

    return Q


def generate_training_data(
    FOM,
    n_samples,
    param_y_res,
    param_z_res,
    y_bounds,
    z_bounds,
    nt,
    save_path,
    seed=0,
    X = None
):
    save_path = Path(save_path)
    data_path = save_path / "training"
    data_path.mkdir(parents=True, exist_ok=True)

    #logger.info(f"Generating {n_samples} training samples")
    print(f"Generating {n_samples} training samples")

    if X is None:
        X = generate_random_q(
            n_samples=n_samples,
            param_y_res=param_y_res,
            param_z_res=param_z_res,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            nt=nt,
            seed=seed,
        )

    Y = []

    for i in range(X.shape[0]):
        #logger.info(f"Solving sample {i + 1}/{n_samples}")
        print(f"Solving sample {i + 1}/{n_samples}")

        q_sample = FOM.Q.make_array(
            np.array([X[i,0,:]])
        )

        u = FOM.solve_state(
            q=q_sample,
            use_cached_operators=True,
            return_higher_orders=False,
        )

        # Convert pyMOR vector array to numpy.
        # Depending on your FOM, either `to_numpy()` or `.data` may be needed.
        if hasattr(u, "to_numpy"):
            u_np = u.to_numpy()
        else:
            u_np = np.asarray(u.data)

        Y.append(u_np.astype(np.float32))

    Y = np.asarray(Y, dtype=np.float32)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, Y_tensor)

    torch.save(
        {
            "X": X_tensor,
            "Y": Y_tensor,
            "dataset": dataset,
        },
        data_path / "fno_training_data.pt",
    )

    np.save(data_path / "X.npy", X)
    np.save(data_path / "Y.npy", Y)

    metadata = {
        "n_samples": n_samples,
        "param_y_res": param_y_res,
        "param_z_res": param_z_res,
        "par_dim": FOM.Q.dim,
        "state_dim": FOM.V.dim,
        "save_path": str(data_path),
        "seed": seed,
    }

    with open(data_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    #logger.info(f"Saved training data to {data_path}")
    print(f"Saved training data to {data_path}")

    return dataset, X_tensor, Y_tensor, data_path


# ----------------------------------------------------------------------
# Generate and save data
# ----------------------------------------------------------------------

