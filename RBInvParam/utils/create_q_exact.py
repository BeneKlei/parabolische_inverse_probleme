import numpy as np

################################# utils #################################

def add_gaussian_patch(arr, center, amp, sigma=0.5, half_size=1, bg_level=1):
    """
    Add a 2D Gaussian patch to `arr` around `center = (iy, iz)`.

    half_size=1 -> 3x3 patch, half_size=2 -> 5x5, etc.
    """
    cy, cz = center

    # define local index window
    ys = np.arange(cy - half_size, cy + half_size + 1)
    zs = np.arange(cz - half_size, cz + half_size + 1)

    # clip to array bounds just in case
    ys = ys[(ys >= 0) & (ys < arr.shape[0])]
    zs = zs[(zs >= 0) & (zs < arr.shape[1])]

    # create meshgrid of local coordinates
    Y, Z = np.meshgrid(ys, zs, indexing='ij')

    # squared distance from center (in index space)
    r2 = (Y - cy)**2 + (Z - cz)**2

    # 2D Gaussian
    gaussian = bg_level + amp * np.exp(-r2 / (2 * sigma**2))

    # add (or assign) values
    arr[ys[:, None], zs[None, :]] = gaussian

def add_constant_patch(arr, center, value, half_size=1):
    """
    Add a constant-valued square patch to `arr`.

    Parameters
    ----------
    arr : 2D array
        The array to modify.
    center : (iy, iz)
        Center index of the patch.
    value : float
        Constant value to assign in the patch.
    half_size : int
        half_size=1 → 3x3 patch
        half_size=2 → 5x5 patch
        etc.
    """
    cy, cz = center

    # index ranges
    ys = np.arange(cy - half_size, cy + half_size + 1)
    zs = np.arange(cz - half_size, cz + half_size + 1)

    # clip to valid indices
    ys = ys[(ys >= 0) & (ys < arr.shape[0])]
    zs = zs[(zs >= 0) & (zs < arr.shape[1])]

    # assign patch
    arr[np.ix_(ys, zs)] = value

#########################################################################