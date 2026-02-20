import numpy as np

################################# utils #################################


import numpy as np

def coord_to_index_1d(x, x_min, x_max, n_intervals, *, clamp=True):
    """
    Map physical coordinate x to nearest node index on a 1D grid with (n_intervals+1) nodes.
    """
    if x_max == x_min:
        raise ValueError("x_min and x_max must differ")

    s = (x - x_min) / (x_max - x_min)          # in [0,1] ideally
    i = int(np.rint(s * n_intervals))          # nearest node index

    if clamp:
        i = max(0, min(n_intervals, i))
    return i


def add_constant_patch_coords(arr, center_coords, value, half_size=1, *,
                              y_bounds=(-15.0, 15.0), z_bounds=(-15.0, 15.0)):
    """
    Add a constant-valued square patch to `arr` using a center specified in physical (y,z) coords.

    Parameters
    ----------
    arr : 2D array of shape (Ny+1, Nz+1)
    center_coords : (y, z) physical coordinates (floats)
    value : float
    half_size : int
        half_size=0 -> 1x1
        half_size=1 -> 3x3
        ...
    y_bounds : (y_min, y_max)
    z_bounds : (z_min, z_max)
    """
    Ny = arr.shape[0] - 1   # number of intervals in y
    Nz = arr.shape[1] - 1   # number of intervals in z

    y0, z0 = center_coords
    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    cy = coord_to_index_1d(y0, y_min, y_max, Ny)
    cz = coord_to_index_1d(z0, z_min, z_max, Nz)

    ys = np.arange(cy - half_size, cy + half_size + 1)
    zs = np.arange(cz - half_size, cz + half_size + 1)

    ys = ys[(ys >= 0) & (ys < arr.shape[0])]
    zs = zs[(zs >= 0) & (zs < arr.shape[1])]

    arr[np.ix_(ys, zs)] = value
    return (cy, cz)  # sometimes handy for debugging

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