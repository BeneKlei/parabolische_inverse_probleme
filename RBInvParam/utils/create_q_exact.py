import numpy as np

################################# utils #################################


import numpy as np


import numpy as np


def coord_to_index_1d(x, x_min, x_max, n_intervals, *, clamp=True):
    """
    Map physical coordinate x to nearest node index on a 1D grid
    with (n_intervals + 1) nodes.
    """
    if x_max == x_min:
        raise ValueError("x_min and x_max must differ")

    s = (x - x_min) / (x_max - x_min)
    i = int(np.rint(s * n_intervals))

    if clamp:
        i = max(0, min(n_intervals, i))
    return i


def add_constant_square_patch_from_center_coords(
    arr,
    center_coords,
    value,
    half_size=1,
    *,
    y_bounds=(-15.0, 15.0),
    z_bounds=(-15.0, 15.0),
):
    """
    Add a constant-valued square patch to `arr` using a center specified
    in physical (y, z) coordinates.

    Parameters
    ----------
    arr : 2D array of shape (Ny+1, Nz+1)
    center_coords : tuple[float, float]
        (y, z) physical coordinates of the patch center
    value : float
    half_size : int
        half_size=0 -> 1x1
        half_size=1 -> 3x3
        half_size=2 -> 5x5
        ...
    y_bounds : tuple[float, float]
    z_bounds : tuple[float, float]

    Returns
    -------
    tuple[int, int]
        (cy, cz) center node indices
    """
    Ny = arr.shape[0] - 1
    Nz = arr.shape[1] - 1

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
    return cy, cz


def add_constant_rect_patch_from_corners_coords(
    arr,
    tl_coords,
    br_coords,
    value,
    *,
    y_bounds=(-15.0, 15.0),
    z_bounds=(-15.0, 15.0),
):
    """
    Add a constant-valued rectangular patch to `arr` using top-left and
    bottom-right physical (y, z) coordinates.

    Parameters
    ----------
    arr : 2D array of shape (Ny+1, Nz+1)
    tl_coords : tuple[float, float]
        (y_top, z_left) physical coordinates
    br_coords : tuple[float, float]
        (y_bottom, z_right) physical coordinates
    value : float
    y_bounds : tuple[float, float]
    z_bounds : tuple[float, float]

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]]
        ((iy_tl, iz_tl), (iy_br, iz_br)) mapped node indices
    """
    Ny = arr.shape[0] - 1
    Nz = arr.shape[1] - 1

    y_top, z_left = tl_coords
    y_bottom, z_right = br_coords

    y_min, y_max = y_bounds
    z_min, z_max = z_bounds

    iy_tl = coord_to_index_1d(y_top, y_min, y_max, Ny)
    iz_tl = coord_to_index_1d(z_left, z_min, z_max, Nz)

    iy_br = coord_to_index_1d(y_bottom, y_min, y_max, Ny)
    iz_br = coord_to_index_1d(z_right, z_min, z_max, Nz)

    y_start, y_end = sorted((iy_tl, iy_br))
    z_start, z_end = sorted((iz_tl, iz_br))

    arr[y_start:y_end + 1, z_start:z_end + 1] = value
    return (iy_tl, iz_tl), (iy_br, iz_br)



# def add_constant_patch_coords(arr, center_coords, value, half_size=1, *,
#                               y_bounds=(-15.0, 15.0), z_bounds=(-15.0, 15.0)):
#     """
#     Add a constant-valued square patch to `arr` using a center specified in physical (y,z) coords.

#     Parameters
#     ----------
#     arr : 2D array of shape (Ny+1, Nz+1)
#     center_coords : (y, z) physical coordinates (floats)
#     value : float
#     half_size : int
#         half_size=0 -> 1x1
#         half_size=1 -> 3x3
#         ...
#     y_bounds : (y_min, y_max)
#     z_bounds : (z_min, z_max)
#     """
#     Ny = arr.shape[0] - 1   # number of intervals in y
#     Nz = arr.shape[1] - 1   # number of intervals in z

#     y0, z0 = center_coords
#     y_min, y_max = y_bounds
#     z_min, z_max = z_bounds

#     cy = coord_to_index_1d(y0, y_min, y_max, Ny)
#     cz = coord_to_index_1d(z0, z_min, z_max, Nz)

#     ys = np.arange(cy - half_size, cy + half_size + 1)
#     zs = np.arange(cz - half_size, cz + half_size + 1)

#     ys = ys[(ys >= 0) & (ys < arr.shape[0])]
#     zs = zs[(zs >= 0) & (zs < arr.shape[1])]

#     arr[np.ix_(ys, zs)] = value
#     return (cy, cz)  # sometimes handy for debugging

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