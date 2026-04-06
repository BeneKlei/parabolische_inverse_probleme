from typing import List, Tuple, Optional
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


# def add_constant_square_patch_from_center_coords(
#     arr,
#     center_coords,
#     value,
#     half_size=1,
#     *,
#     y_bounds=(-15.0, 15.0),
#     z_bounds=(-15.0, 15.0),
# ):
#     """
#     Add a constant-valued square patch to `arr` using a center specified
#     in physical (y, z) coordinates.

#     Parameters
#     ----------
#     arr : 2D array of shape (Ny+1, Nz+1)
#     center_coords : tuple[float, float]
#         (y, z) physical coordinates of the patch center
#     value : float
#     half_size : int
#         half_size=0 -> 1x1
#         half_size=1 -> 3x3
#         half_size=2 -> 5x5
#         ...
#     y_bounds : tuple[float, float]
#     z_bounds : tuple[float, float]

#     Returns
#     -------
#     tuple[int, int]
#         (cy, cz) center node indices
#     """
#     Ny = arr.shape[0] - 1
#     Nz = arr.shape[1] - 1

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
#     return cy, cz

def add_constant_square_patch_from_center_coords(
    arr,
    center_coords,
    value,
    half_size=1,
    *,
    interpolated=False,
    distance="square",
    background = 1.0,
    y_bounds=(-15.0, 15.0),
    z_bounds=(-15.0, 15.0),
):
    """
    Add a patch to `arr` using a center specified in physical (y, z) coords.

    Modes
    -----
    interpolated=False
        Constant-valued square patch.

    interpolated=True
        Center = `value`, then linearly decays to 0 at distance `half_size`.

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
    interpolated : bool
        If True, apply linear decay from the center to the patch boundary.
        If False, fill the patch with a constant value.
    distance : {"square", "radial"}
        Distance metric used when `interpolated=True`:
        - "square": Chebyshev distance, gives square-shaped decay
        - "radial": Euclidean distance, gives circular decay
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

    if not interpolated:
        arr[np.ix_(ys, zs)] = value
        return cy, cz

    # grid
    Y, Z = np.meshgrid(ys, zs, indexing="ij")
    dy = np.abs(Y - cy)
    dz = np.abs(Z - cz)

    if half_size == 0:
        weights = np.ones((len(ys), len(zs)), dtype=float)
    else:
        if distance == "square":
            d = np.maximum(dy, dz)
        elif distance == "radial":
            d = np.sqrt(dy**2 + dz**2)
        else:
            raise ValueError("distance must be 'square' or 'radial'")

        weights = np.clip(1.0 - d / (half_size + 1), 0.0, 1.0)

    patch = background + (value - background) * weights

    arr[np.ix_(ys, zs)] = patch
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

def add_constant_rect_patch_from_corners_coords_variations(
    tl_coords: Tuple[float, float],
    br_coords: Tuple[float, float],
    value: float,
    *,
    shift_scale: float = 1.0,
    size_scale: float = 1.0,
    value_scale: float = 0.1,
    param_y_res: int = 20,
    param_z_res: int = 20,
    y_bounds: Tuple[float, float] = (-15.0, 15.0),
    z_bounds: Tuple[float, float] = (-15.0, 15.0),
    n_variations: int = 10,
    background_value: float = 1.0,
    parameter_factor: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> List[np.ndarray]:
    """
    Generate a list of q_exact arrays with slight variations of a base
    rectangular patch.

    Parameters
    ----------
    tl_coords : tuple[float, float]
        Base top-left (y_top, z_left) coordinates.
    br_coords : tuple[float, float]
        Base bottom-right (y_bottom, z_right) coordinates.
    value : float
        Base patch value.
    shift_scale : float
        Max absolute random shift applied independently to y and z.
    size_scale : float
        Controls relative patch-size perturbation.
    value_scale : float
        Max absolute random perturbation added to `value`.
    param_y_res : int
    param_z_res : int
    y_bounds : tuple[float, float]
    z_bounds : tuple[float, float]
    n_variations : int
    background_value : float
        Fill value for the array outside the rectangle.
    rng : np.random.Generator | None
        Optional RNG for reproducibility.

    Returns
    -------
    List[np.ndarray]
        List of arrays of shape (param_y_res + 1, param_z_res + 1).
    """
    if rng is None:
        rng = np.random.default_rng()

    q_exact_list: List[np.ndarray] = []

    base_tl_y, base_tl_z = tl_coords
    base_br_y, base_br_z = br_coords

    base_size_y = base_br_y - base_tl_y
    base_size_z = base_br_z - base_tl_z

    for _ in range(n_variations):
        q_exact = np.full(
            (param_y_res + 1, param_z_res + 1),
            background_value,
            dtype=float
        )

        dy = rng.uniform(-shift_scale, shift_scale)
        dz = rng.uniform(-shift_scale, shift_scale)

        scale_y = 1.0 + rng.uniform(-size_scale, size_scale) * 0.1
        scale_z = 1.0 + rng.uniform(-size_scale, size_scale) * 0.1

        new_size_y = base_size_y * scale_y
        new_size_z = base_size_z * scale_z

        tl_y = base_tl_y + dy
        tl_z = base_tl_z + dz

        br_y = tl_y + new_size_y
        br_z = tl_z + new_size_z

        patch_value = value + rng.uniform(-value_scale, value_scale)

        add_constant_rect_patch_from_corners_coords(
            q_exact,
            tl_coords=(tl_y, tl_z),
            br_coords=(br_y, br_z),
            value=value,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
        )

        q_exact = parameter_factor * q_exact
        q_exact = q_exact.flatten()
        q_exact = np.array([q_exact])

        q_exact_list.append(q_exact)

    return q_exact_list