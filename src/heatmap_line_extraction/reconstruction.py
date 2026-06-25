"""
reconstruction.py

Heatmap reconstruction utilities.
"""

import cv2
import numpy as np
from scipy.spatial.distance import cdist


# ==========================================================
# IMAGE CROPPING
# ==========================================================

def crop_region(image_rgb, bbox):
    """
    Crop image using bounding box.

    Parameters
    ----------
    image_rgb : ndarray

    bbox : tuple
        (x0,y0,x1,y1)

    Returns
    -------
    cropped : ndarray
    """

    x0, y0, x1, y1 = bbox

    return image_rgb[y0:y1, x0:x1]


# ==========================================================
# COLORBAR SAMPLING
# ==========================================================

def sample_colorbar(
    colorbar_img,
    value_range,
    n_samples=512
):
    """
    Sample colors along colorbar.

    Parameters
    ----------
    colorbar_img : ndarray

    value_range : tuple
        (vmin,vmax)

    n_samples : int

    Returns
    -------
    colors : ndarray
        shape (N,3)

    values : ndarray
        shape (N,)
    """

    height = colorbar_img.shape[0]

    y_positions = np.linspace(
        0,
        height - 1,
        n_samples
    ).astype(int)

    colors = []

    for y in y_positions:

        row = colorbar_img[y, :, :]

        rgb = np.mean(
            row,
            axis=0
        )

        colors.append(rgb)

    colors = np.asarray(colors)

    values = np.linspace(
        value_range[1],
        value_range[0],
        n_samples
    )

    return colors, values


# ==========================================================
# RGB -> VALUE
# ==========================================================

def rgb_to_value(
    rgb,
    cb_colors,
    cb_values
):
    """
    Convert RGB color into physical value.

    Parameters
    ----------
    rgb : array-like

    cb_colors : ndarray

    cb_values : ndarray

    Returns
    -------
    float
    """

    d = np.linalg.norm(
        cb_colors - rgb,
        axis=1
    )

    idx = np.argmin(d)

    return cb_values[idx]


# ==========================================================
# CELL AVERAGING
# ==========================================================

def average_cell_color(cell):
    """
    Average RGB color of one cell.

    Parameters
    ----------
    cell : ndarray

    Returns
    -------
    rgb : ndarray
    """

    return np.mean(
        cell.reshape(-1, 3),
        axis=0
    )


# ==========================================================
# FIELD RECONSTRUCTION
# ==========================================================

def reconstruct_field(
    heatmap_img,
    cb_colors,
    cb_values,
    nx,
    ny,
    flip_y=True
):
    """
    Reconstruct numerical field.

    Parameters
    ----------
    heatmap_img : ndarray

    cb_colors : ndarray

    cb_values : ndarray

    nx : int

    ny : int

    flip_y : bool

    Returns
    -------
    field : ndarray
        shape (ny,nx)
    """

    h, w = heatmap_img.shape[:2]

    cell_w = w / nx
    cell_h = h / ny

    field = np.zeros(
        (ny, nx),
        dtype=float
    )

    for iy in range(ny):

        for ix in range(nx):

            x0 = int(ix * cell_w)
            x1 = int((ix + 1) * cell_w)

            y0 = int(iy * cell_h)
            y1 = int((iy + 1) * cell_h)

            cell = heatmap_img[
                y0:y1,
                x0:x1
            ]

            rgb = average_cell_color(
                cell
            )

            value = rgb_to_value(
                rgb,
                cb_colors,
                cb_values
            )

            field[iy, ix] = value

    if flip_y:
        field = np.flipud(field)

    return field


# ==========================================================
# COMPLETE WORKFLOW
# ==========================================================

def reconstruct_heatmap(
    image_file,
    heatmap_bbox,
    colorbar_bbox,
    resolution,
    value_range
):
    """
    Full reconstruction workflow.

    Parameters
    ----------
    image_file : str

    heatmap_bbox : tuple

    colorbar_bbox : tuple

    resolution : tuple
        (nx,ny)

    value_range : tuple
        (vmin,vmax)

    Returns
    -------
    field : ndarray
    """

    img_bgr = cv2.imread(
        str(image_file)
    )

    if img_bgr is None:

        raise FileNotFoundError(
            image_file
        )

    img = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    heatmap_img = crop_region(
        img,
        heatmap_bbox
    )

    colorbar_img = crop_region(
        img,
        colorbar_bbox
    )

    cb_colors, cb_values = (
        sample_colorbar(
            colorbar_img,
            value_range
        )
    )

    nx, ny = resolution

    field = reconstruct_field(
        heatmap_img,
        cb_colors,
        cb_values,
        nx,
        ny,
        flip_y=True
    )

    return field
