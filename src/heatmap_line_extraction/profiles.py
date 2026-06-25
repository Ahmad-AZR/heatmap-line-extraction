"""
profiles.py

Profile extraction utilities.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator


# ==========================================================
# INTERPOLATOR
# ==========================================================

def build_interpolator(
    field,
    x_range,
    y_range
):
    """
    Build interpolator for reconstructed field.

    Parameters
    ----------
    field : ndarray
        shape (ny,nx)

    x_range : tuple
        (xmin,xmax)

    y_range : tuple
        (ymin,ymax)

    Returns
    -------
    interpolator
    """

    ny, nx = field.shape

    x = np.linspace(
        x_range[0],
        x_range[1],
        nx
    )

    y = np.linspace(
        y_range[0],
        y_range[1],
        ny
    )

    return RegularGridInterpolator(
        (y, x),
        field,
        bounds_error=False,
        fill_value=np.nan
    )


# ==========================================================
# DISTANCE
# ==========================================================

def cumulative_distance(
    x,
    y
):
    """
    Compute cumulative path length.
    """

    dx = np.diff(x)
    dy = np.diff(y)

    ds = np.sqrt(
        dx**2 +
        dy**2
    )

    distance = np.zeros(
        len(x)
    )

    distance[1:] = np.cumsum(ds)

    return distance


# ==========================================================
# SAMPLE PATH
# ==========================================================

def sample_path(
    interpolator,
    x_path,
    y_path
):
    """
    Sample interpolated values
    along arbitrary path.
    """

    points = np.column_stack(
        [y_path, x_path]
    )

    values = interpolator(
        points
    )

    distance = cumulative_distance(
        x_path,
        y_path
    )

    return {
        "distance": distance,
        "x": x_path,
        "y": y_path,
        "value": values
    }


# ==========================================================
# VERTICAL PROFILE
# ==========================================================

def vertical_profile(
    interpolator,
    x_value,
    y_range,
    n_points=500
):
    """
    x = constant
    """

    y = np.linspace(
        y_range[0],
        y_range[1],
        n_points
    )

    x = np.full_like(
        y,
        x_value
    )

    return sample_path(
        interpolator,
        x,
        y
    )


# ==========================================================
# HORIZONTAL PROFILE
# ==========================================================

def horizontal_profile(
    interpolator,
    y_value,
    x_range,
    n_points=500
):
    """
    y = constant
    """

    x = np.linspace(
        x_range[0],
        x_range[1],
        n_points
    )

    y = np.full_like(
        x,
        y_value
    )

    return sample_path(
        interpolator,
        x,
        y
    )


# ==========================================================
# ARBITRARY LINE
# ==========================================================

def line_profile(
    interpolator,
    start,
    end,
    n_points=500
):
    """
    Straight line profile.
    """

    x = np.linspace(
        start[0],
        end[0],
        n_points
    )

    y = np.linspace(
        start[1],
        end[1],
        n_points
    )

    return sample_path(
        interpolator,
        x,
        y
    )


# ==========================================================
# CURVE PROFILE
# ==========================================================

def curve_profile(
    interpolator,
    x,
    y
):
    """
    User-supplied curve.
    """

    return sample_path(
        interpolator,
        np.asarray(x),
        np.asarray(y)
    )


# ==========================================================
# GENERIC EXTRACTOR
# ==========================================================

def extract_profile(
    field,
    x_range,
    y_range,
    profile_type,
    **kwargs
):
    """
    Generic profile extraction.

    Parameters
    ----------
    profile_type

    vertical
    horizontal
    line
    curve

    Returns
    -------
    dict
    """

    interpolator = build_interpolator(
        field,
        x_range,
        y_range
    )

    if profile_type == "vertical":

        return vertical_profile(
            interpolator,
            x_value=kwargs["x"],
            y_range=y_range,
            n_points=kwargs.get(
                "n_points",
                500
            )
        )

    elif profile_type == "horizontal":

        return horizontal_profile(
            interpolator,
            y_value=kwargs["y"],
            x_range=x_range,
            n_points=kwargs.get(
                "n_points",
                500
            )
        )

    elif profile_type == "line":

        return line_profile(
            interpolator,
            start=kwargs["start"],
            end=kwargs["end"],
            n_points=kwargs.get(
                "n_points",
                500
            )
        )

    elif profile_type == "curve":

        return curve_profile(
            interpolator,
            x=kwargs["x_curve"],
            y=kwargs["y_curve"]
        )

    else:

        raise ValueError(
            f"Unknown profile type: "
            f"{profile_type}"
        )
