"""
utils.py

General utility functions.
"""

from pathlib import Path

import cv2
import numpy as np


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(image_file):
    """
    Load image as RGB.
    """

    img_bgr = cv2.imread(str(image_file))

    if img_bgr is None:

        raise FileNotFoundError(
            f"Cannot read image: {image_file}"
        )

    return cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )


# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

def create_output_dir(
    output_dir
):
    """
    Create output directory.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        exist_ok=True,
        parents=True
    )

    return output_dir


# ==========================================================
# PROFILE NAME
# ==========================================================

def build_profile_name(
    profile_type,
    **kwargs
):
    """
    Generate descriptive profile name.
    """

    if profile_type == "vertical":

        return (
            f"vertical_"
            f"{kwargs['x']:.2f}um"
        ).replace(".", "p")

    elif profile_type == "horizontal":

        return (
            f"horizontal_"
            f"{kwargs['y']:.2f}um"
        ).replace(".", "p")

    elif profile_type == "line":

        start = kwargs["start"]
        end = kwargs["end"]

        return (
            f"line_"
            f"x{start[0]:.1f}_y{start[1]:.1f}"
            f"_to_"
            f"x{end[0]:.1f}_y{end[1]:.1f}"
        ).replace(".", "p")

    elif profile_type == "curve":

        return "curve_profile"

    else:

        return "profile"
        
        
        
# ==========================================================
# EXPORT FIELD CSV
# ==========================================================

def save_field_csv(
    field,
    x_range,
    y_range,
    filename
):
    """
    Save reconstructed field.

    Output columns:

    x,y,value
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

    rows = []

    for iy in range(ny):

        for ix in range(nx):

            rows.append([
                x[ix],
                y[iy],
                field[iy, ix]
            ])

    np.savetxt(
        filename,
        np.asarray(rows),
        delimiter=",",
        header="x,y,value",
        comments=""
    )

    print(
        f"Saved: {filename}"
    )
    
# ==========================================================
# EXPORT PROFILE CSV
# ==========================================================

def save_profile_csv(
    profile,
    filename
):
    """
    Save extracted profile.
    """

    data = np.column_stack([

        profile["distance"],

        profile["x"],

        profile["y"],

        profile["value"]

    ])

    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=
        "distance,x,y,value",
        comments=""
    )

    print(
        f"Saved: {filename}"
    )
    
# ==========================================================
# OUTPUT FILES
# ==========================================================

def build_output_files(
    output_dir,
    profile_name,
    resolution
):
    """
    Standardized filenames.
    """

    output_dir = create_output_dir(
        output_dir
    )

    nx, ny = resolution

    return {

        "field_csv":

        output_dir /
        f"field_{nx}x{ny}.csv",

        "field_png":

        output_dir /
        f"field_{nx}x{ny}.png",

        "profile_csv":

        output_dir /
        f"profile_{profile_name}.csv",

        "profile_png":

        output_dir /
        f"{profile_name}_profile.png",

        "comparison_png":

        output_dir /
        f"{profile_name}_comparison.png"

    }
    
# ==========================================================
# VALIDATION
# ==========================================================

def validate_resolution(
    resolution
):
    """
    Validate grid resolution.
    """

    if len(resolution) != 2:

        raise ValueError(
            "resolution must be (nx, ny)"
        )

    nx, ny = resolution

    if nx <= 0 or ny <= 0:

        raise ValueError(
            "resolution values must be positive"
        )

    return nx, ny
