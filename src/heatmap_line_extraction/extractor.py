"""
extractor.py

Main package API.
"""

from pathlib import Path

from .detection import detect_heatmap_and_colorbar

from .reconstruction import (
    reconstruct_heatmap
)

from .profiles import (
    extract_profile
)

from .plotting import (
    plot_reconstructed_map,
    plot_profile,
    plot_comparison
)

from .utils import (
    load_image,
    build_profile_name,
    build_output_files,
    save_field_csv,
    save_profile_csv,
    validate_resolution
)


class HeatmapExtractor:
    """
    Heatmap data extraction workflow.

    Parameters
    ----------
    image_file : str

    resolution : tuple
        (nx, ny)

    value_range : tuple
        (vmin, vmax)

    x_range : tuple
        (xmin, xmax)

    y_range : tuple
        (ymin, ymax)
    """

    def __init__(
        self,
        image_file,
        resolution=(20, 20),
        value_range=(0.0, 1.0),
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        auto_detect=True,
        debug=False
    ):

        self.image_file = str(image_file)

        self.resolution = resolution

        self.value_range = value_range

        self.x_range = x_range

        self.y_range = y_range

        self.auto_detect = auto_detect

        self.debug = debug

        validate_resolution(
            resolution
        )

        self.original_image = load_image(
            self.image_file
        )

        self.heatmap_bbox = None

        self.colorbar_bbox = None

        self.field = None

    # ======================================================
    # DETECT REGIONS
    # ======================================================

    def detect_regions(self):

        (
            self.heatmap_bbox,
            self.colorbar_bbox
        ) = detect_heatmap_and_colorbar(
            self.image_file,
            debug=self.debug
        )

        return (
            self.heatmap_bbox,
            self.colorbar_bbox
        )

    # ======================================================
    # RECONSTRUCT FIELD
    # ======================================================

    def reconstruct(self):

        if self.auto_detect:

            self.detect_regions()

        self.field = reconstruct_heatmap(
            image_file=self.image_file,
            heatmap_bbox=self.heatmap_bbox,
            colorbar_bbox=self.colorbar_bbox,
            resolution=self.resolution,
            value_range=self.value_range
        )

        return self.field

    # ======================================================
    # SAVE RECONSTRUCTED FIELD
    # ======================================================

    def save_field(
        self,
        output_dir="results"
    ):

        if self.field is None:

            self.reconstruct()

        output_files = build_output_files(
            output_dir,
            "field",
            self.resolution
        )

        save_field_csv(
            self.field,
            self.x_range,
            self.y_range,
            output_files["field_csv"]
        )

        plot_reconstructed_map(
            self.field,
            self.x_range,
            self.y_range,
            output_file=output_files["field_png"]
        )

        return output_files

    # ======================================================
    # PROFILE EXTRACTION
    # ======================================================

    def extract_profile(
        self,
        profile_type,
        output_dir="results",
        save_results=True,
        **kwargs
    ):
        """
        Extract profile.

        Parameters
        ----------
        profile_type

        vertical
        horizontal
        line
        curve
        """

        if self.field is None:

            self.reconstruct()

        profile = extract_profile(
            field=self.field,
            x_range=self.x_range,
            y_range=self.y_range,
            profile_type=profile_type,
            **kwargs
        )

        profile_name = build_profile_name(
            profile_type,
            **kwargs
        )

        output_files = build_output_files(
            output_dir,
            profile_name,
            self.resolution
        )

        if save_results:

            save_field_csv(
                self.field,
                self.x_range,
                self.y_range,
                output_files["field_csv"]
            )

            save_profile_csv(
                profile,
                output_files["profile_csv"]
            )

            plot_reconstructed_map(
                self.field,
                self.x_range,
                self.y_range,
                output_file=output_files["field_png"]
            )

            plot_profile(
                profile,
                output_file=output_files["profile_png"],
                title=profile_name
            )

            plot_comparison(
                original_image=self.original_image,
                heatmap_bbox=self.heatmap_bbox,
                field=self.field,
                profile=profile,
                x_range=self.x_range,
                y_range=self.y_range,
                output_file=output_files[
                    "comparison_png"
                ]
            )

        return profile

    # ======================================================
    # SUMMARY
    # ======================================================

    def summary(self):

        print()

        print("HeatmapExtractor")

        print("----------------")

        print(
            f"Image: {self.image_file}"
        )

        print(
            f"Resolution: {self.resolution}"
        )

        print(
            f"Value Range: {self.value_range}"
        )

        print(
            f"X Range: {self.x_range}"
        )

        print(
            f"Y Range: {self.y_range}"
        )

        print()

        if self.heatmap_bbox:

            print(
                f"Heatmap: {self.heatmap_bbox}"
            )

        if self.colorbar_bbox:

            print(
                f"Colorbar: {self.colorbar_bbox}"
            )
