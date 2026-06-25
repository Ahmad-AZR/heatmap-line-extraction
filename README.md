# Heatmap Line Extraction

Python package for extracting quantitative data from scientific heatmap images.

## Features

- Automatic heatmap detection
- Automatic colorbar detection
- Heatmap reconstruction
- Vertical profiles
- Horizontal profiles
- Arbitrary line profiles
- Arbitrary curve profiles
- CSV export
- PNG export

## Installation

pip install heatmap-line-extraction

## Quick Start

```python
from heatmap_line_extraction import HeatmapExtractor

extractor = HeatmapExtractor(
    image_file="heatmap.png",
    resolution=(300,300),
    value_range=(0,100),
    x_range=(0,75),
    y_range=(0,90)
)

extractor.extract_profile(
    profile_type="vertical",
    x=40
)