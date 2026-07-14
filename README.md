# Heatmap Line Extraction

Python package for extracting quantitative data from scientific heatmap images.

## Features

- Automatic heatmap detection
- Optional manual ROI selection
- Automatic fallback to manual selection if detection fails
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
import numpy as np

extractor = HeatmapExtractor(
    image_file="example_heatmap.png",
    resolution=(20,20),
    value_range=(0,3.3),
    x_range=(0,75),
    y_range=(0,90)
)

# Vertical line
extractor.extract_profile(
    profile_type="vertical",
    x=40
)

# Horizontal line
extractor.extract_profile(
    profile_type="horizontal",
    y=30
)

# arbitraray Line
extractor.extract_profile(
    profile_type="line",
    start=(0,0),
    end=(75,90)
)

# arbitrary Curve
x = np.linspace(0,75,500)
y = 45 + 10*np.sin(x/10)

extractor.extract_profile(
    profile_type="curve",
    x_curve=x,
    y_curve=y
)

