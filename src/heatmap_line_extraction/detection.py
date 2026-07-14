"""
detection.py

Automatic detection of:
- heatmap region
- colorbar region
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt



def detect_heatmap_and_colorbar(
    image_file,
    saturation_threshold=20,
    min_area_fraction=0.0001,
    debug=False
):
    """
    Automatically detect heatmap and colorbar.

    Parameters
    ----------
    image_file : str

    saturation_threshold : int
        HSV saturation threshold.

    min_area_fraction : float

    debug : bool

    Returns
    -------
    HEATMAP : tuple
        (x0, y0, x1, y1)

    COLORBAR : tuple
        (x0, y0, x1, y1)
    """

    img_bgr = cv2.imread(str(image_file))
    
    choice = input(
    "\nUse automatic detection? [Y/n]: "
    ).strip().lower()

    if choice == "n":
        return manual_region_selection(image_file)
    


    if img_bgr is None:
        raise FileNotFoundError(image_file)

    img = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    h_img, w_img = img.shape[:2]

    # --------------------------------------------------
    # HSV saturation mask
    # --------------------------------------------------

    hsv = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2HSV
    )

    saturation = hsv[:, :, 1]

    mask = (
        saturation >
        saturation_threshold
    ).astype(np.uint8)

    # --------------------------------------------------
    # Clean mask
    # --------------------------------------------------

    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------
    # Connected components
    # --------------------------------------------------

    nlabels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    min_area = (
        h_img *
        w_img *
        min_area_fraction
    )

    components = []

    for i in range(1, nlabels):

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        components.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "aspect": w / h
        })

    
    print(
    f"Detected {len(components)} components"
    )

    for i, c in enumerate(components):

        print(
            f"{i}: "
            f"area={c['area']}, "
            f"w={c['w']}, "
            f"h={c['h']}"
        )

    if len(components) < 2:

        plt.figure(figsize=(8,6))
        plt.imshow(mask, cmap="gray")
        plt.title("Detection Mask")
        plt.show()
    
        print("Automatic detection failed.")
        print("Switching to manual selection...")

        return manual_region_selection(image_file)
    

    # --------------------------------------------------
    # Heatmap detection
    # Largest roughly-square component
    # --------------------------------------------------

    heatmap_score = []

    for c in components:

        aspect = c["w"] / c["h"]

        score = (
            c["area"]
            - 5000 * abs(aspect - 1)
        )

        heatmap_score.append(score)

    heatmap = components[
        np.argmax(heatmap_score)
    ]

    # --------------------------------------------------
    # Colorbar detection
    # Tall narrow region
    # --------------------------------------------------

    colorbar_candidates = []

    for c in components:

        if c is heatmap:
            continue

        ratio = c["h"] / max(c["w"], 1)

        if ratio > 2:

            distance = abs(
                c["x"]
                -
                (
                    heatmap["x"]
                    +
                    heatmap["w"]
                )
            )

            score = (
                ratio * 100
                -
                distance * 0.1
            )

            colorbar_candidates.append(
                (score, c)
            )

    if len(colorbar_candidates) == 0:
        print("Automatic detection failed.")
        print("Switching to manual selection...")

        return manual_region_selection(image_file)
    

    colorbar = max(
        colorbar_candidates,
        key=lambda x: x[0]
    )[1]

    # --------------------------------------------------
    # Convert format
    # --------------------------------------------------

    HEATMAP = (
        heatmap["x"],
        heatmap["y"],
        heatmap["x"] + heatmap["w"],
        heatmap["y"] + heatmap["h"]
    )

    COLORBAR = (
        colorbar["x"],
        colorbar["y"],
        colorbar["x"] + colorbar["w"],
        colorbar["y"] + colorbar["h"]
    )

    # --------------------------------------------------
    # Debug plot
    # --------------------------------------------------

    if debug:

        fig, ax = plt.subplots(
            figsize=(8, 6)
        )

        ax.imshow(img)

        hx0, hy0, hx1, hy1 = HEATMAP

        rect1 = plt.Rectangle(
            (hx0, hy0),
            hx1 - hx0,
            hy1 - hy0,
            fill=False,
            linewidth=3
        )

        ax.add_patch(rect1)

        cx0, cy0, cx1, cy1 = COLORBAR

        rect2 = plt.Rectangle(
            (cx0, cy0),
            cx1 - cx0,
            cy1 - cy0,
            fill=False,
            linewidth=3
        )

        ax.add_patch(rect2)

        ax.set_title(
            "Detected Heatmap and Colorbar"
        )

        plt.show()

    return HEATMAP, COLORBAR

def manual_region_selection(image_file):

    img_bgr = cv2.imread(str(image_file))
    while True:

        print("\nDraw the HEATMAP.")
        heatmap = cv2.selectROI(
            "Heatmap",
            img_bgr,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow("Heatmap")

        print("\nDraw the COLORBAR.")
        colorbar = cv2.selectROI(
            "Colorbar",
            img_bgr,
            showCrosshair=True,
            fromCenter=False,
        )
        cv2.destroyWindow("Colorbar")

        HEATMAP = (
            int(heatmap[0]),
            int(heatmap[1]),
            int(heatmap[0] + heatmap[2]),
            int(heatmap[1] + heatmap[3]),
        )

        COLORBAR = (
            int(colorbar[0]),
            int(colorbar[1]),
            int(colorbar[0] + colorbar[2]),
            int(colorbar[1] + colorbar[3]),
        )

        # Preview
        preview = img_bgr.copy()

        cv2.rectangle(
            preview,
            (HEATMAP[0], HEATMAP[1]),
            (HEATMAP[2], HEATMAP[3]),
            (0,255,0),
            2,
        )

        cv2.rectangle(
            preview,
            (COLORBAR[0], COLORBAR[1]),
            (COLORBAR[2], COLORBAR[3]),
            (255,0,0),
            2,
        )

        plt.figure(figsize=(8,6))
        plt.imshow(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        plt.title("Manual selection")
        plt.show()

        ans = input(
            "Accept these selections? [Y/n]: "
        ).strip().lower()

        if ans != "n":
            return HEATMAP, COLORBAR

        print("Please select the regions again.")