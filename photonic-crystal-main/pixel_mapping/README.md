# Pixel Mapping — Silicon Hole Diameter Measurement

Utility for measuring the diameter of silicon holes in photonic crystal images, in both pixels and nanometres. Used to compare structural accuracy between real Lumerical FDTD simulation images and GAN-generated outputs.

---

## Purpose

This tool measures the diameter of a single cropped silicon hole from both a real Lumerical FDTD image and a GAN-generated output so the structural difference can be quantified before running full Lumerical FDTD simulation.

---

## Workflow

1. Crop a single silicon hole from the junction area (60-degree waveguide bend) of the target image
2. Pass the cropped image path to `measure_silicon_diameter()`
3. Function returns diameter in pixels and nanometres
4. Repeat for the real image and the generated image
5. Compare diameters — the closer they are, the more structurally accurate the GAN output

The junction area is used specifically because it is the most physically critical region of the photonic crystal structure — hole geometry at the waveguide bend directly affects light transmission efficiency.

---

## Files

| File | Description |
|---|---|
| `pixel_mapping.py` | Main script — loads image, applies Otsu threshold, calls `measure_diameter`, plots grayscale and binary side by side |
| `utils.py` | `measure_diameter()` — row-by-row horizontal span scan on binary matrix, returns widest span as diameter in pixels |

---

## How It Works

### `utils.py` — `measure_diameter(binary_matrix)`

Scans each row of a binary image matrix left-to-right and right-to-left to find the leftmost and rightmost non-zero pixel. The span (right - left) gives the horizontal diameter for that row. Returns the maximum span across all rows.

```
Row scan:  0 0 0 1 1 1 1 0 0
                ^       ^
               left    right   span = right - left
```

Simple, no dependencies, works on any binary matrix.

### `pixel_mapping.py` — `measure_silicon_diameter(path, scale_factor)`

1. Loads image as grayscale via `cv2.imread`
2. Applies Otsu thresholding (`cv2.THRESH_BINARY + cv2.THRESH_OTSU`) to get a clean binary mask
3. Passes normalised binary matrix to `measure_diameter()`
4. Multiplies pixel diameter by `scale_factor` to get physical size in nanometres
5. Plots grayscale and binary images side by side for visual verification

---

## Scale Factor

```python
scale_factor = 5 / 15  # 5 nm per 15 pixels
```

Derived from the Lumerical FDTD simulation metadata for the dataset. Adjust if working with a different resolution or simulation setup.

---

## Usage

```python
from pixel_mapping import measure_silicon_diameter

measure_silicon_diameter(
    path='path/to/cropped_circle.png',
    scale_factor=5/15
)
```

Crop the image to a single silicon hole before passing it in — the function expects one circle in the frame.

---

## Stack

Python, OpenCV, Matplotlib, NumPy
