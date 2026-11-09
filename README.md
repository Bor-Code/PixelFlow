# Pixelflow

**Pixelflow** is a production-ready, composable image processing pipeline library for Python 3.12+ exposed via both a Typer CLI and a FastAPI REST API. It provides strict Pydantic v2 parameter validation, line-aware YAML pipeline definitions, immutable NumPy array execution, structured JSON logging, and multi-worker batch processing.

---

## Features

- **Composable Operations**: 16 built-in OpenCV/NumPy/Pillow operations across geometry, color, spatial filters, edge detection, thresholding, morphology, enhancement, Haar cascade face blurring, and watermarking.
- **Strict Type & Schema Safety**: Pydantic v2 parameter models, `mypy --strict` compliance, and 1-based YAML line-number error reporting.
- **Input Immutability & Edge-Case Hardening**: Input arrays are never mutated; automatic EXIF orientation transpose; decompression bomb protection; seamless handling of 2D grayscale, RGB, RGBA (alpha preservation), and 16-bit (`uint16`) images.
- **High-Throughput CLI & REST API**: `ProcessPoolExecutor` batch processing with Rich progress bars and an async FastAPI server with Request-ID tracing.

---

## Installation

Requires Python 3.11+ (optimized for Python 3.12) and [`uv`](https://github.com/astral-sh/uv):

```bash
uv sync --all-extras
```

Or using `Makefile`:

```bash
make install
```

---

## CLI Usage

### 1. Generate Sample Assets

```bash
uv run python examples/generate_samples.py
```

### 2. Validate a Pipeline

```bash
uv run pixelflow validate examples/document-cleanup.yaml
```

### 3. Run a Pipeline on a Single Image

```bash
uv run pixelflow run examples/document-cleanup.yaml examples/assets/sample_document.png --output examples/assets/cleaned_document.png --verbose
```

```bash
uv run pixelflow run examples/thumbnail-with-watermark.yaml examples/assets/sample_photo.png --output examples/assets/thumb_photo.png
```

### 4. Run Batch Processing Across a Directory

```bash
uv run pixelflow batch examples/thumbnail-with-watermark.yaml examples/assets --output-dir examples/output --workers 4
```

### 5. Inspect Registered Operations

```bash
uv run pixelflow ops list
uv run pixelflow ops describe resize
```

---

## REST API Usage

Start the API server:

```bash
make run
```

### Health & Readiness Checks

```bash
curl -s http://localhost:8000/health
curl -s http://localhost:8000/ready
```

### List Operations and JSON Schemas

```bash
curl -s http://localhost:8000/v1/operations
```

### Validate a Pipeline Definition

```bash
curl -s -X POST http://localhost:8000/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"pipeline": "name: demo\nversion: \"1.0\"\nsteps:\n  - op: flip\n    params:\n      mode: horizontal"}'
```

### Process an Image (Multipart Upload with YAML or JSON Pipeline)

```bash
curl -s -X POST http://localhost:8000/v1/process \
  -H "X-Request-ID: demo-req-001" \
  -F "image=@examples/assets/sample_photo.png;type=image/png" \
  -F "pipeline=<examples/thumbnail-with-watermark.yaml" \
  --output processed_thumb.png
```

---

## Pipeline YAML Reference

A pipeline file must define `name`, `version`, and a non-empty `steps` list. Each step specifies `op`, an optional unique `name`, and `params`. Relative file paths (such as `watermark_path`) are resolved relative to the pipeline YAML file's directory.

```yaml
name: thumbnail-with-watermark
version: "1.0"
steps:
  - op: resize
    name: create_thumbnail
    params:
      width: 320
      height: 240
      keep_aspect_ratio: true
      interpolation: area
  - op: brightness_contrast
    name: boost_contrast
    params:
      brightness: 0.03
      contrast: 1.1
  - op: sharpen
    name: enhance_edges
    params:
      kernel_size: 3
      sigma: 1.0
      amount: 0.8
  - op: watermark
    name: apply_brand_watermark
    params:
      text: "Pixelflow"
      position: bottom-right
      opacity: 0.65
      scale: 0.25
      margin: 12
```

### Built-in Operations

| Category | Operation (`op`) | Key Parameters |
| :--- | :--- | :--- |
| **Geometry** | `resize` | `width`, `height`, `keep_aspect_ratio`, `interpolation` (`nearest`, `linear`, `cubic`, `area`, `lanczos`) |
| **Geometry** | `crop` | `x`, `y`, `width`, `height` |
| **Geometry** | `rotate` | `angle` (`-360..360`), `expand`, `fill_value` |
| **Geometry** | `flip` | `mode` (`horizontal`, `vertical`, `both`) |
| **Color** | `grayscale` | `keep_channels` (`bool`) |
| **Color** | `brightness_contrast` | `brightness` (`-1.0..1.0`), `contrast` (`0.0..5.0`) |
| **Color** | `gamma` | `gamma` (`0.0..10.0`) |
| **Filters** | `gaussian_blur` | `kernel_size` (odd), `sigma_x`, `sigma_y` |
| **Filters** | `median_blur` | `kernel_size` (odd) |
| **Filters** | `bilateral_filter` | `diameter`, `sigma_color`, `sigma_space` |
| **Edges** | `canny` | `threshold1`, `threshold2`, `aperture_size` (`3`, `5`, `7`), `l2_gradient` |
| **Edges** | `sobel` | `dx`, `dy`, `kernel_size` (odd) |
| **Threshold** | `adaptive_threshold` | `max_value`, `method` (`mean`, `gaussian`), `threshold_type` (`binary`, `binary_inv`), `block_size` (odd), `constant` |
| **Threshold** | `otsu_threshold` | `max_value`, `threshold_type` (`binary`, `binary_inv`) |
| **Morphology** | `morphology` | `action` (`erode`, `dilate`, `open`, `close`), `kernel_size` (odd), `kernel_shape` (`rect`, `ellipse`, `cross`), `iterations` |
| **Enhancement** | `histogram_equalization` | `use_clahe`, `clip_limit`, `tile_grid_size` |
| **Enhancement** | `sharpen` | `kernel_size` (odd), `sigma`, `amount`, `threshold` |
| **Detection** | `face_blur` | `scale_factor`, `min_neighbors`, `min_size`, `blur_kernel_size` (odd) |
| **Overlay** | `watermark` | `text`, `watermark_path`, `position` (`top-left`, `top-right`, `bottom-left`, `bottom-right`, `center`, `custom`), `opacity`, `scale`, `margin`, `x`, `y`, `color` |

---

## Adding a Custom Operation

Create a Pydantic parameter model subclassing `OperationParams`, subclass `ImageOperation[YourParams]`, and decorate with `@register_operation`:

```python
from typing import cast
import numpy as np
from pydantic import Field
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class InvertThresholdParams(OperationParams):
    cutoff: int = Field(default=128, ge=0, le=255)


@register_operation("invert_above")
class InvertAboveOperation(ImageOperation[InvertThresholdParams]):
    params_model = InvertThresholdParams

    def _execute(self, image: ImageArray) -> ImageArray:
        result = np.where(image >= self.params.cutoff, 255 - image, image)
        return cast(ImageArray, result.astype(image.dtype))
```

Once imported, `invert_above` is automatically available in YAML pipelines, `pixelflow ops list`, `pixelflow ops describe invert_above`, and `GET /v1/operations`.

---

## Batch Processing Benchmark

Benchmark measured on 32 synthetic `640x480` RGB PNG images running the 4-step `examples/document-cleanup.yaml` pipeline across `1`, `2`, `4`, and `8` worker processes (`ProcessPoolExecutor`):

| Workers (`--workers`) | Images Processed | Total Time (s) | Throughput (images/sec) | Speedup vs 1 Worker |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 32 | 1.84s | 17.39 img/s | 1.00x |
| **2** | 32 | 1.06s | 30.19 img/s | 1.74x |
| **4** | 32 | 0.68s | 47.06 img/s | 2.71x |
| **8** | 32 | 0.54s | 59.26 img/s | 3.41x |

---

## Development & Quality Checks

```bash
make lint
make test
make docker
```

## License

MIT License. See [LICENSE](LICENSE).
