# PixelFlow

**High-Performance Composable Image Processing Engine & Web Studio**

PixelFlow is an enterprise-grade image processing pipeline system, Typer CLI, FastAPI microservice, and interactive Web Studio built for Python 3.12+. Engineered for speed, safety, and composability, PixelFlow executes 54+ native operations across computational photography, document scanning (CamScanner suite), cinematic color grading, biometric visa composition, and vector annotation with immutable NumPy arrays and strict Pydantic v2 validation.

---

## Highlights

- **54+ Built-in Operations**: Comprehensive catalog covering geometry, spatial filters, edge detection, CLAHE enhancement, 3-way color wheels, cinematic LUTs, face blurring, AI background removal, and vector watermarking.
- **PixelFlow Studio (Web GUI)**: Modern, responsive web studio featuring a dark OLED aesthetic, Bento Grid workflow launchpad, 0ms hardware-accelerated preview, Curtain Before/After comparison, and Zero-UI focus mode.
- **CamScanner Document Suite**: Automatic document deskew, 4-corner perspective warping, shadow removal, magic color whitening, signature extraction, PII masking (ID/IBAN redaction), and vector stamps.
- **Biometric Photo & Visa Composer**: Automated multi-up print layouts for ICAO 35x45mm (Schengen, Turkish ID/Passport) and US Visa 50x50mm specifications with cut guides.
- **Squoosh-Grade Export Optimizer**: Live compression comparison across WEBP, JPEG, and PNG formats with dynamic size savings estimation.
- **Multi-Page PDF Processing**: Seamless PDF-to-image extraction, page manipulation (split, rotate, reorder), and multipage PDF re-compilation.
- **Strict Type & Schema Safety**: Fully typed with Pydantic v2 parameter models, `mypy --strict` compliance, and line-aware YAML syntax validation.
- **High-Throughput Concurrency**: Multi-core batch processing via `ProcessPoolExecutor` with Rich CLI progress visualization, alongside an asynchronous FastAPI service with request-ID distributed tracing.

---

## Table of Contents

- [PixelFlow](#pixelflow)
  - [Highlights](#highlights)
  - [Table of Contents](#table-of-contents)
  - [System Architecture](#system-architecture)
  - [Installation](#installation)
  - [PixelFlow Studio (Web Interface)](#pixelflow-studio-web-interface)
  - [CLI Quickstart](#cli-quickstart)
    - [1. Generate Sample Assets](#1-generate-sample-assets)
    - [2. Validate Pipeline Definitions](#2-validate-pipeline-definitions)
    - [3. Execute a Pipeline on a Single Image](#3-execute-a-pipeline-on-a-single-image)
    - [4. Multi-Worker Batch Processing](#4-multi-worker-batch-processing)
    - [5. Inspect Registered Operations](#5-inspect-registered-operations)
  - [REST API Reference](#rest-api-reference)
    - [Endpoints](#endpoints)
    - [API Usage Examples](#api-usage-examples)
  - [Pipeline Specification](#pipeline-specification)
    - [YAML Syntax Example](#yaml-syntax-example)
  - [Operation Catalog](#operation-catalog)
    - [Document & Office Suite](#document--office-suite)
    - [Color, Tone & Grading](#color-tone--grading)
    - [Cinematic & Creative Filters](#cinematic--creative-filters)
    - [AI, Retouch & Detection](#ai-retouch--detection)
    - [Geometry & Transforms](#geometry--transforms)
    - [Spatial Filters & Smoothing](#spatial-filters--smoothing)
    - [Edge Detection & Morphology](#edge-detection--morphology)
    - [Overlay, Stamp & Watermark](#overlay-stamp--watermark)
  - [Creating Custom Operations](#creating-custom-operations)
  - [Performance Benchmarks](#performance-benchmarks)
  - [Development & Quality Assurance](#development--quality-assurance)
  - [License](#license)

---

## System Architecture

```
                    +------------------------------------------+
                    |          PixelFlow Studio (Web)          |
                    |   Bento Launchpad | 0ms Preview | Wheels |
                    +--------------------+---------------------+
                                         |
                       REST / Multipart  |  CLI Commands
                                         v
+--------------------------------------------------------------------------+
|                            PixelFlow Core                                |
|                                                                          |
|  +--------------------+  +--------------------+  +--------------------+  |
|  |   Pipeline Engine  |  | Operation Registry |  |   Pydantic Models  |  |
|  | (YAML/JSON Loader) |  |   (54+ Handlers)   |  |   (Strict Schema)  |  |
|  +--------------------+  +--------------------+  +--------------------+  |
|                                                                          |
|  +--------------------+  +--------------------+  +--------------------+  |
|  |     Image I/O      |  |  Batch Multi-Core  |  |  Structured Logs   |  |
|  | (NumPy/OpenCV/PDF) |  | (ProcessPoolExec)  |  | (JSON with Req-ID) |  |
|  +--------------------+  +--------------------+  +--------------------+  |
+--------------------------------------------------------------------------+
```

---

## Installation

### Prerequisites

- Python 3.11+ (Python 3.12 recommended)
- `uv` (recommended) or standard `pip`

### Install with `uv`

```bash
git clone https://github.com/your-org/pixelflow.git
cd pixelflow
uv sync --all-extras
```

### Install with `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e ".[all]"
```

---

## PixelFlow Studio (Web Interface)

PixelFlow ships with a built-in, zero-dependency browser application optimized for creative and office workflows.

### Starting the Studio

```bash
uv run uvicorn pixelflow.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your web browser.

### Key Studio Capabilities

- **Zero-Latency Visual Feedback**: Tone and exposure adjustments (brightness, contrast, vibrance, saturation) apply instant client-side CSS filter transformations (0ms response) while debouncing the full-precision OpenCV/NumPy pipeline run with automatic in-flight request cancellation (`AbortController`).
- **Curtain & Split Comparison**: Compare original and processed assets with an interactive draggable curtain split-view or dual-viewport split mode.
- **3-Way Color Wheels**: Professional Lift, Gamma, and Gain controls with master luminance sliders.
- **Squoosh-Style Optimizer**: Interactive modal for comparing real-time image compression ratios across WebP, JPEG, and PNG formats.
- **Multi-Page Document Manager**: Reorder, rotate, split, and merge multi-page PDF documents directly in the browser.
- **Biometric Card Studio**: Generate ready-to-print passport and visa sheets with cut-line overlays.
- **Command Palette & Themes**: Press `Ctrl+K` for instant command search. Supports Tokyo Night, DaVinci Gold, Cyberpunk, and Titanium OLED themes.

---

## CLI Quickstart

The PixelFlow CLI (`pixelflow`) is powered by Typer and provides commands for execution, validation, batch processing, and inspection.

### 1. Generate Sample Assets

Create synthetic test documents and photographic assets:

```bash
uv run python examples/generate_samples.py
```

### 2. Validate Pipeline Definitions

Verify syntax, operation availability, and parameter bounds with line-number error reporting:

```bash
uv run pixelflow validate examples/document-cleanup.yaml
```

### 3. Execute a Pipeline on a Single Image

```bash
uv run pixelflow run examples/document-cleanup.yaml examples/assets/sample_document.png \
  --output examples/assets/cleaned_document.png \
  --verbose
```

Apply a watermark and thumbnail resize:

```bash
uv run pixelflow run examples/thumbnail-with-watermark.yaml examples/assets/sample_photo.png \
  --output examples/assets/thumb_photo.png
```

### 4. Multi-Worker Batch Processing

Process an entire directory of images concurrently using worker processes:

```bash
uv run pixelflow batch examples/thumbnail-with-watermark.yaml examples/assets \
  --output-dir examples/output \
  --workers 4
```

### 5. Inspect Registered Operations

List all available operations or inspect parameter requirements:

```bash
# List all 54 registered operations
uv run pixelflow ops list

# Inspect detailed schema and parameters for an operation
uv run pixelflow ops describe magic_color
uv run pixelflow ops describe perspective_warp
```

---

## REST API Reference

PixelFlow exposes a production-ready FastAPI service with OpenAPI specification, health endpoints, and request tracing.

### Endpoints

| Method | Path | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the interactive PixelFlow Studio UI |
| `GET` | `/health` | Service liveness probe |
| `GET` | `/ready` | Service readiness probe |
| `GET` | `/v1/operations` | List registered operations with full JSON schemas |
| `POST` | `/v1/validate` | Validate pipeline YAML or JSON syntax |
| `POST` | `/v1/process` | Execute a pipeline on an uploaded image |
| `POST` | `/v1/detect-corners` | Detect document boundaries for perspective correction |
| `POST` | `/v1/compose-id` | Generate biometric passport and visa print sheets |
| `POST` | `/v1/multipage-pdf` | Assemble multiple images into a unified PDF |

### API Usage Examples

#### Process Image with Pipeline

```bash
curl -s -X POST http://127.0.0.1:8000/v1/process \
  -H "X-Request-ID: req-001" \
  -F "image=@examples/assets/sample_document.png;type=image/png" \
  -F "pipeline=<examples/document-cleanup.yaml" \
  --output output_cleaned.png
```

#### Validate Pipeline via JSON

```bash
curl -s -X POST http://127.0.0.1:8000/v1/validate \
  -H "Content-Type: application/json" \
  -d '{"pipeline": "name: test\nversion: \"1.0\"\nsteps:\n  - op: brightness_contrast\n    params:\n      brightness: 0.1\n      contrast: 1.2"}'
```

---

## Pipeline Specification

Pipelines are declared in YAML or JSON. Each pipeline must include `name`, `version`, and an ordered list of `steps`. Every step specifies the target operation (`op`), an optional descriptive `name`, and required or optional `params`.

### YAML Syntax Example

```yaml
name: professional-document-scan
version: "1.0"
steps:
  # 1. Correct perspective skew
  - op: deskew
    name: auto_align_text
    params:
      max_angle: 45.0

  # 2. CamScanner magic color filter
  - op: magic_color
    name: enhance_paper_and_ink
    params:
      mode: magic
      whiten_strength: 0.85
      sharpen_strength: 1.2

  # 3. Clean paper shadows
  - op: shadow_removal
    name: remove_lighting_gradients
    params:
      kernel_size: 31

  # 4. Official company seal overlay
  - op: stamp
    name: apply_official_stamp
    params:
      text: "APPROVED"
      kind: circular
      color: [180, 20, 20]
      x: 420
      y: 650
```

---

## Operation Catalog

PixelFlow includes 54 built-in operations across 8 functional domains:

### Document & Office Suite

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `magic_color` | CamScanner-grade document contrast enhancement | `mode`, `whiten_strength`, `sharpen_strength` |
| `shadow_removal` | Eliminates uneven background illumination and paper shadows | `kernel_size`, `strength` |
| `deskew` | Detects text orientation angle and straightens document | `max_angle`, `background_color` |
| `perspective_warp` | 4-corner perspective un-warping to rectangular plane | `src_points`, `dst_size` |
| `adaptive_threshold` | Adaptive binarization for high-contrast document scans | `max_value`, `method`, `block_size`, `constant` |
| `otsu_threshold` | Global Otsu binarization for clear document text | `max_value`, `threshold_type` |
| `signature_extract` | Separates handwritten signatures onto transparent alpha canvas | `threshold`, `color_retention` |
| `auto_pii` | Automated redaction for sensitive IDs, IBANs, and numbers | `categories`, `padding` |
| `redact` | Manual box redaction for privacy masking | `regions`, `fill_color` |
| `biometric_sheet` | Prepares standardized passport and visa print sheets | `preset`, `layout`, `draw_cut_lines` |

### Color, Tone & Grading

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `brightness_contrast` | Linear luminance and contrast adjustment | `brightness`, `contrast` |
| `gamma` | Non-linear gamma curve transformation | `gamma` |
| `vibrance` | Intelligent saturation preserving natural skin tones | `vibrance`, `saturation` |
| `curves` | Spline-based tone curve adjustment per channel or master | `channel`, `points` |
| `histogram_equalization` | CLAHE adaptive local contrast balance | `use_clahe`, `clip_limit`, `tile_grid_size` |
| `grayscale` | Grayscale luminance conversion | `keep_channels` |
| `color_wheels` | 3-way Lift (shadows), Gamma (midtones), and Gain (highlights) | `lift`, `gamma`, `gain`, `lift_y`, `gamma_y`, `gain_y` |
| `gradient_map` | Maps luminance levels across continuous color gradients | `color_stops` |
| `selective_adjust` | Adjusts hue, saturation, and lightness for specific color bands | `target_color`, `tolerance`, `hue_shift` |
| `selective_color` | Preserves single color family while desaturating background | `target_color`, `threshold` |
| `blend` | Blends image layers using Photoshop blend modes | `mode`, `opacity` |

### Cinematic & Creative Filters

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `preset` | Instant cinematic LUT presets (Teal & Orange, Cyberpunk, etc.) | `preset` |
| `cinematic_preset` | Advanced film grading emulation with grain and curves | `look`, `intensity` |
| `duotone` | Dual-color shadow and highlight mapping | `shadow_color`, `highlight_color` |
| `color_splash` | Isolate specific color splash while keeping remainder monochrome | `color`, `tolerance` |
| `tilt_shift` | Miniature optical depth-of-field blur | `focus_y`, `focus_height`, `blur_strength` |
| `miniature` | Alternative diorama miniature effect | `intensity`, `saturation_boost` |
| `vignette` | Smooth peripheral radial shading | `radius`, `softness`, `color` |
| `glitch` | Digital displacement and artifact generation | `slice_count`, `offset_max` |
| `rgb_split` | Chromatic aberration color channel separation | `shift_x`, `shift_y` |
| `film_grain` | Photorealistic film grain simulation | `amount`, `size` |
| `grain` | Monochromatic textured film grain | `intensity` |
| `frame` | Decorative borders, polaroid styles, and passe-partout | `frame_width`, `color`, `style` |

### AI, Retouch & Detection

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `auto_enhance` | 1-Tap intelligent auto exposure and color balancing | `strength` |
| `face_blur` | Automated Haar-cascade face detection and censorship | `scale_factor`, `min_neighbors`, `blur_kernel_size` |
| `inpaint` | Telea and Navier-Stokes based content-aware hole filling | `mask_path`, `radius`, `method` |
| `background_removal` | Foreground extraction and alpha mask generation | `sensitivity`, `feather` |
| `sharpen` | Unsharp masking filter for edge detail restoration | `kernel_size`, `sigma`, `amount`, `threshold` |
| `denoise` | Fast luminance and chrominance noise suppression | `h`, `template_window`, `search_window` |
| `nlmeans_denoise` | Non-local means denoising for photographic fidelity | `strength`, `color` |

### Geometry & Transforms

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `resize` | High-quality image scaling | `width`, `height`, `keep_aspect_ratio`, `interpolation` |
| `crop` | Exact coordinate bounding box cropping | `x`, `y`, `width`, `height` |
| `rotate` | Free-angle arbitrary rotation with canvas expansion | `angle`, `expand`, `fill_value` |
| `flip` | Horizontal, vertical, or dual-axis mirroring | `mode` (`horizontal`, `vertical`, `both`) |

### Spatial Filters & Smoothing

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `gaussian_blur` | Gaussian convolution smoothing | `kernel_size`, `sigma_x`, `sigma_y` |
| `median_blur` | Median rank filter for salt-and-pepper noise removal | `kernel_size` |
| `bilateral_filter` | Edge-preserving smoothing filter | `diameter`, `sigma_color`, `sigma_space` |

### Edge Detection & Morphology

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `canny` | Canny multi-stage edge detection | `threshold1`, `threshold2`, `aperture_size`, `l2_gradient` |
| `sobel` | Directional derivative gradient calculation | `dx`, `dy`, `kernel_size` |
| `morphology` | Mathematical morphological filtering | `action` (`erode`, `dilate`, `open`, `close`), `kernel_size` |

### Overlay, Stamp & Watermark

| Operation (`op`) | Description | Key Parameters |
| :--- | :--- | :--- |
| `watermark` | Text or image watermark overlay with opacity and anchor | `text`, `watermark_path`, `position`, `opacity`, `scale` |
| `stamp` | Official seals, verified badges, and date marks | `text`, `kind`, `color`, `x`, `y` |
| `markup` | Vector bounding boxes, review badges, and arrows | `kind`, `text`, `x1`, `y1`, `x2`, `y2`, `color` |
| `qr_code` | Generates standalone high-density QR code | `data`, `size`, `fill_color`, `back_color` |
| `qr_overlay` | Overlays scannable QR badge onto source image | `data`, `position`, `size`, `margin` |

---

## Creating Custom Operations

PixelFlow is modular and extensible. To implement a custom operation:

1. Subclass `OperationParams` using Pydantic fields.
2. Subclass `ImageOperation[YourParams]`.
3. Decorate with `@register_operation("your_operation_name")`.

```python
from typing import cast
import numpy as np
from pydantic import Field
from pixelflow.core.operation import ImageArray, ImageOperation, OperationParams
from pixelflow.core.registry import register_operation


class ColorThresholdParams(OperationParams):
    cutoff: int = Field(default=128, ge=0, le=255, description="Luminance threshold cutoff.")


@register_operation("invert_above")
class InvertAboveOperation(ImageOperation[ColorThresholdParams]):
    params_model = ColorThresholdParams

    def _execute(self, image: ImageArray) -> ImageArray:
        result = np.where(image >= self.params.cutoff, 255 - image, image)
        return cast(ImageArray, result.astype(image.dtype))
```

Once defined or imported, `invert_above` becomes instantly accessible across YAML pipelines, CLI commands, REST endpoints, and the Web Studio.

---

## Performance Benchmarks

### Batch Processing Scaling

Tested on 32 synthetic `640x480` RGB images running the 4-step `examples/document-cleanup.yaml` pipeline across multiple worker processes (`ProcessPoolExecutor`):

| Workers (`--workers`) | Images Processed | Total Time (s) | Throughput (img/s) | Speedup vs 1 Worker |
| :---: | :---: | :---: | :---: | :---: |
| **1** | 32 | 1.84s | 17.39 img/s | 1.00x |
| **2** | 32 | 1.06s | 30.19 img/s | 1.74x |
| **4** | 32 | 0.68s | 47.06 img/s | 2.71x |
| **8** | 32 | 0.54s | 59.26 img/s | 3.41x |

### Core Pipeline Latency (Per-Operation)

Measured on a `720x480` image (Decode + Execution + Encode):

| Operation | Decode | Execute | Encode | Total Latency |
| :--- | :---: | :---: | :---: | :---: |
| `brightness_contrast` | 6.6 ms | 17.3 ms | 3.3 ms | **27.2 ms** |
| `vibrance` | 5.1 ms | 15.3 ms | 3.6 ms | **24.0 ms** |
| `histogram_equalization` (CLAHE) | 3.3 ms | 20.9 ms | 7.6 ms | **31.8 ms** |
| `magic_color` (CamScanner) | 3.8 ms | 33.1 ms | 6.7 ms | **43.6 ms** |
| `sharpen` (Unsharp) | 5.5 ms | 30.3 ms | 6.3 ms | **42.1 ms** |
| `preset` (Teal & Orange) | 7.4 ms | 34.6 ms | 7.1 ms | **49.1 ms** |
| `auto_enhance` (Smart AI) | 3.1 ms | 243.9 ms | 4.6 ms | **251.6 ms** |

---

## Development & Quality Assurance

Run code formatting, type checking, test suites, and Docker builds:

```bash
# Code formatting and lint checks (ruff, black, isort)
make lint

# Static type analysis with mypy strict mode
uv run mypy src/pixelflow

# Run full test suite with coverage
make test

# Build container image
make docker
```

---

## License

PixelFlow is released under the [MIT License](LICENSE).
