# RoomView Floor Plan Normalization Tool

Converts floor plan files (PNG, JPG, SVG, PDF) into clean, drafting-style SVG with standardized wall thickness.

Enforces a RoomView visual standard: uniform styling across all hotels.

## What It Does

| Before | After |
|--------|-------|
| Inconsistent wall thickness, icons, text, gray shading | Clean black-on-white vector with standardized outer/interior wall widths |

## Pipeline

### Raster inputs (PNG/JPG)

1. **Grayscale** — Convert to grayscale
2. **Binary Threshold** — Walls become solid black (tunable, default 200)
3. **Component Filtering** — Remove small components below area threshold (icons, text, noise)
4. **Morphological Close** — 3x3 kernel fills tiny gaps
5. **Skeletonize** — Reduce all walls to 1px centerlines
6. **Outer/Interior Detection** — Flood fill from image borders to separate outer boundary from interior walls
7. **Standardize Thickness** — Dilate outer skeleton to `--outer-thickness`, interior to `--inner-thickness`
8. **Potrace** — Bitmap-to-SVG in fill mode (`alphamax=0`, `opttolerance=0`)
9. **SVG Processing** — Remove transforms, apply styles, set white background

### SVG inputs
- Normalize styles only (no Potrace, no rasterization)

### PDF inputs
- Convert to SVG via `pdf2svg` (no rasterization)
- Normalize styles only

## Installation

### 1. Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Potrace (required for raster inputs)

Bundled for Windows. Otherwise:

```bash
# Mac
brew install potrace

# Linux
sudo apt install potrace
```

### 3. pdf2svg (optional, for PDF inputs)

```bash
# Mac
brew install pdf2svg

# Linux
sudo apt install pdf2svg
```

## Usage

### Quick Start (Batch Mode)

1. Drop your floor plan files into `input/`
2. Run `python normalize_floorplan.py`
3. Find clean SVGs in `output/`

### Single File

```bash
python normalize_floorplan.py floor3.png
python normalize_floorplan.py plan.pdf
python normalize_floorplan.py drawing.svg
```

### With Options

```bash
python normalize_floorplan.py floor.png --threshold 180 --outer-thickness 8 --inner-thickness 4
```

### Debug Mode

```bash
python normalize_floorplan.py floor.png --debug
```

Saves intermediate files:
- `step1_grayscale.png` — grayscale
- `step2_threshold.png` — binary threshold
- `step3_components.png` — after component removal
- `step4_closed.png` — morphological close
- `step5_skeleton.png` — wall centerlines
- `step6_exterior.png` — detected exterior region
- `step6_wall_classes.png` — outer (red) vs interior (blue)
- `step7_standardized.png` — walls at target thickness
- `floorplan_raw.svg` — raw Potrace output
- `step9_styled.svg` — final styled SVG

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `input` | *(optional)* | Input file (PNG, JPG, SVG, PDF). If omitted, batch processes `input/` |
| `--output` | `output/` | Output SVG path or directory |
| `--threshold` | `200` | Binary threshold for raster inputs |
| `--min-component-size` | `50` | Minimum pixel area for connected components |
| `--outer-thickness` | `6` | Outer wall thickness in pixels |
| `--inner-thickness` | `3` | Interior wall thickness in pixels |
| `--debug` | off | Save intermediate files |
| `--overlay` | off | Generate overlay PNG for visual QC |

## Output Requirements

- Solid black only (`fill="#000000"`)
- No gray
- No dashed paths
- No stroke outlines (`stroke="none"`)
- No transforms remaining
- Clean architectural appearance
- Uniform styling across all hotels

## Wall Standardization

Wall thickness does NOT depend on the original image:

1. All walls are skeletonized to 1px centerlines
2. Outer boundary is detected via flood fill from image borders
3. Outer walls are dilated to `--outer-thickness` pixels
4. Interior walls are dilated to `--inner-thickness` pixels
5. Result is traced with Potrace in fill mode

## Testing

```bash
python test_pipeline.py
```

## Logging

The pipeline logs:
- Input type (RASTER / SVG / PDF)
- Whether Potrace was used
- Standardization method applied
- Outer vs interior region detection results
- Final SVG output path
