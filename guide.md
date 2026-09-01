# filter_fit_porosity.py

Geometry filter for preparing binary pore/solid volumes for **OpenLB interpolated
boundary conditions**. Takes a TIFF slice series or a `.vti` grayscale/binary
volume, binarizes it (optionally targeting an exact porosity via a signed-distance-field
threshold search), processes it (downsample/smooth/thicken/pad), validates it,
and writes a clean binary `.vti` (`0 = pore`, `255 = solid`).

## What's Included

### SDF-Based Porosity Control
- **Continuous SDF on grayscale** — `compute_sdf_from_grayscale()` builds the
  signed-distance field directly from raw intensity, without binarizing first,
  so erosion ("acid treatment") can start from any grayscale level.
- **Binary search threshold solve** — `adjust_porosity_sdf()` binary-searches
  the SDF threshold (up to 20 iterations, tolerance 0.001) to hit a target
  porosity on the raw grayscale image.
- **Direct grayscale targeting** via `--porosity` and `--sdf-sigma`.
- **Supersampling** (`--upsample-intermediate`, `--upsample-final`) for
  anti-aliased upsampling instead of blocky nearest-neighbor scaling.
- **Post-processing porosity adjustment** (`--adjust-porosity-post`) — a second,
  independent SDF threshold search (`adjust_porosity_post()`) run on the
  already-processed binary array, for fine-tuning after downsampling/smoothing
  has shifted the porosity away from target.

### Optimizations
- Numba `@jit(nopython=True, parallel=True)` kernels for the two hot loops:
  `_remove_thin_protrusions_kernel()` and `_find_isolated_kernel()`.
- Block-majority downsampling / nearest-neighbor upsampling.
- Isolated-voxel detection and removal (`remove_isolated_voxels()`), run in
  sweeps (up to 10) until no more isolated voxels are found.
- Majority filtering (`majority_filter()`) to remove thin single-voxel
  protrusions from the solid/pore surface.

## Binary Convention
`0 = pore`, `255 = solid` (raw grayscale is treated as `solid = high intensity`
by default; pass `--solid-dark` to invert).

## Key Usage Modes

### Mode 1: SDF-Based Porosity Control (RECOMMENDED)
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --porosity 0.65 --sdf-sigma 2.0 --down 2
```
- Targets porosity on raw grayscale via `adjust_porosity_sdf()`.
- Efficient single-pass processing.
- Final porosity may drift from target due to downstream processing
  (downsampling, smoothing, majority filtering) — use
  `--adjust-porosity-post` to fine-tune it back.

### Mode 2: Manual Threshold
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --threshold 127 --down 2
```
- Simple `binarize_from_threshold()` cutoff.
- Fast processing.
- No porosity targeting — porosity is whatever the threshold happens to give.

### Mode 3: With Supersampling (High Quality)
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --porosity 0.65 --sdf-sigma 2.0 --down 2 \
  --upsample-intermediate 3 --upsample-final 2
```
- Upsamples to an intermediate resolution, then block-majority-downsamples
  back down to the target net factor for anti-aliased boundaries.
- Higher quality but slower; `--upsample-final` must be ≥ 1.0 and
  ≤ `--upsample-intermediate`.

### Mode 4: With Post-Processing Adjustment
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --porosity 0.65 --sdf-sigma 2.0 --down 2 \
  --adjust-porosity-post
```
- Runs `adjust_porosity_post()` after the full pipeline (downsample → smooth →
  thicken → upsample → supersample → final smooth → isolated-voxel removal →
  majority filter) to re-hit the requested porosity.
- Only takes effect when `--porosity` is also set.

## Processing Pipeline (in order)

`process_geometry()` runs, in this sequence:

1. Downsample (`block_majority_downsample`, `--down`)
2. Low-res SDF smoothing (`sdf_smooth`, `--smooth-low-iters` × `--smooth-sigma`)
3. Thicken solid (`thicken_solid`, `--thicken`)
4. Upsample back to original resolution (`upsample_nearest`)
5. Optional supersampling (`--upsample-intermediate` / `--upsample-final`)
6. Full-res final SDF smoothing (`--final-smooth-iters` × `--final-smooth-sigma`)
7. Isolated-voxel removal (up to 10 sweeps, `remove_isolated_voxels`)
8. Majority filter (`--majority-filter`, default 1 iteration)
9. Padding/cropping (`add_padding`, `--pad`, `--pad-xy-only`) — applied **last**

Then, optionally, `--adjust-porosity-post`, followed by validation
(`report_validation`) and writing the `.vti` output.

## Function Summary

**Numba kernels:**
- `_remove_thin_protrusions_kernel()` — flips a voxel if it has ≤ N same-phase
  face neighbors.
- `_find_isolated_kernel()` — flags fully-isolated solid/pore voxels (0 or 6
  same-phase face neighbors), with optional boundary preservation.

**Binarization:**
- `binarize_from_threshold()` — simple threshold, `solid_is_high` controls
  direction.
- `adjust_porosity_sdf()` — SDF-based binary search on raw grayscale (NEW).
- `adjust_porosity_post()` — post-processing SDF threshold fine-tuning on an
  already-binary array.

**SDF utilities:**
- `compute_sdf()` — signed distance field of a binary array (positive = pore,
  negative = solid).
- `compute_sdf_from_grayscale()` — SDF-like field computed directly from raw
  intensity, normalized to `[-20, +20]`.
- `sdf_smooth()` — Gaussian-blur the SDF and re-threshold at 0 to smooth
  boundaries.

**Processing:**
- `block_majority_downsample()` — majority-vote block downsampling.
- `upsample_nearest()` — nearest-neighbor upsampling, with optional exact
  target-shape cropping/padding.
- `thicken_solid()` — binary dilation of the solid phase.
- `remove_isolated_voxels()` — removes/flips fully-isolated solid and pore
  voxels.
- `majority_filter()` — removes thin protrusions via the Numba kernel,
  iterated `--majority-filter` times.
- `add_padding()` — adds pore-space padding (`thickness > 0`) or crops
  (`thickness < 0`); returns the geometry-only slice for porosity accounting.

**I/O:**
- `load_vti_raw()` — read a `.vti` grayscale/binary volume via VTK.
- `load_tiff_series_raw()` — read a directory/glob of `.tif`/`.tiff` slices
  (via `tifffile` if available, else Pillow) and stack into a volume.
- `write_vti()` — write the final `uint8` binary volume to `.vti`.
- `looks_binary_0_255()` — sampled check for whether the input is already
  strictly binary (`0/255` or `0/1`), to skip thresholding.

**Validation:**
- `report_validation()` — reports remaining isolated-solid voxel count and
  porosity; PASS/FAIL against interpolated-BC suitability.
- `calculate_porosity()` — mean pore fraction, optionally restricted to a
  geometry slice (excluding padding).

## Arguments

**Main files:**
- `input` — TIFF directory/glob pattern, or a `.vti` file
- `output` — Output `.vti` file (`0=pore`, `255=solid`)

**Thresholding:**
- `-t, --threshold` — Manual binarization threshold for grayscale inputs (default: 127)
- `--solid-dark` — Invert contrast (solid = low intensity)
- `--porosity` — Target porosity `[0,1]` for the SDF approach; accepts `None`/omit to disable
- `--sdf-sigma` — SDF Gaussian blur sigma (higher = more boundary smearing, default: 0.5)

**Processing:**
- `--down` — Downsample factor, 1 = no downsampling (default: 2)
- `--smooth-low-iters` — Low-res SDF smoothing passes (default: 0)
- `--smooth-sigma` — Gaussian sigma for low-res smoothing (default: 0.6)
- `--thicken` — Dilate solid by N voxels (default: 0)
- `--final-smooth-iters` — Full-res SDF smoothing passes after upsampling (default: 0)
- `--final-smooth-sigma` — Gaussian sigma for final smoothing, keep low: 0.3–0.7 (default: 0.5)

**Supersampling:**
- `--upsample-intermediate` — Intermediate upsampling factor (e.g. 3); must be used with `--upsample-final`
- `--upsample-final` — Target net upsampling factor (e.g. 2); must satisfy `1.0 ≤ upsample-final ≤ upsample-intermediate`

**Geometry:**
- `--pad` — Add N voxels of pore padding (positive) or crop N voxels (negative); applied last, after all other processing
- `--pad-xy-only` — Restrict padding/cropping to X and Y faces (not Z)

**Adjustments:**
- `--adjust-porosity-post` — Fine-tune porosity after processing (requires `--porosity`)
- `--majority-filter` — Protrusion-removal iterations, 0 = off, 2–3 for aggressive filtering (default: 1)
- `--allow-boundary-removal` — Allow isolated-voxel removal to touch boundary voxels

**Other:**
- `-v, --verbose` — Print detailed progress at each pipeline stage
- `--min-gradient` — Gradient threshold used by `report_validation()` (default: 0.2)

## Outputs

- `<output>.vti` — the final binary geometry (`uint8`, 0/255).
- `actual_porosity.txt` — the final porosity computed over the geometry region
  only (padding excluded), written next to where the script is run. Intended
  for downstream AiiDA workflow consumption.

## Example Workflows

### Quick test with defaults:
```bash
python3 filter_fit_porosity.py input.vti output.vti
```

### Precise porosity control:
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --porosity 0.65 --sdf-sigma 1.5 --adjust-porosity-post
```

### High-quality output:
```bash
python3 filter_fit_porosity.py input/ output.vti \
  --porosity 0.65 --sdf-sigma 2.0 --down 1 \
  --upsample-intermediate 3 --upsample-final 1.5 \
  --final-smooth-iters 1 --final-smooth-sigma 0.5
```

### Manual threshold with padding:
```bash
python3 filter_fit_porosity.py input.vti output.vti \
  --threshold 100 --down 2 --pad 20
```

## Requirements

`numpy`, `scipy`, `vtk`, `numba`, and either `tifffile` (preferred) or `Pillow`
for TIFF-series input.
