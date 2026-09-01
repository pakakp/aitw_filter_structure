# AITW_filter_structure

Geometry-filtering tool that turns grayscale micro-CT data (TIFF slice stacks
or `.vti` volumes) into clean binary pore/solid geometries suitable for
**OpenLB interpolated boundary conditions**, with optional exact porosity
targeting.

## Description

`filter_fit_porosity.py` is the core script. Given a stack of TIFF slices or a
`.vti` volume, it:

1. **Binarizes** the grayscale data — either with a manual threshold, or by
   solving for the signed-distance-field (SDF) threshold that hits a
   requested target porosity ("acid treatment" style erosion, directly on
   the raw grayscale, no premature binarization).
2. **Processes** the binary geometry: downsamples, smooths boundaries via SDF
   thresholding, optionally thickens the solid phase, upsamples back to full
   resolution (optionally with anti-aliased supersampling), removes isolated
   solid/pore voxels, and applies a majority filter to strip thin surface
   protrusions.
3. Optionally **re-targets porosity** after processing, since downsampling
   and smoothing can shift it away from the original target.
4. **Validates** the result (checks for remaining isolated voxels) and
   **writes** a clean binary `.vti` file, plus an `actual_porosity.txt` with
   the final porosity (geometry region only, excluding any padding).

See [`guide.md`](guide.md) for full usage modes, the processing-pipeline
order, the complete function reference, and all command-line arguments.

## Installation

Requires Python 3 with:

```bash
pip install numpy scipy vtk numba tifffile
```

`tifffile` is preferred for reading TIFF slice series; `Pillow` works as a
fallback if `tifffile` isn't installed.

## Usage

Minimal example — binarize a TIFF stack at a fixed target porosity:

```bash
python3 filter_fit_porosity.py input_slices/ output.vti \
  --porosity 0.65 --sdf-sigma 2.0 --down 2
```

This targets 65% porosity directly on the raw grayscale, downsamples by 2×
for efficiency, smooths the phase boundary, and writes `output.vti`
(`0 = pore`, `255 = solid`) along with `actual_porosity.txt`.

For manual thresholding, high-quality supersampled output, and
post-processing porosity fine-tuning, see the additional modes and full
argument list in [`guide.md`](guide.md).

## Support

Open an issue in this repository, or contact the maintainer listed below.

## Roadmap

- N/A — see project status.

## Contributing

Contributions welcome. Please open a merge request describing the change;
for anything touching the SDF porosity-targeting logic, include before/after
porosity numbers on a test volume.

## Authors and acknowledgment

Maintained by Antti Puisto (VTT).

## License

GPL.

## Project status

Active.
