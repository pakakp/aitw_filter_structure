#!/usr/bin/env python3
"""
filter_fit_porosity.py — Geometry filter for OpenLB interpolated boundary conditions.

Features:
  - SDF-based porosity control: --porosity with --sdf-sigma for "acid treatment" simulation
  - Grayscale-aware processing: treats raw intensity as continuous density field
  - Downsampling and smoothing: efficient multi-resolution processing
  - Supersampling: high-quality upsampling via anti-aliasing
  - Padding/cropping: add pore space or crop geometry
  - Validation: checks for isolated voxels and geometry quality

Binary convention: 0 = pore, 255 = solid
"""

import argparse
import glob
import os
import sys

import numpy as np
import vtk
from numba import jit
from scipy import ndimage
from vtk.util import numpy_support

# ══════════════════════════════════════════════════════════════════════════════
# NUMBA-OPTIMIZED FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

@jit(nopython=True, cache=True, parallel=True)
def _remove_thin_protrusions_kernel(arr, neighbor_threshold):
    """Numba-compiled kernel for protrusion removal."""
    nx, ny, nz = arr.shape
    result = arr.copy()
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                phase = arr[i, j, k]
                same_phase_neighbors = 0
                if arr[i-1, j, k] == phase: same_phase_neighbors += 1
                if arr[i+1, j, k] == phase: same_phase_neighbors += 1
                if arr[i, j-1, k] == phase: same_phase_neighbors += 1
                if arr[i, j+1, k] == phase: same_phase_neighbors += 1
                if arr[i, j, k-1] == phase: same_phase_neighbors += 1
                if arr[i, j, k+1] == phase: same_phase_neighbors += 1
                
                if same_phase_neighbors <= neighbor_threshold:
                    result[i, j, k] = 0 if phase == 255 else 255
    
    return result


@jit(nopython=True, cache=True, parallel=True)
def _find_isolated_kernel(arr, preserve_boundaries):
    """Numba-compiled kernel for isolated voxel detection."""
    nx, ny, nz = arr.shape
    solid = arr == 255
    isolated_solid = np.zeros_like(solid)
    isolated_pore = np.zeros_like(solid)
    boundary = np.zeros_like(solid)
    
    if preserve_boundaries:
        boundary[0,:,:] = boundary[-1,:,:] = True
        boundary[:,0,:] = boundary[:,-1,:] = True
        boundary[:,:,0] = boundary[:,:,-1] = True
    
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            for k in range(1, nz-1):
                if boundary[i,j,k]: continue
                
                neighbor_count = (solid[i-1,j,k] + solid[i+1,j,k] +
                                 solid[i,j-1,k] + solid[i,j+1,k] +
                                 solid[i,j,k-1] + solid[i,j,k+1])
                
                if solid[i,j,k] and neighbor_count == 0:
                    isolated_solid[i,j,k] = True
                if not solid[i,j,k] and neighbor_count == 6:
                    isolated_pore[i,j,k] = True
    
    return isolated_solid, isolated_pore


# ══════════════════════════════════════════════════════════════════════════════
# Helper Functions
# ══════════════════════════════════════════════════════════════════════════════

def binarize_from_threshold(raw, threshold, solid_is_high=True):
    """Convert grayscale to binary."""
    if solid_is_high:
        solid = raw >= threshold
    else:
        solid = raw <= threshold
    return np.where(solid, 255, 0).astype(np.uint8)


def block_majority_downsample(arr, factor):
    """Downsample by majority vote in blocks."""
    if factor <= 1:
        return arr.copy()

    nx, ny, nz = arr.shape
    rx = (factor - nx % factor) % factor
    ry = (factor - ny % factor) % factor
    rz = (factor - nz % factor) % factor
    if rx or ry or rz:
        arr = np.pad(arr, ((0, rx), (0, ry), (0, rz)), constant_values=0)

    nx2, ny2, nz2 = arr.shape
    blk = arr.reshape(nx2//factor, factor, ny2//factor, factor, nz2//factor, factor)
    solid_count = np.sum(blk == 255, axis=(1, 3, 5))
    majority = solid_count > (factor**3 / 2)
    return np.where(majority, 255, 0).astype(np.uint8)


def upsample_nearest(arr, factor, target_shape=None):
    """Upsample using nearest neighbor."""
    if factor <= 1:
        res = arr.copy()
    else:
        res = np.repeat(np.repeat(np.repeat(arr, factor, axis=0),
                                  factor, axis=1), factor, axis=2)
    if target_shape is not None:
        tx, ty, tz = target_shape
        res = res[:tx, :ty, :tz]
        pad = ((0, max(0, tx - res.shape[0])),
               (0, max(0, ty - res.shape[1])),
               (0, max(0, tz - res.shape[2])))
        if any(p[1] > 0 for p in pad):
            res = np.pad(res, pad, constant_values=0)
    return res


def compute_sdf(arr):
    """Signed distance field: positive in pore, negative in solid."""
    solid = arr == 255
    d_pore  = ndimage.distance_transform_edt(~solid).astype(np.float32)
    d_solid = ndimage.distance_transform_edt(solid).astype(np.float32)
    sdf = d_pore.copy()
    sdf[solid] = -d_solid[solid]
    return sdf


def compute_sdf_from_grayscale(raw, solid_is_high=True):
    """
    Compute SDF directly from grayscale image without binarization.
    Treats grayscale as continuous density field for solid phase.
    This enables true "acid treatment" - eroding from any intensity level.
    
    Args:
        raw: Grayscale image (typically 0-255 range)
        solid_is_high: True if high values = solid
    
    Returns:
        SDF where positive = pore region, negative = solid region
    """
    raw_min = float(np.min(raw))
    raw_max = float(np.max(raw))
    
    if raw_max == raw_min:
        return np.ones_like(raw, dtype=np.float32) * 100.0
    
    raw_norm = (raw.astype(np.float32) - raw_min) / (raw_max - raw_min)
    
    if solid_is_high:
        # High grayscale = solid (negative SDF)
        # Low grayscale = pore (positive SDF)
        sdf = (0.5 - raw_norm) * 40.0
    else:
        sdf = (raw_norm - 0.5) * 40.0
    
    return sdf.astype(np.float32)


def sdf_smooth(arr, sigma):
    """SDF-based smoothing."""
    sdf = compute_sdf(arr)
    sdf_g = ndimage.gaussian_filter(sdf, sigma=sigma)
    return np.where(sdf_g < 0.0, 255, 0).astype(np.uint8)


def thicken_solid(arr, iterations):
    """Dilate solid phase."""
    if iterations <= 0:
        return arr.copy()
    struct = ndimage.generate_binary_structure(3, 1)
    solid = ndimage.binary_dilation(arr == 255, structure=struct, iterations=iterations)
    return np.where(solid, 255, 0).astype(np.uint8)


def remove_isolated_voxels(arr, preserve_boundaries=True):
    """Remove isolated voxels."""
    isolated_solid, isolated_pore = _find_isolated_kernel(arr, preserve_boundaries)
    
    n_iso_solid = int(np.sum(isolated_solid))
    n_iso_pore = int(np.sum(isolated_pore))
    
    arr_out = arr.copy()
    arr_out[isolated_solid] = 0
    arr_out[isolated_pore] = 255
    
    return arr_out, {'removed_solid': n_iso_solid, 'removed_pore': n_iso_pore}


def majority_filter(arr, iterations=1):
    """Remove single-voxel protrusions using Numba."""
    if iterations <= 0:
        return arr.copy()
    
    print(f"  Applying majority filter ({iterations} iteration{'s' if iterations != 1 else ''})...")
    
    result = arr.copy()
    
    for iteration in range(iterations):
        result = _remove_thin_protrusions_kernel(result, neighbor_threshold=3)
        if iteration == 0:
            # Report change after first iteration
            n_changed = np.sum(result != arr)
            print(f"    Changed {n_changed:,} voxels in iteration 1")
    
    final_changed = np.sum(result != arr)
    print(f"    Total voxels changed: {final_changed:,}")
    
    return result


def add_padding(arr, thickness, xy_only=False):
    """Add pore padding or crop geometry."""
    if thickness == 0:
        # No padding - geometry is entire array
        return arr.copy(), (slice(None), slice(None), slice(None))
    
    if thickness > 0:
        # PADDING: add pore space
        if xy_only:
            pad_width = ((thickness, thickness), (thickness, thickness), (0, 0))
        else:
            pad_width = ((thickness, thickness), (thickness, thickness), (thickness, thickness))
        
        # Pad with 0 (pore)
        arr_out = np.pad(arr, pad_width, mode='constant', constant_values=0)
        
        # Geometry region is the original data, not the padding
        nx, ny, nz = arr_out.shape
        if xy_only:
            geometry_slice = (slice(thickness, nx-thickness),
                            slice(thickness, ny-thickness),
                            slice(None))
        else:
            geometry_slice = (slice(thickness, nx-thickness),
                            slice(thickness, ny-thickness),
                            slice(thickness, nz-thickness))
    
    else:
        # CROPPING: remove voxels from edges
        crop = abs(thickness)
        nx, ny, nz = arr.shape
        
        if xy_only:
            # Crop only X and Y
            if crop * 2 >= nx or crop * 2 >= ny:
                raise ValueError(f"Crop amount {crop} too large for shape {arr.shape}")
            arr_out = arr[crop:nx-crop, crop:ny-crop, :]
        else:
            # Crop all dimensions
            if crop * 2 >= nx or crop * 2 >= ny or crop * 2 >= nz:
                raise ValueError(f"Crop amount {crop} too large for shape {arr.shape}")
            arr_out = arr[crop:nx-crop, crop:ny-crop, crop:nz-crop]
        
        # After cropping, entire array is geometry
        geometry_slice = (slice(None), slice(None), slice(None))
    
    return arr_out, geometry_slice


def porosity_arg_type(value):
    """
    argparse type for --porosity. Tolerates the literal string 'None'/'none'
    (and empty string) as meaning "no target" -- equivalent to omitting the
    flag entirely -- rather than crashing when a caller passes it explicitly
    instead of leaving it out. Defense-in-depth: callers should still omit
    the flag when they mean "off", but this keeps the tool robust if they
    don't.
    """
    if value is None or value.strip().lower() in ('none', ''):
        return None
    return float(value)


def calculate_porosity(arr, geometry_slice=None):
    """Calculate porosity."""
    if geometry_slice is None:
        return float(np.mean(arr == 0))
    else:
        geometry_region = arr[geometry_slice]
        return float(np.mean(geometry_region == 0))


def adjust_porosity_sdf(raw, target, sdf_sigma=0.5, solid_is_high=True, verbose=False):
    """
    Adjust porosity by thresholding SDF of raw grayscale image.
    Uses binary search to find optimal SDF threshold achieving target porosity.
    
    Args:
        raw: Grayscale image (0-255 range)
        target: Target porosity [0, 1]
        sdf_sigma: Gaussian blur sigma for SDF (higher = more boundary smearing)
        solid_is_high: True if high grayscale = solid
        verbose: Print iteration details
    
    Returns:
        Binary array with target porosity (already binarized from SDF thresholding)
    """
    target_porosity = target
    tolerance = 0.001
    
    print(f"\nAdjusting porosity via SDF threshold (acid treatment simulation)...")
    print(f"  Target porosity: {target_porosity:.6f}")
    print(f"  SDF blur sigma: {sdf_sigma} (boundary smearing)")
    
    # Compute SDF directly on raw grayscale (no premature binarization)
    print(f"  Computing SDF on raw grayscale image...")
    sdf = compute_sdf_from_grayscale(raw, solid_is_high=solid_is_high)
    sdf_smooth_arr = ndimage.gaussian_filter(sdf, sigma=sdf_sigma)
    
    # Binary search on SDF threshold
    threshold_min = float(np.min(sdf_smooth_arr))
    threshold_max = float(np.max(sdf_smooth_arr))
    
    print(f"  SDF range: [{threshold_min:.2f}, {threshold_max:.2f}]")
    print(f"  Searching for optimal SDF threshold...")
    
    best_threshold = 0.0
    best_error = float('inf')
    
    for iteration in range(20):
        threshold_mid = (threshold_min + threshold_max) / 2.0
        
        arr_test = np.where(sdf_smooth_arr < threshold_mid, 255, 0).astype(np.uint8)
        porosity_test = np.mean(arr_test == 0)
        error = abs(porosity_test - target_porosity)
        
        if verbose:
            print(f"    iter {iteration+1}: SDF_threshold={threshold_mid:+.4f}, "
                  f"porosity={porosity_test:.6f}, error={error:.6f}")
        
        if error < best_error:
            best_error = error
            best_threshold = threshold_mid
        
        if error < tolerance:
            if verbose:
                print(f"    ✓ Converged!")
            break
        
        if porosity_test < target_porosity:
            # Porosity too low - need more pore, so need LOWER (more negative) threshold
            threshold_max = threshold_mid
        else:
            # Porosity too high - need less pore, so need HIGHER (more positive) threshold
            threshold_min = threshold_mid
        
        if abs(threshold_max - threshold_min) < 1e-6:
            if verbose:
                print(f"    Converged (threshold range < 1e-6)")
            break
    
    # Apply best SDF threshold to get final binary array
    arr_final = np.where(sdf_smooth_arr < best_threshold, 255, 0).astype(np.uint8)
    final_porosity = np.mean(arr_final == 0)
    
    print(f"  Erosion level (SDF threshold): {best_threshold:+.4f} voxel distance")
    print(f"  Result porosity: {final_porosity:.6f}")
    print(f"  Target: {target_porosity:.6f}, Error: {abs(final_porosity - target_porosity):.6f}")
    
    return arr_final


def adjust_porosity_post(arr, target, geometry_slice=None, verbose=False):
    """
    Post-processing porosity adjustment: fine-tune already-binarized array via SDF thresholding.
    Useful for adjusting after downsampling/upsampling effects.
    
    Args:
        arr: Already-binarized binary array
        target: Target porosity [0, 1]
        geometry_slice: Optional slice for geometry region (excluding padding)
        verbose: Print iteration details
    
    Returns:
        Adjusted binary array
    """
    current = calculate_porosity(arr, geometry_slice)
    
    if abs(current - target) < 1e-5:
        print(f"  Porosity already at target: {current:.6f}")
        return arr.copy()
    
    print(f"  Current porosity: {current:.6f}, target: {target:.6f}")
    print(f"  Using SDF threshold tuning (post-processing)...")
    
    sdf = compute_sdf(arr)
    sdf_smooth = ndimage.gaussian_filter(sdf, sigma=0.5)
    
    threshold_min = -5.0
    threshold_max = +5.0
    
    best_threshold = 0.0
    best_error = abs(current - target)
    
    for iteration in range(20):
        threshold_mid = (threshold_min + threshold_max) / 2.0
        
        # Apply threshold
        arr_test = np.where(sdf_smooth < threshold_mid, 255, 0).astype(np.uint8)
        porosity_test = calculate_porosity(arr_test, geometry_slice)
        error = abs(porosity_test - target)
        
        if verbose:
            print(f"    iter {iteration+1}: threshold={threshold_mid:+.4f}, "
                  f"porosity={porosity_test:.6f}, error={error:.6f}")
        
        # Track best so far
        if error < best_error:
            best_error = error
            best_threshold = threshold_mid
        
        # Check if we've converged
        if error < 1e-5:
            if verbose:
                print(f"    ✓ Converged!")
            break
        
        # Binary search update
        if porosity_test < target:
            # Need more pore → increase threshold (shift boundary into solid)
            threshold_max = threshold_mid
        else:
            # Need more solid → decrease threshold (shift boundary into pore)
            threshold_min = threshold_mid
        
        # Safety: if search range is too small, we're done
        if abs(threshold_max - threshold_min) < 1e-6:
            if verbose:
                print(f"    Converged (threshold range < 1e-6)")
            break
    
    # Apply best threshold found
    arr_final = np.where(sdf_smooth < best_threshold, 255, 0).astype(np.uint8)
    final_porosity = calculate_porosity(arr_final, geometry_slice)
    
    print(f"  Porosity adjusted: {current:.6f} → {final_porosity:.6f}")
    print(f"  Error: {abs(final_porosity - target):.6f}")
    
    return arr_final


def report_validation(arr, min_gradient=0.2, geometry_slice=None):
    """Validate geometry for Bouzidi BCs."""
    if geometry_slice is not None:
        arr_region = arr[geometry_slice]
    else:
        arr_region = arr
    
    solid = arr_region == 255
    struct = ndimage.generate_binary_structure(3, 1)
    
    solid_eroded = ndimage.binary_erosion(solid, structure=struct, iterations=1, border_value=False)
    interface = solid & ~solid_eroded
    n_iface = int(np.sum(interface))
    
    kernel = np.array([
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,1,0],[1,0,1],[0,1,0]],
        [[0,0,0],[0,1,0],[0,0,0]]
    ], dtype=np.uint8)
    
    neighbour_count = ndimage.convolve(solid.astype(np.uint8), kernel, mode='nearest', cval=0)
    remaining_iso = int(np.sum(solid & (neighbour_count == 0)))

    print(f"  Isolated solid left: {remaining_iso}")
    print(f"  Porosity           : {np.mean(arr_region==0):.6f}")
    
    if remaining_iso == 0:
        print("  ✓ PASS — geometry suitable for interpolated BC")
    else:
        if remaining_iso > 0:
            print(f"  ✗ {remaining_iso} isolated voxels remain")


# ══════════════════════════════════════════════════════════════════════════════
# I/O Functions
# ══════════════════════════════════════════════════════════════════════════════

def load_tiff_series_raw(path):
    """Load TIFF series."""
    try:
        import tifffile
        _have_tifffile = True
    except ImportError:
        _have_tifffile = False
    
    try:
        from PIL import Image as PILImage
        _have_pil = True
    except ImportError:
        _have_pil = False
    
    if not _have_tifffile and not _have_pil:
        sys.exit("ERROR: Install tifffile or Pillow to read TIFF files")
    
    if os.path.isdir(path):
        pattern = os.path.join(path, "*.tiff")
        files = sorted(glob.glob(pattern))
        if not files:
            pattern = os.path.join(path, "*.tif")
            files = sorted(glob.glob(pattern))
    else:
        files = sorted(glob.glob(path))
    
    if not files:
        sys.exit(f"ERROR: No TIFF files found at {path!r}")
    
    print(f"  Found {len(files)} TIFF slices")
    
    slices = []
    if _have_tifffile:
        import tifffile
        for f in files:
            slices.append(tifffile.imread(f).astype(np.float32))
    else:
        from PIL import Image as PILImage
        for f in files:
            slices.append(np.array(PILImage.open(f), dtype=np.float32))
    
    volume = np.stack(slices, axis=0)
    raw = np.transpose(volume, (2, 1, 0))
    
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)
    return raw, spacing, origin


def load_vti_raw(path):
    """Load VTI file."""
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(path)
    reader.Update()
    vd = reader.GetOutput()
    
    spacing = vd.GetSpacing()
    origin = vd.GetOrigin()
    dims = vd.GetDimensions()
    
    scalars = vd.GetPointData().GetScalars() or vd.GetPointData().GetArray(0)
    raw = numpy_support.vtk_to_numpy(scalars).reshape(dims, order='F').astype(np.float32)
    
    return raw, spacing, origin


def write_vti(path, arr, spacing=(1,1,1), origin=(0,0,0), array_name="ImageFile"):
    """Write VTI file."""
    arr = arr.astype(np.uint8)
    flat = arr.flatten(order='F')
    
    vtk_arr = numpy_support.numpy_to_vtk(flat, deep=True,
                                        array_type=vtk.VTK_UNSIGNED_CHAR)
    vtk_arr.SetName(array_name)
    
    image = vtk.vtkImageData()
    image.SetDimensions(arr.shape[0], arr.shape[1], arr.shape[2])
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.GetPointData().SetScalars(vtk_arr)
    
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image)
    writer.SetDataModeToAscii()
    writer.Write()


def looks_binary_0_255(raw):
    """Detect if already binary."""
    rmin = float(np.min(raw))
    rmax = float(np.max(raw))
    if not ((rmin == 0.0 and rmax == 255.0) or (rmin == 0.0 and rmax == 1.0)):
        return False
    
    flat = raw.ravel()
    n = flat.size
    if n == 0:
        return False
    step = max(1, n // 200000)
    samp = flat[::step]
    if rmax == 255.0:
        return np.all((samp == 0.0) | (samp == 255.0))
    else:
        return np.all((samp == 0.0) | (samp == 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# Main Processing Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def process_geometry(arr, args, original_shape):
    """Main processing pipeline.
    
    Args:
        arr: Already-binarized binary array (0=pore, 255=solid)
        args: Command line arguments
        original_shape: Original shape for reference
    """
    
    if args.verbose:
        print(f"  Binarised porosity: {float(np.mean(arr == 0)):.4f}")
    
    # Store shape for downsampling/upsampling
    shape_after_binarize = arr.shape
    
    # Downsample
    if args.down > 1:
        arr_low = block_majority_downsample(arr, args.down)
        if args.verbose:
            print(f"  Low-res shape: {arr_low.shape}, porosity: {float(np.mean(arr_low == 0)):.4f}")
    else:
        arr_low = arr.copy()
    
    # Low-res smoothing
    if args.smooth_low_iters > 0:
        print(f"\nSDF smoothing at low res ({args.smooth_low_iters} × σ={args.smooth_sigma})…")
        for i in range(args.smooth_low_iters):
            arr_low = sdf_smooth(arr_low, args.smooth_sigma)
        if args.verbose:
            print(f"  After low-res smoothing: porosity={float(np.mean(arr_low == 0)):.4f}")
    
    # Thicken
    if args.thicken > 0:
        arr_low = thicken_solid(arr_low, args.thicken)
        if args.verbose:
            print(f"  After thickening: porosity={float(np.mean(arr_low == 0)):.4f}")
    
    # Upsample back to original resolution
    if args.down > 1:
        arr_final = upsample_nearest(arr_low, args.down, target_shape=shape_after_binarize)
        if args.verbose:
            print(f"  After upsample: porosity={float(np.mean(arr_final == 0)):.4f}")
    else:
        arr_final = arr_low.copy()
    
    # Supersampling (high-quality anti-aliased upsampling)
    if args.upsample_intermediate is not None:
        print(f"\n*** SUPERSAMPLING ***")
        print(f"High-quality upsampling via anti-aliasing:")
        print(f"  Current shape: {arr_final.shape}")
        print(f"  Strategy: upsample {args.upsample_intermediate}× → downsample to {args.upsample_final}× net")
        
        intermediate_factor = int(np.round(args.upsample_intermediate))
        print(f"\nStep 1: Upsampling {intermediate_factor}× to intermediate resolution…")
        arr_intermediate = upsample_nearest(arr_final, intermediate_factor)
        print(f"  Intermediate shape: {arr_intermediate.shape}")
        print(f"  Porosity: {np.mean(arr_intermediate==0):.4f}")
        
        # Downsample to final resolution
        downsample_factor = args.upsample_intermediate / args.upsample_final
        print(f"\nStep 2: Downsampling by {downsample_factor:.2f}× (anti-aliasing)…")
        
        target_shape = tuple(int(np.round(d * args.upsample_final)) for d in arr_final.shape)
        
        if downsample_factor > 1.0:
            ds_factor = int(np.round(downsample_factor))
            if abs(downsample_factor - ds_factor) > 0.01:
                print(f"  WARNING: Non-integer downsample factor {downsample_factor:.2f} rounded to {ds_factor}")
            
            arr_final = block_majority_downsample(arr_intermediate, ds_factor)
            
            # Adjust to exact target shape if needed
            if arr_final.shape != target_shape:
                print(f"  Adjusting to exact target shape {target_shape}…")
                arr_final = upsample_nearest(arr_final, 1, target_shape=target_shape)
        else:
            arr_final = arr_intermediate
        
        print(f"  Final shape: {arr_final.shape}")
        print(f"  Porosity after supersampling: {np.mean(arr_final==0):.4f}")
        print(f"  Net upsampling: {arr_final.shape[0]/original_shape[0]:.2f}×")
    
    # Final smoothing
    if args.final_smooth_iters > 0:
        print(f"\n*** FINAL SMOOTHING (after upsampling) ***")
        print(f"SDF smoothing at full resolution ({args.final_smooth_iters} × σ={args.final_smooth_sigma})…")
        for i in range(args.final_smooth_iters):
            arr_final = sdf_smooth(arr_final, args.final_smooth_sigma)
        if args.verbose:
            print(f"  After final smoothing: porosity={float(np.mean(arr_final == 0)):.4f}")
    
    # Remove isolated voxels
    preserve_boundaries = not args.allow_boundary_removal
    total_solid = total_pore = 0
    for sweep in range(10):
        arr_final, stats = remove_isolated_voxels(arr_final, preserve_boundaries=preserve_boundaries)
        n = stats['removed_solid'] + stats['removed_pore']
        total_solid += stats['removed_solid']
        total_pore  += stats['removed_pore']
        if n == 0:
            break
    
    if args.verbose and (total_solid > 0 or total_pore > 0):
        print(f"  Removed {total_solid} isolated solid, {total_pore} isolated pore voxels")
    
    # Majority filter
    print(f"\nApplying majority filter to remove surface irregularities…")
    arr_final = majority_filter(arr_final, args.majority_filter)
    
    # ADD PADDING AT THE VERY END
    if args.pad != 0:
        arr_final, geometry_slice = add_padding(arr_final, args.pad, xy_only=args.pad_xy_only)
        if args.verbose:
            geom_porosity = calculate_porosity(arr_final, geometry_slice)
            total_porosity = float(np.mean(arr_final == 0))
            print(f"  After padding: shape={arr_final.shape}")
            print(f"    Porosity (geometry only): {geom_porosity:.4f}")
            if args.pad > 0:
                print(f"    Porosity (including padding): {total_porosity:.4f}")
    else:
        # No padding - geometry is entire array
        geometry_slice = (slice(None), slice(None), slice(None))
    
    return arr_final, geometry_slice


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="Prepare binary geometry for OpenLB interpolated boundary conditions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    p.add_argument("input", help="TIFF directory / glob pattern, or .vti file")
    p.add_argument("output", help="Output .vti file (0=pore, 255=solid)")

    p.add_argument("-t", "--threshold", type=float, default=127,
                   help="Binarisation threshold for grayscale inputs")
    p.add_argument("--solid-dark", action="store_true",
                   help="Solid is darker than pore (solid=raw<=thr)")
    
    p.add_argument("--down", type=int, default=2,
                   help="Downsample factor (1=no downsampling)")
    p.add_argument("--smooth-low-iters", type=int, default=0,
                   help="SDF smoothing passes at low resolution")
    p.add_argument("--smooth-sigma", type=float, default=0.6,
                   help="Gaussian sigma for low-res SDF smoothing")
    p.add_argument("--thicken", type=int, default=0,
                   help="Dilate solid by N voxels (thicken walls, narrow throats)")
    
    p.add_argument("--final-smooth-iters", type=int, default=0,
                   help="SDF smoothing passes AFTER upsampling (removes blocky artifacts)")
    p.add_argument("--final-smooth-sigma", type=float, default=0.5,
                   help="Gaussian sigma for final smoothing (keep low: 0.3-0.7)")
    
    p.add_argument("--upsample-intermediate", type=float, default=None,
                   help="Supersampling: first upsample by this factor (e.g., 3)")
    p.add_argument("--upsample-final", type=float, default=None,
                   help="Supersampling: then downsample to this net factor (e.g., 2). "
                        "Creates smoother upsampling via anti-aliasing.")
    
    p.add_argument("--pad", type=int, default=0,
                   help="Add N voxels of pore padding (positive) or crop N voxels (negative). "
                        "Example: --pad 10 adds padding, --pad -5 crops 5 voxels from edges.")
    p.add_argument("--pad-xy-only", action="store_true",
                   help="Apply padding/cropping only to X and Y faces (not Z)")
    
    p.add_argument("--porosity", type=porosity_arg_type, default=None,
                   help="Target porosity [0, 1]. Uses SDF threshold search on raw grayscale. "
                        "Pass 'None' or omit to disable (use --threshold instead).")
    p.add_argument("--sdf-sigma", type=float, default=0.5,
                   help="SDF Gaussian blur sigma (higher = more boundary smearing)")
    p.add_argument("--adjust-porosity-post", action="store_true",
                   help="Fine-tune porosity after processing via post-processing SDF adjustment")
    
    p.add_argument("--allow-boundary-removal", action="store_true",
                   help="Allow removal of boundary voxels")
    
    p.add_argument("--majority-filter", type=int, default=1,
                   help="Remove thin protrusions (iterations). 0=off, 2-3 for aggressive filtering.")
    
    p.add_argument("--min-gradient", type=float, default=0.2,
                   help="Gradient threshold for validation")
    p.add_argument("-v", "--verbose", action="store_true", help="Print detailed progress")

    args = p.parse_args()
    
    # Validation
    if args.porosity is not None and not (0.0 < args.porosity < 1.0):
        sys.exit("ERROR: --porosity must be between 0 and 1 (exclusive)")
    
    if (args.upsample_intermediate is None) != (args.upsample_final is None):
        sys.exit("ERROR: --upsample-intermediate and --upsample-final must be used together")
    if args.upsample_intermediate is not None:
        if args.upsample_intermediate <= 0 or args.upsample_final <= 0:
            sys.exit("ERROR: Upsample factors must be positive")
        if args.upsample_final > args.upsample_intermediate:
            sys.exit("ERROR: --upsample-final must be ≤ --upsample-intermediate")
        if args.upsample_final < 1.0:
            sys.exit("ERROR: --upsample-final must be ≥ 1.0")

    solid_is_high = not args.solid_dark
    
    print("="*70)
    print("OLB GEOMETRY FILTER (Working SDF Porosity Control)")
    print("="*70)

    if args.input.lower().endswith(".vti"):
        print(f"Loading VTI: {args.input}")
        raw, spacing, origin = load_vti_raw(args.input)
    else:
        print(f"Loading TIFF series from: {args.input}")
        raw, spacing, origin = load_tiff_series_raw(args.input)

    print(f"  Raw shape : {raw.shape}")
    print(f"  Raw range : [{float(np.min(raw)):.3f}, {float(np.max(raw)):.3f}]")
    
    original_shape = raw.shape
    original_porosity = None
    
    print("\nDetermining threshold...")
    
    # ── Determine binarization threshold ───────────────────────────────────
    
    if looks_binary_0_255(raw):
        # Binary input
        if float(np.max(raw)) == 1.0:
            arr = (raw * 255.0).astype(np.uint8)
        else:
            arr = raw.astype(np.uint8)
        print("  Detected binary input (0/255) — skipping thresholding.")
        original_porosity = calculate_porosity(arr)
        geometry_slice = (slice(None), slice(None), slice(None))
    else:
        # Grayscale input
        if args.porosity is not None:
            # Use SDF-based porosity targeting on raw grayscale
            arr = adjust_porosity_sdf(
                raw, args.porosity, sdf_sigma=args.sdf_sigma, 
                solid_is_high=solid_is_high, verbose=args.verbose
            )
            original_porosity = calculate_porosity(arr)
            print(f"  Initial binarized porosity: {original_porosity:.6f}")
            geometry_slice = (slice(None), slice(None), slice(None))
        else:
            # Manual threshold
            if args.verbose:
                print(f"  Using manual threshold: {args.threshold:.6f}")
            arr = binarize_from_threshold(raw, args.threshold, solid_is_high=solid_is_high)
            original_porosity = calculate_porosity(arr)
            print(f"  Binarised porosity: {original_porosity:.6f}")
            geometry_slice = (slice(None), slice(None), slice(None))
    
    # ── Process geometry ───────────────────────────────────────────────────
    
    print(f"\n" + "="*70)
    print("PROCESSING")
    print("="*70 + "\n")
    
    arr_final, geometry_slice = process_geometry(arr, args, original_shape)
    
    # Optional post-processing porosity adjustment
    if args.adjust_porosity_post and args.porosity is not None:
        print(f"\n" + "="*70)
        print("POST-PROCESSING POROSITY ADJUSTMENT")
        print("="*70)
        arr_final = adjust_porosity_post(arr_final, args.porosity, 
                                        geometry_slice=geometry_slice, 
                                        verbose=args.verbose)
    
    # ── Validate ───────────────────────────────────────────────────────────
    
    print(f"\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    print("Validation:")
    report_validation(arr_final, min_gradient=args.min_gradient, geometry_slice=geometry_slice)
    
    # ── Write ──────────────────────────────────────────────────────────────
    
    print(f"\n" + "="*70)
    print("OUTPUT")
    print("="*70)
    
    print(f"\nWriting: {args.output}")
    write_vti(args.output, arr_final, spacing=spacing, origin=origin)
    
    # Write actual porosity to file (for AiiDA workflow)
    # IMPORTANT: Use geometry region only (excluding padding)
    final_porosity = calculate_porosity(arr_final, geometry_slice)
    porosity_file = "actual_porosity.txt"
    with open(porosity_file, 'w') as f:
        f.write(f"{final_porosity:.6f}\n")
    print(f"Writing: {porosity_file}")
    print(f"  Porosity (geometry only): {final_porosity:.6f}")
    if geometry_slice != (slice(None), slice(None), slice(None)):
        print(f"  Note: Padding excluded from porosity calculation")
    
    print("Done.")

    print("\n" + "="*70)
    if args.porosity is not None:
        print(f"Target porosity (requested)         : {args.porosity:.6f}")
    print(f"Initial porosity (binarised)        : {original_porosity:.6f}")
    print(f"Final porosity (after processing)   : {final_porosity:.6f}")
    print(f"Output shape                        : {arr_final.shape}")
    print("="*70)
    
    if args.porosity is not None:
        final_error = abs(final_porosity - args.porosity)
        if final_error > 0.05:
            print("\n⚠️  Final porosity differs from target.")
            print("    This is expected when processing (smoothing, filtering)")
            print("    substantially changes geometry.")
            if not args.adjust_porosity_post:
                print("    Try: --adjust-porosity-post for fine-tuning after processing")


if __name__ == "__main__":
    main()
