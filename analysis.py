"""
JWST NIRSpec Kernel Phase Interferometry - Analysis Functions

This module provides analysis functionality for companion detection and contrast
curve generation using chi-squared fitting methods.

Functions:
    - grid_chisq: Perform chi-squared fitting for companion detection
    - create_search_grids: Set up parameter grids for searches

Dependencies:
    - xara: Core kernel phase interferometry library
    - numpy: Numerical computations
"""

import numpy as np
import xara
from typing import Tuple, List


def grid_chisq(grid_sep: np.ndarray, grid_pa: np.ndarray, grid_contrast: np.ndarray,
               kernels: np.ndarray, sci_parang: List[float], kpm: np.ndarray,
               uv_sel: np.ndarray, waves: float, verbose: bool = False) -> Tuple[np.ndarray, List]:
    """
    Produce a chi-squared fit curve to the data for companion detection.
    
    This function uses the chi-squared method to produce a contrast curve by
    testing different combinations of separation, position angle, and contrast
    values against observed kernel phase data.
    
    Args:
        grid_sep (np.ndarray): Grid of separation values to test (mas)
        grid_pa (np.ndarray): Grid of position angle values to test (degrees)
        grid_contrast (np.ndarray): Grid of contrast values to test (linear)
        kernels (np.ndarray): Calibrated kernel phase data 
        sci_parang (List[float]): Array of parallactic angles (degrees)
        kpm (np.ndarray): Kernel phase matrix
        uv_sel (np.ndarray): Points to sample in the UV-plane
        waves (float): Wavelength band in question (nm)
        verbose (bool): Print progress information (default: False)
    
    Returns:
        tuple: (significance, keep_fits) where:
            - significance (np.ndarray): Significance values for each parameter combination
            - keep_fits (List): Diagnostic information containing example test binaries
    
    Note:
        This function requires the `parang_cvis_binary` function from the xara library.
        If not available, a warning will be printed and placeholder values used.
    """
    # Fixed constants
    num_kps = kernels.shape[1] if kernels.ndim > 1 else len(kernels)
    
    # Variables to store the grid KPs and corresponding params
    result = np.zeros((grid_contrast.shape[0], grid_sep.shape[0], grid_pa.shape[0]))
    total = len(grid_contrast) * len(grid_pa) * len(grid_sep)
    count = 0
    
    # Create a list to keep the diagnostic info
    keep_fits = [kernels]
    
    # Iterate through each contrast, separation and position angle
    for i, contrast in enumerate(grid_contrast):
        for j, sep in enumerate(grid_sep):
            for k, pa in enumerate(grid_pa):
                
                # Create the KPs for the test binary
                try:
                    # Try to use parang_cvis_binary from xara
                    if hasattr(xara, 'parang_cvis_binary'):
                        test_binary = xara.parang_cvis_binary(
                            [sep, pa, contrast], kpm, uv_sel[:, 0], uv_sel[:, 1],
                            waves, sci_parang, num_kps=kpm.shape[0]
                        )
                    else:
                        # Check if it's in a submodule
                        test_binary = getattr(xara, 'parang_cvis_binary', None)
                        if test_binary is None:
                            raise AttributeError("parang_cvis_binary not found")
                        test_binary = test_binary(
                            [sep, pa, contrast], kpm, uv_sel[:, 0], uv_sel[:, 1],
                            waves, sci_parang, num_kps=kpm.shape[0]
                        )
                except (AttributeError, NameError):
                    # Placeholder implementation if function is not available
                    if count == 0:  # Only print warning once
                        print("Warning: parang_cvis_binary function not found in xara. Using placeholder.")
                    test_binary = np.zeros_like(kernels)
                
                # Measure the chi-squared w.r.t the observations
                ratio = (kernels - test_binary) ** 2
                
                # Sum up the differences into a single chi-squared
                chi_sq = np.sum(ratio)
                
                result[i, j, k] = chi_sq
                
                # Keep track of progress
                if verbose and count % 250 == 0:
                    print(f"{count}/{total}")
                    
                    # Add a test binary to the list
                    keep_fits.append(([sep, pa, contrast], test_binary))
                
                count += 1
    
    # Compute the null model (no companion)
    null_model = np.sum(kernels ** 2)
    
    # Diagnostic info
    print("Diagnostics:")
    print(f"Null: {null_model}")
    print(f"Min result: {np.min(result)}")
    print(f"Max result: {np.max(result)}")
    
    # Scale the chi^2 so best fit = num degrees of freedom
    degrees_of_freedom = kernels.ravel().shape[0] - 3.0
    min_result = np.min(result)
    result = result * (degrees_of_freedom / min_result)
    
    significance = result - (null_model * degrees_of_freedom / min_result)
    
    print(f"Min sig: {np.min(significance)}")
    print(f"Max sig: {np.max(significance)}")
    
    return significance, keep_fits


def create_search_grids(sep_max: float = 300.0, sep_step: float = 5.0,
                       pa_step: float = 15.0, mag_min: float = 0.0, 
                       mag_max: float = 10.0, mag_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create complete parameter grids for companion search.
    
    Args:
        sep_max (float): Maximum separation in mas (default: 300.0)
        sep_step (float): Separation step size in mas (default: 5.0)
        pa_step (float): Position angle step in degrees (default: 15.0)
        mag_min (float): Minimum magnitude difference (default: 0.0)
        mag_max (float): Maximum magnitude difference (default: 10.0)
        mag_steps (int): Number of magnitude steps (default: 100)
    
    Returns:
        tuple: (grid_sep, grid_pa, grid_contrast, grid_mags)
    """
    grid_sep = np.arange(0.0, sep_max, sep_step)
    grid_pa = np.arange(0.0, 360.0, pa_step)
    grid_mags = np.arange(mag_min, mag_max, (mag_max - mag_min) / mag_steps)
    grid_contrast = 10 ** (grid_mags / 2.5)
    
    return grid_sep, grid_pa, grid_contrast, grid_mags