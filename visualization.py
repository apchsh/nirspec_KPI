"""
JWST NIRSpec Kernel Phase Interferometry - Visualization Functions

This module provides visualization functionality for plotting pupil models,
kernel phase data, and contrast curves.

Functions:
    - plot_contrast_curve: Generate contrast curve plots
    - visualize_pupil_model: Show pupil with sampling points
    - visualize_kernel_phases: Plot extracted kernel phases

Dependencies:
    - matplotlib: Plotting and visualization
    - numpy: Numerical computations
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional


def plot_contrast_curve(significance: np.ndarray, grid_sep: np.ndarray, 
                       grid_mags: np.ndarray, levels: Optional[List[float]] = None,
                       figsize: Tuple[int, int] = (10, 7), dpi: float = 100.0) -> None:
    """
    Plot the contrast curve from significance data.
    
    Args:
        significance (np.ndarray): 3D significance array from grid_chisq
        grid_sep (np.ndarray): Separation grid used in the search
        grid_mags (np.ndarray): Magnitude grid used in the search
        levels (List[float], optional): Contour levels. Defaults to [0,1,2,3,4,5]
        figsize (tuple): Figure size (width, height) in inches (default: (10,7))
        dpi (float): Figure resolution (default: 100.0)
    """
    if levels is None:
        levels = [0, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create coordinate grids
    gsep, gcontrast = np.meshgrid(grid_sep, grid_mags)
    
    # Average over position angle axis
    sig = np.mean(significance, axis=2)
    
    # Setup plot
    plt.rcParams.update({'font.size': 16.0})
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Create contour plot
    contour = plt.contourf(gsep, gcontrast, np.sqrt(sig), levels=levels)
    cbar = plt.colorbar(contour)
    cbar.set_label('Detection Significance (σ)')
    
    plt.xlabel("Separation (mas)")
    plt.ylabel("Contrast (mags)")
    plt.title("Companion Detection Contrast Curve")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def visualize_pupil_model(pupil: np.ndarray, model: np.ndarray, pupil_pix_size: float,
                         figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Visualize the loaded pupil model with discrete model coordinates.
    
    Args:
        pupil (np.ndarray): 2D pupil transmission array
        model (np.ndarray): Discrete model coordinates
        pupil_pix_size (float): Pupil pixel size in meters per pixel
        figsize (tuple): Figure size (width, height) in inches
    """
    plt.figure(figsize=figsize)
    plt.imshow(pupil)
    
    # Convert model coordinates to pixel coordinates
    pupil_coords = model / pupil_pix_size + pupil.shape[0] / 2.0
    plt.scatter(pupil_coords[:, 0], pupil_coords[:, 1], color='r', s=1, alpha=0.7)
    
    plt.colorbar(label='Pupil Transmission')
    plt.title('JWST Pupil Model with Discrete Sample Points')
    plt.xlabel('X (pixels)')
    plt.ylabel('Y (pixels)')
    plt.show()


def visualize_kernel_phases(kernel_phases: np.ndarray, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Visualize the extracted kernel phase data.
    
    Args:
        kernel_phases (np.ndarray): Extracted kernel phase data
        figsize (tuple): Figure size (width, height) in inches
    """
    kp_data = kernel_phases.flatten()
    
    plt.figure(figsize=figsize)
    plt.scatter(range(len(kp_data)), kp_data, alpha=0.7, s=2)
    plt.xlabel('Kernel Phase Index')
    plt.ylabel('Kernel Phase (radians)')
    plt.title(f'Extracted Kernel Phases ({len(kp_data)} measurements)')
    plt.grid(True, alpha=0.3)
    plt.show()