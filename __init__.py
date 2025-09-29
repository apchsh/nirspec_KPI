"""
JWST NIRSpec Kernel Phase Interferometry Pipeline

A modular pipeline for processing JWST NIRSpec data to extract kernel phase 
measurements for companion detection and astrometry.

Modules:
    core: Core functionality for data loading and kernel phase extraction
    analysis: Analysis functions for companion detection and contrast curves  
    visualization: Plotting and visualization functions

Example usage:
    >>> from nirspec_KPI import core, analysis, visualization
    >>> 
    >>> # Load data and initialize
    >>> pupil, pupil_pix_size = core.load_pupil_data("pupil.fits")
    >>> model = core.create_discrete_model(pupil, pupil_pix_size)
    >>> kpo = core.initialize_kpo(model)
    >>> 
    >>> # Extract kernel phases
    >>> image, wavelength = core.load_cube_data("cube.fits")
    >>> kp_data = core.extract_kernel_phases(kpo, image, wavelength)
    >>> 
    >>> # Analyze and visualize
    >>> grids = analysis.create_search_grids()
    >>> kmp, uv_sel = core.get_kpo_matrices(kpo)
    >>> significance, _ = analysis.grid_chisq(*grids, kp_data, [90.0], kmp, uv_sel, wavelength)
    >>> visualization.plot_contrast_curve(significance, grids[0], grids[3])
"""

__version__ = "1.0.0"
__author__ = "JWST NIRSpec KPI Pipeline"

# Import all modules for convenience
from . import core
from . import analysis  
from . import visualization

__all__ = ['core', 'analysis', 'visualization']