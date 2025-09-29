# JWST NIRSpec Kernel Phase Interferometry Pipeline

A comprehensive Python pipeline for processing JWST NIRSpec Integral Field Spectroscopy (IFS) data to extract kernel phase measurements for detecting companions and performing high-precision astrometry.

## Overview

This pipeline implements kernel phase interferometry techniques specifically adapted for JWST NIRSpec IFS observations. Kernel phases are closure phases that are immune to atmospheric and instrumental phase errors, making them ideal for detecting faint companions around bright stars.

The pipeline processes JWST NIRSpec data cubes through several stages:
1. **Data Loading**: Flexible FITS file handling with metadata extraction
2. **Preprocessing**: Image centering, background subtraction, and quality control
3. **Kernel Phase Extraction**: Generate kernel phase measurements using the XARA library
4. **Companion Analysis**: Chi-squared fitting for companion detection
5. **Visualization**: Contrast curves and significance maps

## Installation

### Prerequisites

- Python 3.8+
- Required packages:
  ```bash
  pip install numpy matplotlib fitsio scipy
  ```

### XARA Library

The pipeline requires the XARA (eXtreme Angular Resolution Astronomy) library:
```bash
pip install xara
```

## Module Structure

The pipeline is organized into focused modules:

### `core.py`
Core functionality for data loading and kernel phase extraction:
- `load_pupil_data()` - Load JWST pupil models
- `create_discrete_model()` - Create discrete sampling models
- `initialize_kpo()` - Initialize Kernel Phase Objects
- `load_cube_data()` - Load NIRSpec IFS data cubes
- `extract_kernel_phases()` - Extract kernel phase measurements
- `get_kpo_matrices()` - Extract matrices from KPO objects

### `analysis.py`
Analysis functions for companion detection:
- `grid_chisq()` - Chi-squared fitting for parameter estimation
- `create_search_grids()` - Set up parameter grids for searches

### `visualization.py`
Plotting and visualization functions:
- `plot_contrast_curve()` - Generate detection contrast curves
- `visualize_pupil_model()` - Show pupil with sampling points
- `visualize_kernel_phases()` - Plot extracted kernel phases

### `cube.py`
NIRSpec IFS data cube preprocessing:
- `DataCube` class for handling NIRSpec IFS data
- Image centering and alignment
- Background subtraction
- Quality control and preprocessing

## Quick Start

### Basic Kernel Phase Extraction

```python
from nirspec_KPI import core, visualization

# Load JWST pupil model
pupil, pupil_pix_size = core.load_pupil_data("jwst_pupil_RevW_npix1024.fits")

# Create discrete model
model = core.create_discrete_model(pupil, pupil_pix_size)

# Initialize KPO
kpo = core.initialize_kpo(model)

# Load and process data cube
image, wavelength = core.load_cube_data("nirspec_cube.fits")
kernel_phases = core.extract_kernel_phases(kpo, image, wavelength)

# Visualize results
visualization.visualize_pupil_model(pupil, model, pupil_pix_size)
visualization.visualize_kernel_phases(kernel_phases)
```

### Companion Detection Analysis

```python
from nirspec_KPI import core, analysis, visualization

# ... (previous setup steps) ...

# Create search parameter grids
grid_sep, grid_pa, grid_contrast, grid_mags = analysis.create_search_grids(
    sep_max=300.0,    # Maximum separation (mas)
    sep_step=5.0,     # Separation step size (mas) 
    pa_step=15.0,     # Position angle step (degrees)
    mag_max=10.0      # Maximum contrast (magnitudes)
)

# Get KPO matrices
kpm, uv_sel = core.get_kpo_matrices(kpo)

# Perform companion search
parallactic_angles = [90.0]  # Example PA
significance, diagnostics = analysis.grid_chisq(
    grid_sep, grid_pa, grid_contrast, kernel_phases,
    parallactic_angles, kpm, uv_sel, wavelength
)

# Plot contrast curve
visualization.plot_contrast_curve(significance, grid_sep, grid_mags)
```

### NIRSpec Data Cube Preprocessing

```python
from nirspec_KPI.cube import DataCube

# Load NIRSpec IFS data from directory
cube = DataCube(dir_="/path/to/nirspec/data", verbose=True)

# Preprocessing steps
background_values = cube.median_subtract()
x0, y0 = cube.find_center(method="BCEN", hwhm=30.0)
fine_x0, fine_y0 = cube.rough_center(frame_center=False)

# Access processed data
processed_data = cube.data
parallactic_angles = cube.parang
wavelengths = cube.wavelengths
```

## Data Requirements

### JWST Pupil Models
- Standard JWST pupil transmission files (e.g., `jwst_pupil_RevW_npix1024.fits`)
- Available from the JWST documentation or generated using WebbPSF

### NIRSpec IFS Data
- Calibrated NIRSpec IFS data cubes in FITS format
- Data should be in standard JWST pipeline output format
- Supports various FITS extensions: 'SCI', 'DATA', or numeric extensions

### Expected FITS Structure
The pipeline automatically detects and handles:
- **Data Extensions**: SCI, DATA, or fallback to extension [1]
- **Header Keywords**: PA_APER, ROLL_REF, V3_ANGLE, EXPSTART, etc.
- **Wavelength Information**: From dedicated extensions or WCS headers

## Key Features

### Adaptive Processing
- **Dynamic image sizing**: Automatically adapts to different NIRSpec detector formats
- **Flexible centroiding**: Adaptive windowing based on image dimensions
- **Robust error handling**: Graceful failure modes with informative diagnostics

### JWST Integration
- **Native FITS handling**: Direct support for JWST pipeline outputs
- **Metadata extraction**: Automatic extraction of observation parameters
- **Wavelength support**: Full spectral dimension handling

### Kernel Phase Analysis
- **High-precision measurements**: Sub-milliarcsecond astrometric precision
- **Calibration-free**: Immune to atmospheric and instrumental phase errors
- **Companion detection**: Optimized for detecting faint companions

## Output Products

### Kernel Phase Data
- Extracted kernel phase measurements for each spectral channel
- Associated uncertainties and quality metrics
- Metadata including observation parameters

### Companion Analysis
- **Significance maps**: 3D significance arrays (separation, position angle, contrast)
- **Contrast curves**: Detection limits as a function of separation
- **Best-fit parameters**: Companion properties if detected

### Diagnostic Plots
- Pupil model visualization with sampling points
- Kernel phase measurements
- Contrast curves with confidence levels
- Data quality assessments

## Contributing

This pipeline is designed for research applications in high-contrast imaging and astrometry. For questions, bug reports, or contributions, please refer to the repository issues.

## References

- Martinache, F. (2010). "Kernel Phase in Fizeau Interferometry" *ApJ*, 724, 464
- Ireland, M. J. (2013). "Kernel phase and polarimetry" *MNRAS*, 433, 1718
- JWST NIRSpec Documentation: https://jwst-docs.stsci.edu/nirspec

## License

See LICENSE file for details.
