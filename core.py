"""
JWST NIRSpec Kernel Phase Interferometry - Core Functions

This module provides core functionality for processing JWST pupil data and 
extracting kernel phase measurements from astronomical images.

Functions:
    - load_pupil_data: Load JWST pupil models from FITS files
    - create_discrete_model: Create discrete sampling model from pupil data
    - initialize_kpo: Initialize the Kernel Phase Object
    - load_cube_data: Load and extract data from IFS cubes
    - extract_kernel_phases: Extract kernel phase measurements
    - get_kpo_matrices: Extract matrices from KPO objects

Dependencies:
    - xara: Core kernel phase interferometry library
    - numpy: Numerical computations
    - fitsio: FITS file I/O for astronomical data
"""

import numpy as np
import fitsio
import xara
from typing import Tuple


def load_pupil_data(pupil_file: str) -> Tuple[np.ndarray, float]:
    """
    Load JWST pupil model from FITS file.
    
    Args:
        pupil_file (str): Path to the JWST pupil FITS file
        
    Returns:
        tuple: (pupil_array, pupil_pixel_size) where pupil_array is the 2D pupil
               transmission map and pupil_pixel_size is in meters per pixel
        
    Raises:
        FileNotFoundError: If the pupil file cannot be found
        IOError: If there's an error reading the FITS file
    """
    try:
        pupil = fitsio.read(pupil_file)
        pupil_diameter = 6.5  # meters
        pupil_pix_size = pupil_diameter / float(pupil.shape[0])
        print(f"Loaded pupil data with shape: {pupil.shape}")
        print(f"Pupil pixel size: {pupil_pix_size:.6f} m/pixel")
        return pupil, pupil_pix_size
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find pupil file: {pupil_file}") from e
    except Exception as e:
        raise IOError(f"Error reading pupil file: {e}") from e


def create_discrete_model(pupil: np.ndarray, pupil_pix_size: float, 
                         sample_factor: int = 30) -> np.ndarray:
    """
    Create a discrete model from pupil data.
    
    Args:
        pupil (np.ndarray): 2D pupil transmission array
        pupil_pix_size (float): Pupil pixel size in meters per pixel
        sample_factor (int): Sampling factor for the discrete model (default: 30)
        
    Returns:
        np.ndarray: Discrete model coordinates array of shape (N_points, 2)
    """
    sample_interval = sample_factor * pupil_pix_size
    model = xara.core.create_discrete_model(
        pupil, pupil_pix_size, sample_interval, binary=False
    )
    print(f"Created discrete model with {model.shape[0]} points")
    return model


def initialize_kpo(model: np.ndarray, bmax: float = 6.5) -> xara.KPO:
    """
    Initialize the Kernel Phase Object (KPO).
    
    Args:
        model (np.ndarray): Discrete model coordinates from create_discrete_model()
        bmax (float): Maximum baseline length in meters (default: 6.5)
        
    Returns:
        xara.KPO: The initialized KPO object
    """
    kpo = xara.KPO(array=model, bmax=bmax)
    print("KPO initialized successfully")
    return kpo


def load_cube_data(cube_file: str, wavelength_key: str = "WAVELN13", 
                   slice_index: int = 13) -> Tuple[np.ndarray, float]:
    """
    Load data cube and extract image slice and wavelength.
    
    Args:
        cube_file (str): Path to the data cube FITS file
        wavelength_key (str): Header keyword for wavelength (default: "WAVELN13")
        slice_index (int): Index of the slice to extract (default: 13)
        
    Returns:
        tuple: (image_slice, wavelength) where image_slice is 2D array and 
               wavelength is float
               
    Raises:
        FileNotFoundError: If the cube file cannot be found
        KeyError: If the wavelength key is not found in header
        IndexError: If the slice index is out of bounds
    """
    try:
        with fitsio.FITS(cube_file) as f:
            header = f[0].read_header()
            wavelength = header[wavelength_key]
            
            # Check if slice_index is valid
            cube_shape = f[0].get_dims()
            if slice_index >= cube_shape[0]:
                raise IndexError(f"Slice index {slice_index} out of bounds for cube with {cube_shape[0]} slices")
            
            image = f[0][slice_index, :, :]
            print(f"Loaded image slice {slice_index} with shape: {image.shape}")
            print(f"Wavelength: {wavelength}")
            return image, wavelength
            
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Cannot find cube file: {cube_file}") from e
    except KeyError as e:
        raise KeyError(f"Wavelength key '{wavelength_key}' not found in header") from e


def extract_kernel_phases(kpo: xara.KPO, image: np.ndarray, wavelength: float, 
                         pixel_size: float = 100, recenter: bool = False) -> np.ndarray:
    """
    Extract kernel phase data from a single image.
    
    Args:
        kpo (xara.KPO): Initialized KPO object
        image (np.ndarray): 2D image array
        wavelength (float): Wavelength of the observation
        pixel_size (float): Pixel size in mas (default: 100)
        recenter (bool): Whether to recenter the image (default: False)
        
    Returns:
        np.ndarray: Extracted kernel phase data of shape (1, N_kernel_phases)
    """
    kpo.extract_KPD_single_cube(image, pixel_size, wavelength, recenter=recenter)
    print(f"Extracted kernel phases with shape: {kpo.KPDT[0].shape}")
    return kpo.KPDT[0]


def get_kpo_matrices(kpo: xara.KPO) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract kernel phase matrix and UV coordinates from KPO object.
    
    Args:
        kpo (xara.KPO): Initialized KPO object
        
    Returns:
        tuple: (kpm, uv_sel) where kpm is the kernel phase matrix and 
               uv_sel is the UV coordinate selection
    """
    kpm = kpo.kpi.KPM
    uv_sel = kpo.kpi.UVC
    return kpm, uv_sel