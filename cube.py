#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""JWST NIRSpec Datacube Manipulation

This module contains functionality for common manipulations needed for JWST NIRSpec
IFS data cubes prior to calculation of the kernel phases. This includes operations 
like windowing, centering data, background subtraction, and extracting relevant 
metadata from FITS headers.

The DataCube class is designed to handle JWST NIRSpec IFS data cubes and provides
methods for preprocessing the data for kernel phase interferometry analysis.

Key Features:
    - Load and stack NIRSpec IFS data cubes
    - Extract observation metadata (parallactic angles, times, wavelengths)
    - Image centering and alignment
    - Background subtraction
    - Data quality preprocessing

Example:
    >>> cube = DataCube(dir_="/path/to/nirspec/data")
    >>> cube.median_subtract()
    >>> cube.find_center()
    >>> cube.rough_center()

Dependencies:
    - numpy: Numerical computations
    - fitsio: FITS file I/O
    - xara.core: Centroiding algorithms
    - scipy.optimize: Optimization routines
"""

import numpy as np 
import fitsio
# Optional matplotlib import for debugging
# import matplotlib.pyplot as plt 

from xara.core import super_gauss0, determine_origin
from os.path import join 
from glob import glob 
from numpy.fft import fft2, fftshift
from scipy.optimize import minimize
from typing import Tuple, Optional, List, Union

def create_circular_mask(h: int, w: int, center: Optional[Tuple[float, float]] = None, 
                        radius: Optional[float] = None) -> np.ndarray:
    """
    Create a circular mask for a given height and width.

    Parameters
    ----------
    h : int
        Image height.
    w : int
        Image width.
    center : tuple of int, optional
        The (x, y) center of the circle. If None, the image center is used.
    radius : int, optional
        The radius of the circle. If None, defaults to the minimum distance from the center to an edge.

    Returns
    -------
    np.ndarray of bool
        A boolean mask with True inside the circle.
    """
    
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    return mask

def min_func(delta: np.ndarray, fft_image: np.ndarray, mask: np.ndarray, 
             uvc: np.ndarray, imsize: int) -> float:
    """
    Objective function for phase correction optimization.

    Parameters
    ----------
    delta : array_like
        Initial offset guess, e.g. [dx, dy].
    fft_image : np.ndarray
        The FFT of the image computed via fftshift(fft2(fftshift(image))).
    mask : np.ndarray of bool
        Circular mask to apply during minimization.
    uvc : np.ndarray
        Array of UV coordinates.
    imsize : int
        The size of the image (assumed square).

    Returns
    -------
    float
        The residual value after applying a phase correction.
    """
    
    # Create the correction array and mask out the zeros 
    correction = np.exp(1j * 2 * np.pi * uvc.dot(delta / imsize))
    correction = np.reshape(correction, (imsize, imsize))
    
    # Apply the correction 
    temp = fft_image.copy() * correction
    fft_angle = np.angle(temp) * mask 

    # Compute the residual value 
    residual = np.sum(np.abs(fft_angle))  
        
    return residual 

def find_min_clip(image: np.ndarray, lim: float = 0.1, negative: bool = False, 
                  symmetric: bool = True) -> Tuple[int, int, int, int]:
    """
    Determine the minimum boundaries to which an image can be clipped without loss of data.

    Parameters
    ----------
    image : np.ndarray
        Input image array.
    lim : float, optional
        Threshold value to decide clipping (default is 0.1).
    negative : bool, optional
        If True, considers abs(image) < lim; otherwise, uses image < lim (default is False).
    symmetric : bool, optional
        If True, returns square boundaries; otherwise, returns rectangular boundaries (default is True).

    Returns
    -------
    tuple of int
        A tuple (xl, yl, xh, yh) representing the lower and upper boundaries.
    """
    
    image_size = image.shape[0]
    center = image_size / 2.0 

    xl, yl, xh, yh = -1, -1, -1, -1

    # Mask out everything in the image 
    if negative:
        image[np.abs(image) < lim] = np.nan
    else:
        image[image < lim] = np.nan 

    # Slide through the image left-to-right
    for slider in range(image_size): 

        x_slice = image[slider, :]
        y_slice = image[:, slider]

        num_x_nans = np.sum(np.isnan(x_slice))
        num_y_nans = np.sum(np.isnan(y_slice)) 

        if not(num_x_nans == image_size) and (xl==-1):
            xl = slider 
        if not(num_y_nans == image_size) and (yl==-1): 
            yl = slider

    # Slide through the image right-to-left
    for slider in reversed(range(image_size)): 

        x_slice = image[slider, :]
        y_slice = image[:, slider]

        num_x_nans = np.sum(np.isnan(x_slice))
        num_y_nans = np.sum(np.isnan(y_slice)) 

        if not(num_x_nans == image_size) and (xh==-1):
            xh = slider 
        if not(num_y_nans == image_size) and (yh==-1): 
            yh = slider

    # Even out the borders if symmetric is set
    if symmetric: 
        
        #THIS CODE IS CLUNKY 
        #Replace it by considering distance from image_size
        #This is much easier to do since xl is the distnace
        #from 0
        xl_dist = center-xl; xh_dist = xh-center
        yl_dist = center-yl; yh_dist = yh-center 

        delta_x = xh_dist-xl_dist
        delta_y = yh_dist-yl_dist

        if delta_x > 0: 
            xh -= delta_x 
            xh = int(np.round(xh)) 
        else:
            xl -= delta_x
            xl = int(np.round(xl)) 

        if delta_y > 0:
            yh -= delta_y
            yh = int(np.round(yh)) 
        else:
            yl -= delta_y 
            yl = int(np.round(yl))
    
    return xl, yl, xh, yh

class DataCube(): 
    """Class to handle JWST NIRSpec IFS data cubes for kernel phase interferometry.

    This class provides functionality for preprocessing JWST NIRSpec Integral Field
    Spectroscopy data cubes. It can read series of FITS files and stack them into
    a 4D data structure, extract relevant observation metadata from headers, and
    perform common preprocessing operations required for kernel phase analysis.

    The class supports both reading data from directories of FITS files and
    accepting pre-loaded data arrays for flexibility in different workflows.

    Parameters
    ----------
    dir_ : str, optional
        Directory containing NIRSpec IFS FITS files (default: "")
    extension : str, optional
        File extension for FITS files (default: ".fits")
    idata : np.ndarray, optional
        Pre-loaded 4D data cube (nframes, nspec_bins, y, x) (default: None)
    iparang : array_like, optional
        Pre-loaded parallactic angles for each frame (default: None)  
    itime : array_like, optional
        Pre-loaded observation times for each frame (default: None)
    verbose : bool, optional
        Enable verbose output during processing (default: True)

    Attributes
    ----------
    data : np.ndarray
        4D data cube with shape (nframes, nspec_bins, y, x)
    parang : np.ndarray
        Parallactic angles for each frame (degrees)
    time : np.ndarray
        Observation times for each frame
    wavelengths : np.ndarray
        Wavelength array for spectral bins (if available)
    x0, y0 : np.ndarray
        Measured centroid positions for each frame and spectral bin
    frame_x0, frame_y0 : np.ndarray
        Average centroid positions per frame (averaged over spectral bins)
    """

    def __init__(self, dir_="", extension=".fits", idata=None, iparang=None, itime=None, verbose=True):
        """
        Initialize the DataCube object.

        Parameters
        ----------
        dir_ : str, optional
            Directory containing the reduced data files.
        extension : str, optional
            File extension for fits files (default is ".fits").
        idata : np.ndarray or None, optional
            Preloaded data cube. If provided, the directory is ignored.
        iparang : array_like or None, optional
            Parallactic angles corresponding to the data.
        itime : array_like or None, optional
            Observation times corresponding to the data.
        verbose : bool, optional
            If True, prints information during initialization.

        Notes
        -----
        If idata is None, the DataCube is built by reading fits files from dir_.
        """
 
        if idata is None:
            data, parang, time, wavelengths = [], [], [], []

            # Build list of NIRSpec science images
            fits = sorted(glob(join(dir_, "*" + extension))) 
            if verbose: 
                print(f"Found {len(fits)} NIRSpec IFS files in {dir_}")

            # Store filenames for reference
            self._filename = []

            for file_ in fits:
                self._filename.append(file_)

                try:
                    with fitsio.FITS(file_) as f:
                        # NIRSpec IFS data is typically in the 'SCI' extension
                        # Check available extensions first
                        if verbose and len(data) == 0:  # Print info for first file only
                            print(f"Available extensions in {file_}:")
                            for i, ext in enumerate(f):
                                try:
                                    ext_name = ext.get_extname() if hasattr(ext, 'get_extname') else f"EXT_{i}"
                                    print(f"  [{i}]: {ext_name}")
                                except:
                                    print(f"  [{i}]: Unknown extension")

                        # Try to load data from SCI extension, fallback to extension 1
                        cube_loaded = False
                        for ext_name in ['SCI', 1, 'DATA']:
                            try:
                                if isinstance(ext_name, str):
                                    cube = f[ext_name][:, :, :]
                                else:
                                    cube = f[ext_name][:, :, :]
                                cube_loaded = True
                                if verbose and len(data) == 0:
                                    print(f"Loaded data from extension: {ext_name}")
                                    print(f"Data cube shape: {cube.shape}")
                                break
                            except (KeyError, IndexError, ValueError):
                                continue
                        
                        if not cube_loaded:
                            raise ValueError(f"Could not find suitable data extension in {file_}")

                        # Extract header information for JWST/NIRSpec
                        header = f[0].read_header()  # Primary header
                        
                        # Extract parallactic angle (JWST uses PA_APER or similar)
                        pa = self._extract_parallactic_angle(header)
                        parang.append(pa)
                        
                        # Extract observation time
                        obs_time = self._extract_observation_time(header)
                        time.append(obs_time)
                        
                        # Try to extract wavelength information
                        if len(wavelengths) == 0:  # Only do this for the first file
                            wl_array = self._extract_wavelengths(f, cube.shape[0])
                            wavelengths = wl_array

                        data.append(cube)

                except Exception as e:
                    print(f"Error loading {file_}: {e}")
                    continue

            if not data:
                raise ValueError(f"No valid data cubes loaded from {dir_}")

            # Create numpy arrays
            data = np.stack(data, axis=0)
            parang = np.array(parang) if parang else np.array([])
            time = np.array(time) if time else np.array([])
            wavelengths = np.array(wavelengths) if len(wavelengths) > 0 else np.array([]) 
        
        else:
            data = idata
            time = itime if itime is not None else []
            parang = iparang if iparang is not None else []
            wavelengths = np.array([])

        if verbose: 
            print(f"Final data cube shape: {data.shape}")
            if len(parang) > 0:
                print(f"Parallactic angles: {len(parang)} values, range {np.min(parang):.1f} to {np.max(parang):.1f} deg")

        # Save the data cube and metadata
        self.data = data 
        self.parang = parang
        self.time = time 
        self.wavelengths = wavelengths
        self.x0 = None
        self.y0 = None

    def _extract_parallactic_angle(self, header):
        """Extract parallactic angle from JWST header keywords."""
        # JWST uses different keywords for parallactic angle
        pa_keywords = ['PA_APER', 'ROLL_REF', 'V3_ANGLE', 'PA_V3']
        
        for keyword in pa_keywords:
            try:
                return float(header[keyword])
            except (KeyError, ValueError, TypeError):
                continue
        
        # If no standard PA keyword found, return 0 and warn
        print("Warning: No parallactic angle found in header, using 0.0")
        return 0.0
    
    def _extract_observation_time(self, header):
        """Extract observation time from JWST header keywords."""
        # Try different time keywords
        time_keywords = ['EXPSTART', 'DATE-OBS', 'TIME-OBS', 'MJD-OBS']
        
        for keyword in time_keywords:
            try:
                return header[keyword]
            except KeyError:
                continue
                
        print("Warning: No observation time found in header")
        return 0.0
    
    def _extract_wavelengths(self, fits_file, n_spectral_bins):
        """Extract wavelength information from NIRSpec FITS file."""
        wavelengths = []
        
        # Try to get wavelengths from different possible locations
        try:
            # Method 1: Look for wavelength extension
            if 'WAVELENGTH' in [ext.get_extname() for ext in fits_file if hasattr(ext, 'get_extname')]:
                wavelengths = fits_file['WAVELENGTH'][:]
            # Method 2: Look for wavelength table
            elif 'EXTRACT1D' in [ext.get_extname() for ext in fits_file if hasattr(ext, 'get_extname')]:
                wavelengths = fits_file['EXTRACT1D']['WAVELENGTH'][:]
            # Method 3: Try to reconstruct from header keywords
            else:
                header = fits_file[0].read_header()
                # Look for standard WCS keywords
                if all(key in header for key in ['CRVAL3', 'CDELT3']):
                    crval = header['CRVAL3']  # Reference wavelength
                    cdelt = header['CDELT3']  # Wavelength step
                    crpix = header.get('CRPIX3', 1)  # Reference pixel
                    wavelengths = crval + (np.arange(n_spectral_bins) - (crpix - 1)) * cdelt
                else:
                    print("Warning: Could not extract wavelength information")
                    wavelengths = np.arange(n_spectral_bins)  # Fallback to indices
                    
        except Exception as e:
            print(f"Warning: Error extracting wavelengths: {e}")
            wavelengths = np.arange(n_spectral_bins)  # Fallback to indices
            
        return np.array(wavelengths) 

    def median_subtract(self, verbose=False):
        """
        Subtract the median background value from each image in the data cube.

        Parameters
        ----------
        verbose : bool, optional
            If True, prints additional information. Default is False.

        Returns
        -------
        np.ndarray
            A 2D array of background values subtracted from each frame and spectral bin.
        """
        
        # Calculate the background value 
        mask = self.data == 0 
        self.data[mask] = np.nan
  
        background_values = np.nanmedian(self.data, axis=(2, 3))
        
        # Re-shape array so it can easily be subtracted 
        background_values = np.expand_dims(background_values, axis=(2, 3)) 
       
        # Apply background correction
        self.data = np.subtract(self.data, background_values) 
        self.data[mask] = 0 

        return background_values
 
    def find_center(self, method="BCEN", hwhm=30.0, verbose=False):
        """
        Center the images to the nearest pixel.

        Uses either the XARA 'BCEN' algorithm or a centroid method to compute the image center
        after applying a super-Gaussian window.

        Parameters
        ----------
        method : str
            Centroiding method, one of {"BCEN", "CENT"}.
        hwhm : float, optional
            Half-width at half-maximum for the super-Gaussian function (default is 30.0).
        verbose : bool, optional
            If True, prints additional information during processing.

        Returns
        -------
        tuple of np.ndarray
            Tuple (x0, y0) with the center coordinates for each frame and band.
        """

        # Make a temporary copy of the data which can be modified
        temp = self.data.copy() 

        # Setup some parameters about the image 
        num_frames = temp.shape[0] 
        num_bands = temp.shape[1] 
        image_size = temp.shape[2]  # assuming it's square here! 
        image_center = image_size / 2.0 

        # Check the method variable 
        if method not in ["BCEN", "CENT"]: 
            print("Error: invalid centroid method selected.")
            return None, None 
 
        # Setup the arrays for storing the results 
        x0 = np.zeros((num_frames, num_bands))
        y0 = np.zeros((num_frames, num_bands)) 

        # Mask to apply before centroiding 
        mask = super_gauss0(image_size, image_size, 
                image_center, image_center, hwhm) 
 
        # Find the image centroids 
        for frame in range(num_frames): 

            for band in range(num_bands): 

                # Extract the image 
                image = temp[frame, band, :, :]
                image_size = image.shape[0]
                center = image_size // 2
                
                # Dynamically determine stamp size based on image size
                # Use central quarter of the image for centroiding
                stamp_half_size = min(image_size // 4, 50)  # Max 50 pixels, adaptive
                x_min = max(0, center - stamp_half_size)
                x_max = min(image_size, center + stamp_half_size)
                y_min = max(0, center - stamp_half_size) 
                y_max = min(image_size, center + stamp_half_size)
                
                # Extract stamp for centroiding
                stamp = image[y_min:y_max, x_min:x_max]
                
                # Handle NaNs and bad pixels
                if np.any(np.isnan(stamp)) or np.any(~np.isfinite(stamp)):
                    stamp = stamp.copy()
                    stamp[~np.isfinite(stamp)] = 0.0

                # Find the centroid 
                try:
                    x, y = determine_origin(stamp, algo=method, verbose=False)
                    
                    # Convert back to full image coordinates
                    x0[frame, band] = (x + x_min) - image.shape[1] / 2.0  
                    y0[frame, band] = (y + y_min) - image.shape[0] / 2.0
                except Exception as e:
                    if verbose:
                        print(f"Centroiding failed for frame {frame}, band {band}: {e}")
                    # Use image center as fallback
                    x0[frame, band] = 0.0
                    y0[frame, band] = 0.0

        # Set the center values of the array 
        self.x0 = x0
        self.y0 = y0
        self.frame_x0 = np.mean(self.x0, axis=1) 
        self.frame_y0 = np.mean(self.y0, axis=1) 

        return x0, y0 

    def rough_center(self, frame_center=False):
        """
        Center each image in the data cube to the nearest pixel.

        Shifts each image based on either a common frame center or individual band centers.

        Parameters
        ----------
        frame_center : bool, optional
            If True, use the overall frame center for all bands; if False, use individual centers (default is False).

        Returns
        -------
        tuple of np.ndarray
            A tuple (fine_x0, fine_y0) representing the sub-pixel shifts for each frame and band.
        """
        
        num_frames = self.data.shape[0] 
        num_bands = self.data.shape[1] 
        half_size = self.data.shape[2] // 2.0 
  
        # Iterate through and re-center the frames 
        for frame in range(num_frames): 
            for band in range(num_bands): 
                image = self.data[frame, band, :, :] 
            
                if frame_center: 
                    xcent, ycent = self.frame_x0[frame], self.frame_y0[frame] 
                else:
                    xcent, ycent = self.x0[frame, band], self.y0[frame, band] 

                # Roll the image to center it
                image = np.roll(image, -int(np.round(ycent)), axis=0)
                image = np.roll(image, -int(np.round(xcent)), axis=1) 

                self.data[frame, band, :, :] = image

        # Finally adjust the x0 and y0 positions to take into account the shift
        self.fine_framex0 = self.frame_x0 - np.round(self.frame_x0) 
        self.fine_framey0 = self.frame_y0 - np.round(self.frame_y0) 
        self.fine_x0 = self.x0 - np.round(self.x0) 
        self.fine_y0 = self.y0 - np.round(self.y0) 

        return self.fine_x0, self.fine_y0

    def minimize_center(self, mask_radius=None):
        """
        Optimize the image center using FFT phase minimization.

        Parameters
        ----------
        mask_radius : int, optional
            Radius of the circular mask applied during optimization. 
            If None, defaults to image_size // 5 (adaptive).

        Returns
        -------
        tuple of np.ndarray
            A tuple (x0, y0) of optimized center offsets for each frame and band.
        """

        # Check prerequisites
        if (self.fine_x0 is None or self.fine_y0 is None or 
            np.any(self.fine_x0 == None) or np.any(self.fine_y0 == None)):
            print("ERROR: need to run find_center() and rough_center() first") 
            return None, None

        # Extract parameters
        num_frames = self.data.shape[0] 
        num_bands = self.data.shape[1] 
        image_size = self.data.shape[2] 
        half_size = image_size // 2
        center_coord = image_size / 2.0 - 0.5  # Center coordinate for even-sized arrays
    
        # Set adaptive mask radius
        if mask_radius is None:
            mask_radius = image_size // 5  # Adaptive based on image size
            
        x0, y0 = np.zeros((num_frames, num_bands)), \
                np.zeros((num_frames, num_bands)) 

        # Create circular mask centered on image center
        mask = create_circular_mask(image_size, image_size, 
                                  center=(center_coord, center_coord), 
                                  radius=mask_radius)
 
        # Create coordinate grid for phase correction
        x, y = np.meshgrid(np.arange(image_size), np.arange(image_size), sparse=False)
        uvc = np.vstack((x.flatten(), y.flatten())).T
        uvc = uvc.astype('float')
        uvc -= center_coord  # Center the coordinates

        # Iterate through and re-center the frames 
        for frame in range(num_frames): 
            if frame % 5 == 0: 
                print(f"Processed {frame}/{num_frames}")

            for band in range(num_bands): 
                image = self.data[frame, band, :, :] 
                
                fft_image = fftshift(fft2(fftshift(image))) 

                delta = np.array([self.fine_x0[frame, band], self.fine_y0[frame, band]])
                
                result = minimize(min_func, delta, 
                        args=(fft_image, mask, uvc, image_size), 
                        bounds=((-1.0, 1.0), (-1.0, 1.0)), 
                        method='Nelder-Mead')

                if result.success: 
                    x0[frame, band] = result.x[0] 
                    y0[frame, band] = result.x[1] 
                else:
                    print(f"Failed to converge on frame {frame}, band {band}") 

        self.cent_x0 = x0 
        self.cent_y0 = y0 

        return x0, y0

