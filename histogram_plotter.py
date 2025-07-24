# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:34:11 2023

@author: isabe
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit

#Seaborn's color palette
sns.set(style="whitegrid")
# Defined palette
color_palette = sns.color_palette("Set2")
second_colour = color_palette[0]
third_colour = color_palette[1]
fourth_colour = color_palette[3]

### GAUSSIAN
# function for Gaussian
def gaussian(x, A, mean, std_dev):
    return A * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))


# FWHM
def calculate_fwhm(x, y):
    half_max_y = max(y) / 2
    x_left = x[np.where(y >= half_max_y)[0][0]]
    x_right = x[np.where(y >= half_max_y)[0][-1]] 
    fwhm = x_right - x_left
    return fwhm


#FWHM error
def add_subtract_errors_gaussians(x, params, errors):
    params = np.asarray(params)
    errors = np.asarray(errors)
    
    # Original Gaussian
    gaussian_original = gaussian(x, *params)
    fwhm_original = calculate_fwhm(x, gaussian_original)
    
    # Gaussian with added errors
    params_plus = params + errors
    gaussian_plus = gaussian(x, *params_plus)
    fwhm_plus = calculate_fwhm(x, gaussian_plus)
    
    # Gaussian with subtracted errors
    params_minus = params - errors
    gaussian_minus = gaussian(x, *params_minus)
    fwhm_minus = calculate_fwhm(x, gaussian_minus)
    
    fwhm_error = fwhm_original - fwhm_minus
    
    return fwhm_error


def analyze_data(file_path, file_path_mf, label, bins=100, fit_range=None, phase_shift_error = 0):
    # Open the HDF5 files in read-only mode
    h5file = h5py.File(file_path, 'r')
    h5file_mf = h5py.File(file_path_mf, 'r')

    # Access specific datasets
    dataset = h5file['event_time']  
    dataset_mf = h5file_mf['matched_filtered'] 

    # Read data from datasets
    data = dataset[()]
    data_mf = dataset_mf[()]

    # Create histogram 
    hist, bin_edges = np.histogram(data_mf, bins=bins)
    bin_centers = (bin_edges[1:] + bin_edges[:-1]) / 2

    # Plot histogram
    plt.hist(data_mf, bins=bins, facecolor=color_palette[0], linewidth=0, alpha=0.6)

    # Gaussian fit
    initial_guess = [np.max(hist), np.mean(data_mf[300:1200]), np.std(data_mf[300:1200])]
    popt, pcov = curve_fit(gaussian, bin_centers, hist, p0=initial_guess)

    # Extract fitted params
    A, mean, std_dev = popt
    
    # Calculate the errors using the covariance matrix
    errors = np.sqrt(np.diag(pcov))
    delta_A, delta_mean, delta_std_dev = errors

    # Create x values for the Gaussian
    x = np.linspace(min(data_mf[0:1450]), max(data_mf[0:1450]), 1450)
    # Calculate the corresponding y values for the Gaussian fit curve
    y = gaussian(x, A, mean, std_dev)

    # Plot Gaussian fit
    plt.plot(x, y, 'g', linestyle='--', linewidth=2, label=f"{label} fit")

    # FWHM calculation
    fwhm_original = calculate_fwhm(x, y)
    
    #FWHM error
    # Example usage:
    x_values = np.linspace(min(bin_centers), max(bin_centers), 1000)
    original_params = A, mean, std_dev
    fwhm_error = add_subtract_errors_gaussians(x_values, original_params, errors)
    
    #energy resolution
    phase_shift = x[np.array(y).argmax()]
    energy_res = phase_shift/calculate_fwhm(x,y)

    #energy res error
    energy_res_error = energy_res*np.sqrt((phase_shift_error/phase_shift)**2+(fwhm_error/fwhm_original)**2)
    
    # Print and return results
    print(f'{label} Gaussian fit params:', A, '+/-', errors[0], mean, '+/-', errors[1], std_dev, '+/-', errors[2])
    print(f'{label} FWHM value:', fwhm_original, '+/-', fwhm_error )
    print(f'{label} energy res value:', energy_res, '+/-', energy_res_error )
    
    return A, mean, std_dev, fwhm_original, fwhm_error, energy_res, energy_res_error

# Example usage:
file_path = r"D:\Isabelle\isabelle_data\250717\GREEN40\output\PERFECT_MF\GREEN40\event_time.h5"
file_path_mf =r"D:\Isabelle\isabelle_data\250717\GREEN40\output\PERFECT_MF\GREEN40\matched_filtered.h5"

# You can call this function for different datasets
analyze_data(file_path, file_path_mf, label="Red Laser", bins=100, fit_range=[300, 1200])

plt.xlabel('Phase Shift')
plt.ylabel('Counts')
plt.grid(False)
plt.title('Gaussian Fit for the Data with 1 Res at 33dB')
plt.legend()
plt.show()




