# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 14:42:23 2025

@author: wlmd95
"""

"""
INSTRUCTIONS: JUST ADD RED, GREEN AND BLUE FILES IF NEEDED IN THE FUNCTION, EDIT FIT RANGES ETC ETC
"""


import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit
import re

sns.set(style="whitegrid")
color_palette = sns.color_palette("Set2")

# Gaussian function
def gaussian(x, A, mean, std_dev):
    return A * np.exp(-(x - mean) ** 2 / (2 * std_dev ** 2))

# FWHM computation
def calculate_fwhm(x, y):
    half_max = max(y) / 2
    indices = np.where(y >= half_max)[0]
    if len(indices) < 2:
        raise ValueError("FWHM cannot be computed; peak too narrow.")
    return x[indices[-1]] - x[indices[0]]

# Error in FWHM from covariance matrix
def fwhm_error_from_cov(x, params, errors):
    try:
        fwhm_nominal = calculate_fwhm(x, gaussian(x, *params))
        fwhm_minus = calculate_fwhm(x, gaussian(x, *(params - errors)))
        return fwhm_nominal - fwhm_minus
    except ValueError as e:
        print(f"Warning: {e} — setting FWHM error to 0.")
        return 0.0

# Gaussian fit
def fit_gaussian(data, bins, fit_range):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    mask = (bin_centers >= fit_range[0]) & (bin_centers <= fit_range[1])
    if not np.any(mask):
        raise ValueError(f"No histogram bins in range {fit_range}.")

    A_guess = np.max(hist[mask])
    mean_guess = bin_centers[mask][np.argmax(hist[mask])]
    std_guess = np.std(data)
    popt, pcov = curve_fit(gaussian, bin_centers[mask], hist[mask], p0=[A_guess, mean_guess, std_guess])
    errors = np.sqrt(np.diag(pcov))
    x_fit = np.linspace(min(data), max(data), 2000)
    y_fit = gaussian(x_fit, *popt)
    return popt, errors, x_fit, y_fit, hist, bin_centers

# Extract metadata
def extract_info_from_path(path):
    date_match = re.search(r'(\d{6})', path)
    colour_match = re.search(r'(RED|GREEN|BLUE)', path.upper())
    return date_match.group(1) if date_match else "UnknownDate", colour_match.group(1).capitalize() if colour_match else "Unknown"

# Generalised plot function
def plot_histograms(red_file=None, green_file=None, blue_file=None,
                    bins=100, phase_shift_error=0.0,
                    fit_ranges=None):

    input_files = {
        "Red": red_file,
        "Green": green_file,
        "Blue": blue_file
    }

    results = {}
    all_data = []

    # Plot setup
    plt.figure(figsize=(8, 6))
    for colour, file_path in input_files.items():
        if file_path is None:
            continue

        with h5py.File(file_path, 'r') as f:
            data = f[list(f.keys())[0]][()]
        date_str, detected_colour = extract_info_from_path(file_path)

        fit_range = fit_ranges.get(colour.lower(), [3, 11])

        # Fit
        popt, errors, x_fit, y_fit, hist, bin_centers = fit_gaussian(data, bins, fit_range)
        fwhm = calculate_fwhm(x_fit, y_fit)
        fwhm_err = fwhm_error_from_cov(x_fit, popt, errors)
        peak_shift = x_fit[np.argmax(y_fit)]
        resolution = peak_shift / fwhm
        resolution_err = resolution * np.sqrt((phase_shift_error / peak_shift) ** 2 + (fwhm_err / fwhm) ** 2)

        # Plot
        plt.hist(data, bins=bins, alpha=0.3, label=f"{colour} Data", color=colour.lower())
        plt.plot(x_fit, y_fit, linestyle='--', linewidth=2, color=colour.lower(), label=f"{colour} Fit")

        # Store results
        results[colour.lower()] = {
            "mean": popt[1],
            "fwhm": fwhm,
            "res": resolution,
            "res_err": resolution_err
        }

        print(f"[{colour.upper()}] Mean: {popt[1]:.4f} ± {errors[1]:.4f}, "
              f"FWHM: {fwhm:.4f} ± {fwhm_err:.4f}, "
              f"Res: {resolution:.4f} ± {resolution_err:.4f}")

        all_data.append((colour, data))

    # Final plot adjustments
    plt.xlabel("Phase Shift (rad)", fontsize=14)
    plt.ylabel("Counts", fontsize=14)
    plt.title(f"{date_str} – Gaussian Fit for Laser Data", fontsize=15)
    plt.legend()
    plt.minorticks_on()
    plt.tick_params(which='both', direction='in', top=True, right=True)
    plt.grid(False)
    plt.tight_layout()
    plt.show()

    return results


green_file = r"D:\Isabelle\isabelle_data\250717\GREEN45\output\PERFECT_MF\GREEN45\matched_filtered.h5"
red_file   = r"D:\Isabelle\isabelle_data\250717\RED45\output\PERFECT_MF\RED45\matched_filtered.h5"

# Without blue
plot_histograms(
    red_file=r"D:\Isabelle\isabelle_data\250717\RED45\output\PERFECT_MF\RED45\matched_filtered.h5",
    green_file=r"D:\Isabelle\isabelle_data\250717\GREEN45\output\PERFECT_MF\GREEN45\matched_filtered.h5",
    bins=100,
    phase_shift_error=0.005,
    fit_ranges={
        "red": [3, 11],
        "green": [3, 11]
    }
)

