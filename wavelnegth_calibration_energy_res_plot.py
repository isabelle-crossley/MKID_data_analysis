# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 13:59:22 2025

@author: wlmd95
"""

"""
INSTRUCTIONS: fill out top arrays having used gaussian_fit_all_lasers.py then 
select which indices you want for which lasers you are calibrating
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Map of wavelength to corresponding data
wavelengths_nm = np.array([405, 520, 635])   # blue, green, red lasers
colors = ['blue', 'green', 'red']
phases = np.array([12.6, 10.0, 7.91])
phase_errors = np.array([0.10, 0.0116, 0.005])
energy_res = np.array([14.6, 13.0, 11.9])
energy_res_err = np.array([2, 0.2, 0.2])

# Select any 2 or 3 lasers by index
# e.g., [0,1] for blue and green; [0,1,2] for all
selected_indices = [1, 2]  # Change this as needed

# Filter arrays accordingly
x = phases[selected_indices]
x_err = phase_errors[selected_indices]
y = wavelengths_nm[selected_indices]
energy = energy_res[selected_indices]
energy_err = energy_res_err[selected_indices]
color_labels = [colors[i] for i in selected_indices]

def linear(x, m, c):
    return m * x + c

def get_params_cal(xdata, ydata, initial_guess):
    if len(xdata) >= 3:
        params, cov = curve_fit(linear, xdata, ydata, p0=initial_guess, maxfev=50000)
        perr = np.sqrt(np.diag(cov))
    elif len(xdata) == 2:
        # manual linear fit for 2 points
        m = (ydata[1] - ydata[0]) / (xdata[1] - xdata[0])
        c = ydata[0] - m * xdata[0]
        params = [m, c]
        perr = [np.nan, np.nan]  # error unknown
    else:
        raise ValueError("Need at least 2 data points for a linear fit.")
    
    fit_y = linear(xdata, *params)
    return fit_y, params[0], params[1], perr

# --- Phase vs Wavelength plot ---
initial_guess = [0.004, 0]
cal_fit, m, c, mcerrors = get_params_cal(x, y, initial_guess)
print("Wavelength vs Phase:")
print(f"  Slope (m): {m:.4f}")
print(f"  Intercept (c): {c:.4f}")
print(f"  Slope error: {mcerrors[0]}")
print(f"  Intercept error: {mcerrors[1]}")

plt.figure(figsize=(5, 6.4))
plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=14)
plt.grid(False)
plt.xlabel('Phase Shift', fontsize=20)
plt.ylabel('Wavelength (nm)', fontsize=20)
plt.plot(x, cal_fit, color='black', alpha=0.8)

# Plot with color labels
for i in range(len(x)):
    plt.errorbar(x[i], y[i], xerr=x_err[i], fmt='o', color=colors[selected_indices[i]], capsize=3, label=color_labels[i])
plt.legend()
plt.show()

# --- Wavelength vs Energy Resolution plot ---
initial_guess = [0.004, 0]
cal_fit_2, m2, c2, mcerrors2 = get_params_cal(y, energy, initial_guess)
print("\nEnergy Resolution vs Wavelength:")
print(f"  Slope (m): {m2:.4f}")
print(f"  Intercept (c): {c2:.4f}")
print(f"  Slope error: {mcerrors2[0]}")
print(f"  Intercept error: {mcerrors2[1]}")

plt.figure(figsize=(5, 6.4))
plt.minorticks_on()
plt.tick_params(which='both', direction='in', top=True, right=True, labelsize=14)
plt.grid(False)
plt.xlabel('Wavelength (nm)', fontsize=20)
plt.ylabel('Energy Resolution', fontsize=20)
plt.plot(y, cal_fit_2, color='black', alpha=0.8)

for i in range(len(y)):
    plt.errorbar(y[i], energy[i], yerr=energy_err[i], fmt='o', color=colors[selected_indices[i]], capsize=3, label=color_labels[i])
plt.legend()
plt.show()
