# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 16:16:50 2025

@author: wlmd95
"""

import numpy as np
import os
import matplotlib.pyplot as plt


directory = "D:\data"  
filename = "blue_0.bin"


path = os.path.join(directory, filename)


data = np.fromfile(path, dtype=np.int16)

#choose window for readability 
N = min(200000, data.size)

#convert to degrees assuming Int16 spans [-180°, 180°]
degrees = (data.astype(np.float64) / 32768.0) * 180.0
degrees_window = degrees[:N]
unwrapped_deg = np.unwrap(np.deg2rad(degrees_window)) * (180.0 / np.pi)

#plot phase stream (degrees)
plt.figure()
plt.plot(np.arange(N), degrees_window)
plt.title(f"Phase (degrees) assuming Int16 spans [-180°, 180°) (first {N} samples)")
plt.xlabel("Sample index")
plt.ylabel("Phase [°]")
plt.show()

#plot unwrapped phase in degrees
plt.figure()
plt.plot(np.arange(N), unwrapped_deg)
plt.title(f"Unwrapped Phase (degrees) (first {N} samples)")
plt.xlabel("Sample index")
plt.ylabel("Phase [°]")
plt.show()

# Save unwrapped degrees
np.save(os.path.join(directory, f"phase_unwrapped_degrees_first_{N}_samples.npy"), unwrapped_deg)
