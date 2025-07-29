#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:23:24 2024

@author: jzgs33
"""

import h5py
import numpy as np
import os
import time
import scipy.signal

"""
INSTRUCTIONs: change SD to filter out thermal events and foler paths - and espeically templates!!
"""

SD = 5

# File path to the H5 tables containing the phase streams.
folder_path_stream = r"D:\Isabelle\isabelle_data\250717\RED45"

# File path to the H5 tables containing the phase profiles.
folder_path_template = r"C:\Users\wlmd95\OneDrive - Durham University\Documents\PhD\L4_Project\MKID project\templates\template_300"

# Save phase_shift and event_time to separate H5 files
output_folder = folder_path_stream + r"/output/PERFECT_MF/RED45"  # Specify the folder where the output H5 files will be saved

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Start the timer
start_time = time.time()

# Load in the phase data from the exposures stream H5 tables in a specified folder.
phase_stream = []
for file_name in os.listdir(folder_path_stream):
    if file_name.endswith(".h5"):
        file_path = os.path.join(folder_path_stream, file_name)
        with h5py.File(file_path, "r") as f:
            res = f['Resonators']
            res1 = res['Res1']
            phase_stream.append(res1['Data'][:])

phase_template = []
for file_name in os.listdir(folder_path_template):
    if file_name.endswith(".h5"):
        file_path = os.path.join(folder_path_template, file_name)
        with h5py.File(file_path, "r") as f:
            phase_template.append(f['data'][:])


# Convert the lists to numpy arrays for easier manipulation
phase_stream = np.array(phase_stream, dtype=np.float32)

phase_template = np.array(phase_template, dtype=np.float32)

sum_of_template = np.sum(phase_template)

sum_of_template = sum_of_template - (len(phase_template[0]) * np.min(phase_template))

phase_template = (1/phase_template)

phase_template = phase_template - np.min(phase_template)

# Combine the arrays of multiple exposures in the phase_stream list into one longer continuous array.
combined_phase_stream = np.concatenate(phase_stream, axis=0)

# High-pass filter
high_pass_cutoff = 0.01
b_high, a_high = scipy.signal.butter(2, high_pass_cutoff, btype='high')
high_pass_filtered = scipy.signal.filtfilt(b_high, a_high, combined_phase_stream)

# Optimal filtering (Matched filter)
matched_filtered = scipy.signal.correlate(high_pass_filtered, phase_template[0], mode='same', method='auto')

#Calculate the noise. 
noise_dark = np.copy(matched_filtered)
noise_dark[np.where(noise_dark > 8)] = np.NAN

noise_dark_std = np.nanstd(noise_dark)

noise_dark_rms = np.sqrt(np.nanmean(noise_dark**2))

noise_dark_mean = np.nanmean(noise_dark)

dark_height = noise_dark_mean + (noise_dark_rms * SD)


peaks = scipy.signal.find_peaks(matched_filtered, height=(dark_height), threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=None, plateau_size=None)

event_time = peaks[0] - ((len(phase_template[0]) // 2) - 8)

matched_filtered = (matched_filtered * sum_of_template)

# Save matched_filtered[peaks[0]] to a separate H5 file
output_file_matched_filtered = os.path.join(output_folder, "matched_filtered.h5")
with h5py.File(output_file_matched_filtered, "w") as f:
    f.create_dataset("matched_filtered", data=matched_filtered[peaks[0]])

# Save event_time
output_file_event_time = os.path.join(output_folder, "event_time.h5")
with h5py.File(output_file_event_time, "w") as f:
    f.create_dataset("event_time", data=event_time)
    
# Sampling rate (assuming 1 sample per second for simplicity, adjust as needed)
sampling_rate = 1e6  # Hz

# Calculate the time axis
time_axis = np.arange(len(matched_filtered)) / sampling_rate

time_axis_temp = np.arange(len(phase_template)) / sampling_rate

# Print the execution time.
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")