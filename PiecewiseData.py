import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial

# Given signal data
signal = [1.28, 1.41, 1.16, 0.41, -0.48, -0.99, -0.66, 0.31, 1.09, 1.09,
          0.33, -0.74, -1.71, -2.48, -3.11, -3.7, -4.35, -4.98, -5.28, -4.96,
          -4.03, -2.81, -1.85, -1.38, -1.27, -1.21, -1.35, -0.37, 1.36, 3.02,
          3.99, 4.23, 4.18, 3.83, 3.23, 2.72, 2.4, 2.14, 1.83, 1.5, 1.21, 0.96,
          0.71, 0.45, 0.2, -0.01, -0.09, -0.02, 0.21, 0.56, 0.95]

# Time vector
t = np.linspace(0, 5, len(signal))  # Generate time vector

# --- Piecewise Polynomial Approximation ---
# Define segment boundaries
segments = [0, 1, 2, 3, 4, 5]  # Divide into 5 equal segments
reconstructed_piecewise = np.zeros_like(t)

# Fit polynomials to each segment
for i in range(len(segments) - 1):
    # Determine the indices for the current segment
    start_idx = int(segments[i] * len(signal) / 5)
    end_idx = int(segments[i + 1] * len(signal) / 5)
    
    # Extract the time and signal data for the current segment
    t_seg = t[start_idx:end_idx]
    signal_seg = signal[start_idx:end_idx]
    
    # Fit a 3rd-degree polynomial to the segment
    p = Polynomial.fit(t_seg, signal_seg, deg=3)
    reconstructed_piecewise[start_idx:end_idx] = p(t_seg)

# --- Plot the Results ---
plt.figure(figsize=(10, 6))

# Original Signal
plt.plot(t, signal, 'b-', label='Original Signal', linewidth=2)

# Piecewise Polynomial Approximation
plt.plot(t, reconstructed_piecewise, 'r--', label='Piecewise Polynomial', linewidth=2)

# Plot Settings
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Piecewise Polynomial Approximation')
plt.legend()
plt.grid(True)
plt.show()
