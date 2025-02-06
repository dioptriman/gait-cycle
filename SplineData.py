import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Given signal data
signal = [1.28, 1.41, 1.16, 0.41, -0.48, -0.99, -0.66, 0.31, 1.09, 1.09, 
          0.33, -0.74, -1.71, -2.48, -3.11, -3.7, -4.35, -4.98, -5.28, 
          -4.96, -4.03, -2.81, -1.85, -1.38, -1.27, -1.21, -1.35, -0.37, 
          1.36, 3.02, 3.99, 4.23, 4.18, 3.83, 3.23, 2.72, 2.4, 2.14, 1.83, 
          1.5, 1.21, 0.96, 0.71, 0.45, 0.2, -0.01, -0.09, -0.02, 0.21, 
          0.56, 0.95]

# Generate time vector matching the signal length
t = np.linspace(0, 5, len(signal))  # Create a time vector

# Perform cubic spline interpolation
original_t = np.linspace(0, 5, len(signal))  # Original time vector based on signal length
spline_interp = interp1d(original_t, signal, kind='cubic')  # Cubic spline interpolation
reconstructed_signal = spline_interp(t)  # Interpolated values at the new time points

# Plot the original signal and the fitted spline curve
plt.figure(figsize=(10, 6))
plt.plot(t, signal, 'b-', label='Original Signal', linewidth=1.5)  # Original signal
plt.plot(t, reconstructed_signal, 'r--', label='Fitted Spline', linewidth=1.5)  # Spline curve
plt.xlabel('Time (s)')
plt.ylabel('Left Hip Rotation')
plt.title('Figure 6 - Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()
