import numpy as np
import matplotlib.pyplot as plt

# Given signal data
signal = [1.28, 1.41, 1.16, 0.41, -0.48, -0.99, -0.66, 0.31, 1.09, 1.09,
          0.33, -0.74, -1.71, -2.48, -3.11, -3.7, -4.35, -4.98, -5.28, -4.96,
          -4.03, -2.81, -1.85, -1.38, -1.27, -1.21, -1.35, -0.37, 1.36, 3.02,
          3.99, 4.23, 4.18, 3.83, 3.23, 2.72, 2.4, 2.14, 1.83, 1.5, 1.21, 0.96,
          0.71, 0.45, 0.2, -0.01, -0.09, -0.02, 0.21, 0.56, 0.95]

# Time vector
t = np.linspace(0, 5, len(signal))  # Generate time vector

# Fourier Transform
N = len(signal)  # Number of data points
Y = np.fft.fft(signal)  # Compute FFT
frequencies = np.fft.fftfreq(N, d=(t[1] - t[0]))  # Frequencies
amplitudes = np.abs(Y) / N  # Amplitudes of Fourier components

# Reconstruct signal using limited Fourier terms
num_terms = 20  # Choose the number of Fourier terms to use
reconstructed_signal = np.zeros_like(t)

for k in range(num_terms):
    amplitude = 2 * np.abs(Y[k]) / N  # Amplitude scaling
    phase = np.angle(Y[k])  # Phase
    frequency = frequencies[k]  # Frequency
    # Add the Fourier term to the reconstruction
    reconstructed_signal += amplitude * np.cos(2 * np.pi * frequency * t + phase)

# Plot the original and reconstructed signal
plt.figure(figsize=(10, 6))
plt.plot(t, signal, 'b-', label='Original Signal', linewidth=2)  # Original signal
plt.plot(t, reconstructed_signal, 'r--', label='Fourier Approximation', linewidth=2)  # Reconstructed signal
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Fourier Series Approximation')
plt.legend()
plt.grid(True)
plt.show()

print(reconstructed_signal)
