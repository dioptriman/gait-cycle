import numpy as np
import matplotlib.pyplot as plt

# Data points
data = [1.28, 1.41, 1.16, 0.41, -0.48, -0.99, -0.66, 0.31, 1.09, 1.09, 
        0.33, -0.74, -1.71, -2.48, -3.11, -3.7, -4.35, -4.98, -5.28, 
        -4.96, -4.03, -2.81, -1.85, -1.38, -1.27, -1.21, -1.35, -0.37, 
        1.36, 3.02, 3.99, 4.23, 4.18, 3.83, 3.23, 2.72, 2.4, 2.14, 1.83, 
        1.5, 1.21, 0.96, 0.71, 0.45, 0.2, -0.01, -0.09, -0.02, 0.21, 
        0.56, 0.95]
n = len(data)
x = np.linspace(0, 2 * np.pi, n)

# Compute DFT
dft = np.fft.fft(data)
a0 = dft[0].real / n  # DC component
an = 2 * dft[1:n//2].real / n  # Cosine coefficients
bn = -2 * dft[1:n//2].imag / n  # Sine coefficients

# Reconstruct using Fourier series
reconstructed = a0 + sum(
    an[k] * np.cos((k + 1) * x) + bn[k] * np.sin((k + 1) * x) for k in range(len(an))
)

# Print Fourier coefficients
results = {
    "a0 (DC Component)": a0,
    "an (Cosine Coefficients)": an,
    "bn (Sine Coefficients)": bn,
    "Reconstructed Data": reconstructed
}

# Plot original data and reconstructed Fourier series
plt.plot(x, data, label="Original Data")
plt.plot(x, reconstructed, label="Fourier Series Approximation", linestyle="--")
plt.legend()
plt.title("Fourier Series Approximation")
plt.xlabel("x")
plt.ylabel("y")
plt.grid()
plt.show()

print(results)
