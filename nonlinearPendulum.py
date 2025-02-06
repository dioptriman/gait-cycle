import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
g = 9.81  # Acceleration due to gravity (m/s^2)
L = 1.0   # Length of the pendulum (m)

# Differential equation for the pendulum
def pendulum_eq(t, y):
    theta, omega = y
    dtheta_dt = omega
    domega_dt = -(g / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]

# Time span for the simulation
t_span = (0, 10)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Initial conditions for the pendulum
initial_conditions = [
    (np.pi / 3, 0),  # Moderate initial angle, no initial velocity
    (np.pi / 6, 0),  # Large initial angle, no initial velocity
    (0, 2),          # No initial angle, high initial velocity
    (np.pi, 0),      # Upside-down position, no initial velocity    
]

# Solve and plot phase plane
plt.figure(figsize=(10, 8))

for theta0, omega0 in initial_conditions:
    sol = solve_ivp(pendulum_eq, t_span, [theta0, omega0], t_eval=t_eval, method='RK45')
    plt.plot(sol.y[0], sol.y[1], label=f"$\\theta_0={theta0:.2f}, \\omega_0={omega0:.2f}$")

# Plot formatting
plt.title("Phase Plane Plot of Nonlinear Pendulum", fontsize=14)
plt.xlabel(r"$\theta$ (rad)", fontsize=12)
plt.ylabel(r"$\dot{\theta}$ (rad/s)", fontsize=12)
plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.show()
