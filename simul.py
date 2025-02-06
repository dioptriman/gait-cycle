import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.animation as animation

# Define signal data for left and right hip and knee flexion
# Left hip flexion (already given)
left_hip_flexion = [
    -16.56, -16.82, -16.53, -15.4, -13.44, -10.65, -7.15, -3.16, 0.95, 4.93, 
    8.63, 11.96, 14.96, 17.62, 19.89, 21.76, 23.24, 24.33, 24.98, 25.26, 
    25.25, 25.04, 24.8, 24.66, 24.66, 24.7, 24.61, 24.45, 24.13, 23.68, 
    22.96, 21.78, 20.24, 18.41, 16.38, 14.19, 11.91, 9.58, 7.22, 4.87, 
    2.57, 0.33, -1.85, -3.98, -6.06, -8.1, -10.06, -11.88, -13.5, -14.85, -15.88
]
t_left_hip = np.linspace(0, 1, len(left_hip_flexion))

# Right hip flexion (provided)
right_hip_flexion = [
    24.61, 24.45, 24.13, 23.68, 22.96, 21.78, 20.24, 18.41, 16.38, 14.19, 
    11.91, 9.58, 7.22, 4.87, 2.57, 0.33, -1.85, -3.98, -6.06, -8.1, -10.06, 
    -11.88, -13.5, -14.85, -15.88, -16.56, -16.82, -16.53, -15.4, -13.44, 
    -10.65, -7.15, -3.16, 0.95, 4.93, 8.63, 11.96, 14.96, 17.62, 19.89, 
    21.76, 23.24, 24.33, 24.98, 25.26, 25.25, 25.04, 24.8, 24.66, 24.66, 24.7
]
t_right_hip = np.linspace(0, 1, len(right_hip_flexion))

# Left knee angle (new data)
left_knee_angle = [
    -8.2, -11.29, -15.19, -20.17, -26.06, -32.67, -39.6, -46.16, -51.68, -55.66, 
    -57.83, -58.25, -57.28, -55.06, -51.58, -46.99, -41.48, -35.19, -28.31, -21.23, 
    -14.44, -8.48, -4.3, -2.27, -2.35, -4.14, -3.94, -6.66, -9.45, -12.26, -14.76, 
    -16.37, -17.04, -16.93, -16.24, -15.15, -13.87, -12.46, -10.95, -9.42, -7.96, 
    -6.59, -5.37, -4.29, -3.4, -2.72, -2.37, -2.44, -3, -4.08, -5.83
]
t_left_knee = np.linspace(0, 1, len(left_knee_angle))

# Right knee angle (provided, same as left knee data for demonstration purposes)
knee_right_angle = [
    -3.94, -6.66, -9.45, -12.26, -14.76, -16.37, -17.04, -16.93, -16.24, -15.15,
    -13.87, -12.46, -10.95, -9.42, -7.96, -6.59, -5.37, -4.29, -3.4, -2.72,
    -2.37, -2.44, -3, -4.08, -5.83, -8.2, -11.29, -15.19, -20.17, -26.06,
    -32.67, -39.6, -46.16, -51.68, -55.66, -57.83, -58.25, -57.28, -55.06, -51.58,
    -46.99, -41.48, -35.19, -28.31, -21.23, -14.44, -8.48, -4.3, -2.27, -2.35, -4.14
]
t_right_knee = np.linspace(0, 1, len(knee_right_angle))


# Time vector for interpolation
t_interp = np.linspace(0, 1, 500)

# Perform cubic spline interpolation for all signals
cs_left_hip = CubicSpline(t_left_hip, left_hip_flexion)
cs_right_hip = CubicSpline(t_right_hip, right_hip_flexion)
cs_left_knee = CubicSpline(t_left_knee, left_knee_angle)
cs_right_knee = CubicSpline(t_right_knee, knee_right_angle)

# Reconstructed signals after interpolation
reconstructed_left_hip = cs_left_hip(t_interp)
reconstructed_right_hip = cs_right_hip(t_interp)
reconstructed_left_knee = cs_left_knee(t_interp)
reconstructed_right_knee = cs_right_knee(t_interp)

# Convert angles to radians
theta_left_hip = np.deg2rad(reconstructed_left_hip)
theta_right_hip = np.deg2rad(reconstructed_right_hip)
theta_left_knee = np.deg2rad(reconstructed_left_knee)
theta_right_knee = np.deg2rad(reconstructed_right_knee)

# Compute angular velocities (dtheta/dt) using numerical differentiation
dtheta_left_hip_dt = np.gradient(theta_left_hip, t_interp)
dtheta_right_hip_dt = np.gradient(theta_right_hip, t_interp)
dtheta_left_knee_dt = np.gradient(theta_left_knee, t_interp)
dtheta_right_knee_dt = np.gradient(theta_right_knee, t_interp)

# Pendulum simulation parameters
L1 = 1  # Length of the first pendulum rod
L2 = 1  # Length of the second pendulum rod

# Compute positions of the pendulum masses for each leg (hip and knee)
x_left_hip = L1 * np.sin(theta_left_hip)
y_left_hip = -L1 * np.cos(theta_left_hip)
x_left_knee = x_left_hip + L2 * np.sin(theta_left_knee)
y_left_knee = y_left_hip - L2 * np.cos(theta_left_knee)

x_right_hip = L1 * np.sin(theta_right_hip)
y_right_hip = -L1 * np.cos(theta_right_hip)
x_right_knee = x_right_hip + L2 * np.sin(theta_right_knee)
y_right_knee = y_right_hip - L2 * np.cos(theta_right_knee)

# Create figure for the angle plot
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
ax3.plot(t_interp, reconstructed_left_hip, label="Left Hip", color='b')
ax3.plot(t_interp, reconstructed_right_hip, label="Right Hip", color='r')
ax3.plot(t_interp, reconstructed_left_knee, label="Left Knee", color='g')
ax3.plot(t_interp, reconstructed_right_knee, label="Right Knee", color='y')

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angle (degrees)')
ax3.set_title('Joint Angles Over Time')
ax3.legend()
ax3.grid(True)

# Create figure for the phase plot
fig2, ax2 = plt.subplots(1, 1, figsize=(7, 5))
ax2.plot(theta_left_hip, dtheta_left_hip_dt, label="Left Hip", color='b')
ax2.plot(theta_right_hip, dtheta_right_hip_dt, label="Right Hip", color='r')
ax2.plot(theta_left_knee, dtheta_left_knee_dt, label="Left Knee", color='g')
ax2.plot(theta_right_knee, dtheta_right_knee_dt, label="Right Knee", color='y')

ax2.set_xlabel('Angle (rad)')
ax2.set_ylabel('Angular Velocity (rad/s)')
ax2.set_title('Phase Plot of Hip and Knee Angles vs. Angular Velocities')
ax2.legend()
ax2.grid(True)

# Create figure for the double pendulum animation
fig1, ax1 = plt.subplots()
ax1.set_xlim(-2.5, 2.5)
ax1.set_ylim(-2.5, 1)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Y (m)')
ax1.set_title('Double Pendulum Animation for Both Legs')
ax1.grid(True)

# Create line objects for the rods and masses
rod_left_hip, = ax1.plot([], [], 'b-', lw=2)
rod_left_knee, = ax1.plot([], [], 'g-', lw=2)
rod_right_hip, = ax1.plot([], [], 'r-', lw=2)
rod_right_knee, = ax1.plot([], [], 'y-', lw=2)

mass_left_hip, = ax1.plot([], [], 'bo', markersize=10, markerfacecolor='b')
mass_left_knee, = ax1.plot([], [], 'go', markersize=10, markerfacecolor='g')
mass_right_hip, = ax1.plot([], [], 'ro', markersize=10, markerfacecolor='r')
mass_right_knee, = ax1.plot([], [], 'yo', markersize=10, markerfacecolor='y')

def init():
    rod_left_hip.set_data([], [])
    rod_left_knee.set_data([], [])
    rod_right_hip.set_data([], [])
    rod_right_knee.set_data([], [])
    mass_left_hip.set_data([], [])
    mass_left_knee.set_data([], [])
    mass_right_hip.set_data([], [])
    mass_right_knee.set_data([], [])
    return rod_left_hip, rod_left_knee, rod_right_hip, rod_right_knee, mass_left_hip, mass_left_knee, mass_right_hip, mass_right_knee

def animate(i):
    # Update rod positions
    rod_left_hip.set_data([0, x_left_hip[i]], [0, y_left_hip[i]])
    rod_left_knee.set_data([x_left_hip[i], x_left_knee[i]], [y_left_hip[i], y_left_knee[i]])
    rod_right_hip.set_data([0, x_right_hip[i]], [0, y_right_hip[i]])
    rod_right_knee.set_data([x_right_hip[i], x_right_knee[i]], [y_right_hip[i], y_right_knee[i]])

    # Update mass positions (ensure these are passed as sequences)
    mass_left_hip.set_data([x_left_hip[i]], [y_left_hip[i]])  # Enclose values in lists
    mass_left_knee.set_data([x_left_knee[i]], [y_left_knee[i]])  # Enclose values in lists
    mass_right_hip.set_data([x_right_hip[i]], [y_right_hip[i]])  # Enclose values in lists
    mass_right_knee.set_data([x_right_knee[i]], [y_right_knee[i]])  # Enclose values in lists

    return rod_left_hip, rod_left_knee, rod_right_hip, rod_right_knee, mass_left_hip, mass_left_knee, mass_right_hip, mass_right_knee


# Save animation as a .mp4 file
ani = animation.FuncAnimation(fig1, animate, frames=len(t_interp), init_func=init, blit=True, interval=30)
ani.save('leg_animation_1.mp4', writer='ffmpeg', fps=30)

# Show the angle and phase plots
plt.show()
