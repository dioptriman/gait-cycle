import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.animation as animation

# Define signal data for left and right hip and knee flexion (for one cycle)
left_hip_flexion = [
    -16.56, -16.82, -16.53, -15.4, -13.44, -10.65, -7.15, -3.16, 0.95, 4.93, 
    8.63, 11.96, 14.96, 17.62, 19.89, 21.76, 23.24, 24.33, 24.98, 25.26, 
    25.25, 25.04, 24.8, 24.66, 24.66, 24.7, 24.61, 24.45, 24.13, 23.68, 
    22.96, 21.78, 20.24, 18.41, 16.38, 14.19, 11.91, 9.58, 7.22, 4.87, 
    2.57, 0.33, -1.85, -3.98, -6.06, -8.1, -10.06, -11.88, -13.5, -14.85, -15.88
]
t_left_hip = np.linspace(0, 1, len(left_hip_flexion))

right_hip_flexion = [
    24.61, 24.45, 24.13, 23.68, 22.96, 21.78, 20.24, 18.41, 16.38, 14.19, 
    11.91, 9.58, 7.22, 4.87, 2.57, 0.33, -1.85, -3.98, -6.06, -8.1, -10.06, 
    -11.88, -13.5, -14.85, -15.88, -16.56, -16.82, -16.53, -15.4, -13.44, 
    -10.65, -7.15, -3.16, 0.95, 4.93, 8.63, 11.96, 14.96, 17.62, 19.89, 
    21.76, 23.24, 24.33, 24.98, 25.26, 25.25, 25.04, 24.8, 24.66, 24.66, 24.7
]
t_right_hip = np.linspace(0, 1, len(right_hip_flexion))

left_knee_angle = [
    -8.2, -11.29, -15.19, -20.17, -26.06, -32.67, -39.6, -46.16, -51.68, -55.66, 
    -57.83, -58.25, -57.28, -55.06, -51.58, -46.99, -41.48, -35.19, -28.31, -21.23, 
    -14.44, -8.48, -4.3, -2.27, -2.35, -4.14, -3.94, -6.66, -9.45, -12.26, -14.76, 
    -16.37, -17.04, -16.93, -16.24, -15.15, -13.87, -12.46, -10.95, -9.42, -7.96, 
    -6.59, -5.37, -4.29, -3.4, -2.72, -2.37, -2.44, -3, -4.08, -5.83, 0
]
t_left_knee = np.linspace(0, 1, len(left_knee_angle))

knee_right_angle = [
    -3.94, -6.66, -9.45, -12.26, -14.76, -16.37, -17.04, -16.93, -16.24, -15.15,
    -13.87, -12.46, -10.95, -9.42, -7.96, -6.59, -5.37, -4.29, -3.4, -2.72,
    -2.37, -2.44, -3, -4.08, -5.83, -8.2, -11.29, -15.19, -20.17, -26.06,
    -32.67, -39.6, -46.16, -51.68, -55.66, -57.83, -58.25, -57.28, -55.06, -51.58,
    -46.99, -41.48, -35.19, -28.31, -21.23, -14.44, -8.48, -4.3, -2.27, -2.35, -4.14, 0
]
t_right_knee = np.linspace(0, 1, len(knee_right_angle))

# Time vector for a single cycle
t_interp = np.linspace(0, 1, 1000)

# Interpolate data using cubic splines for the first cycle
cs_left_hip = CubicSpline(t_left_hip, left_hip_flexion)
cs_right_hip = CubicSpline(t_right_hip, right_hip_flexion)
cs_left_knee = CubicSpline(t_left_knee, left_knee_angle)
cs_right_knee = CubicSpline(t_right_knee, knee_right_angle)

# Simulate the first cycle
reconstructed_left_hip_1 = cs_left_hip(t_interp)
reconstructed_right_hip_1 = cs_right_hip(t_interp)
reconstructed_left_knee_1 = cs_left_knee(t_interp)
reconstructed_right_knee_1 = cs_right_knee(t_interp)

# Reset initial conditions for the second cycle (use the final values of the first cycle)
cs_left_hip_2 = CubicSpline(t_interp, reconstructed_left_hip_1)
cs_right_hip_2 = CubicSpline(t_interp, reconstructed_right_hip_1)
cs_left_knee_2 = CubicSpline(t_interp, reconstructed_left_knee_1)
cs_right_knee_2 = CubicSpline(t_interp, reconstructed_right_knee_1)

# Simulate the second cycle
reconstructed_left_hip_2 = cs_left_hip_2(t_interp)
reconstructed_right_hip_2 = cs_right_hip_2(t_interp)
reconstructed_left_knee_2 = cs_left_knee_2(t_interp)
reconstructed_right_knee_2 = cs_right_knee_2(t_interp)

# Combine both cycles for the two-cycle simulation
reconstructed_left_hip = np.concatenate([reconstructed_left_hip_1, reconstructed_left_hip_2])
reconstructed_right_hip = np.concatenate([reconstructed_right_hip_1, reconstructed_right_hip_2])
reconstructed_left_knee = np.concatenate([reconstructed_left_knee_1, reconstructed_left_knee_2])
reconstructed_right_knee = np.concatenate([reconstructed_right_knee_1, reconstructed_right_knee_2])

# Create a larger time vector for the two cycles
t_interp_two_cycles = np.linspace(0, 2, len(reconstructed_left_hip))

# Create figure for the angle plot
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 5))
ax3.plot(t_interp_two_cycles, reconstructed_left_hip, label="Left Hip", color='b')
ax3.plot(t_interp_two_cycles, reconstructed_right_hip, label="Right Hip", color='r')
ax3.plot(t_interp_two_cycles, reconstructed_left_knee, label="Left Knee", color='g')
ax3.plot(t_interp_two_cycles, reconstructed_right_knee, label="Right Knee", color='y')

ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Angle (degrees)')
ax3.set_title('Joint Angles Over Time (Two Cycles)')
ax3.legend()
ax3.grid(True)

# Create a function for the animation update
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)

# Initialize the plot elements for the animation
line_left_hip, = ax.plot([], [], 'bo-', label="Left Hip")
line_right_hip, = ax.plot([], [], 'ro-', label="Right Hip")
line_left_knee, = ax.plot([], [], 'go-', label="Left Knee")
line_right_knee, = ax.plot([], [], 'yo-', label="Right Knee")

# Function to initialize the animation
def init():
    line_left_hip.set_data([], [])
    line_right_hip.set_data([], [])
    line_left_knee.set_data([], [])
    line_right_knee.set_data([], [])
    return line_left_hip, line_right_hip, line_left_knee, line_right_knee

# Function to update the animation
def update(frame):
    # Update the position of the left leg (double pendulum)
    hip_left_x = np.sin(np.radians(reconstructed_left_hip[frame]))
    hip_left_y = -np.cos(np.radians(reconstructed_left_hip[frame]))
    knee_left_x = hip_left_x + np.sin(np.radians(reconstructed_left_knee[frame]))
    knee_left_y = hip_left_y - np.cos(np.radians(reconstructed_left_knee[frame]))

    # Update the position of the right leg (double pendulum)
    hip_right_x = np.sin(np.radians(reconstructed_right_hip[frame]))
    hip_right_y = -np.cos(np.radians(reconstructed_right_hip[frame]))
    knee_right_x = hip_right_x + np.sin(np.radians(reconstructed_right_knee[frame]))
    knee_right_y = hip_right_y - np.cos(np.radians(reconstructed_right_knee[frame]))

    # Plot the updated positions
    line_left_hip.set_data([0, hip_left_x], [0, hip_left_y])
    line_left_knee.set_data([hip_left_x, knee_left_x], [hip_left_y, knee_left_y])

    line_right_hip.set_data([0, hip_right_x], [0, hip_right_y])
    line_right_knee.set_data([hip_right_x, knee_right_x], [hip_right_y, knee_right_y])

    return line_left_hip, line_right_hip, line_left_knee, line_right_knee

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=len(t_interp_two_cycles), init_func=init, blit=True, interval=50)

# # Set up the writer for saving the animation
# writer = animation.FFMpegWriter(fps=20, metadata=dict(artist="Dika"), bitrate=1800)

# # Save the animation as a video file (e.g., "joint_angles_animation.mp4")
# ani.save("joint_angles_animation_leg.mp4", writer=writer)

# Show the plot
plt.show()
