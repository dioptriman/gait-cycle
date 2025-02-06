import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants for the hip and knee (representing the leg)
l1 = 1.0  # Length of the thigh (hip to knee)
l2 = 1.0  # Length of the shin (knee to foot)

# Gait cycle parameters
gait_duration = 2.0  # Duration for one full gait cycle in seconds
frequency = 1 / gait_duration  # Frequency of the gait cycle
omega = 2 * np.pi * frequency  # Angular frequency for sinusoidal motion
t_eval = np.linspace(0, gait_duration, 500)  # Time points for the gait cycle

# Angular motion formulas (hip and knee angles)
# The angle oscillates for both hip (thigh) and knee (shin)
theta_dh = 15 * np.sin(omega * t_eval) + 15  # Hip angle oscillates between -15° and +15°
theta_dk = 25 * np.sin(omega * t_eval) + 25  # Knee angle oscillates between 0° and 50°

# Convert angles to radians for calculation
theta_dh_rad = np.radians(theta_dh)
theta_dk_rad = np.radians(theta_dk)

# Set up the figure for animation
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Animation setup (first subplot)
ax[0].set_xlim(-2, 2)
ax[0].set_ylim(-2, 2)
ax[0].set_aspect('equal')
ax[0].set_xlabel('X Position (m)')
ax[0].set_ylabel('Y Position (m)')
ax[0].set_title('Gait Cycle Animation (Hip and Knee)')

# Angle vs Time Plot setup (second subplot)
ax[1].plot(t_eval, theta_dh, label='Hip (Thigh)', color='b')
ax[1].plot(t_eval, theta_dk, label='Knee (Shin)', color='r')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Angle (degrees)')
ax[1].set_title('Angle vs Time')
ax[1].legend()

# Initialize lines for the hip and knee (representing the leg)
line1, = ax[0].plot([], [], 'o-', lw=2, label='Hip (Thigh)')
line2, = ax[0].plot([], [], 'o-', lw=2, label='Knee (Shin)')
time_text = ax[0].text(0.05, 0.95, '', transform=ax[0].transAxes)

# Function to initialize the plot
def init():
    # Initialize positions to (0, 0) for both the hip and knee
    line1.set_data([], [])
    line2.set_data([], [])
    time_text.set_text('')
    return line1, line2, time_text

# Function to animate the plot
def animate(i):
    # Update the positions of the hip and knee using the current angles
    x1 = l1 * np.sin(theta_dh_rad[i])  # X position of knee based on hip angle
    y1 = -l1 * np.cos(theta_dh_rad[i])  # Y position of knee based on hip angle
    x2 = x1 + l2 * np.sin(theta_dk_rad[i])  # X position of foot based on knee angle
    y2 = y1 - l2 * np.cos(theta_dk_rad[i])  # Y position of foot based on knee angle
    
    # Update the lines for the hip (thigh) and knee (shin)
    line1.set_data([0, x1], [0, y1])  # Line from origin to knee (hip)
    line2.set_data([x1, x2], [y1, y2])  # Line from knee to foot (shin)
    
    # Update the time text
    time_text.set_text(f'Time: {t_eval[i]:.2f}s')
    
    return line1, line2, time_text

# Create the animation (using the first subplot)
ani = FuncAnimation(fig, animate, frames=len(t_eval), init_func=init, interval=20, blit=True)

# To save the animation, uncomment the line below (requires ffmpeg)
# ani.save('gait_cycle_animation.mp4', writer='ffmpeg', fps=30)

plt.tight_layout()
plt.show()
