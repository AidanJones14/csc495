import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlLatRoll import ctrlLatRoll  # Import the roll controller
from ctrlLatYaw import ctrlLatYaw  # Import the yaw controller

# Instantiate hummingbird, controllers, and reference classes
hummingbird = HummingbirdDynamics(alpha=0.1)
roll_controller = ctrlLatRoll()  # Roll controller (inner loop)
yaw_controller = ctrlLatYaw()  # Yaw controller (outer loop)

# Reference signals for yaw and roll
psi_ref = SignalGenerator(amplitude=30. * np.pi / 180., frequency=0.02)  # Yaw reference
theta_ref = SignalGenerator(amplitude=15. * np.pi / 180., frequency=0.05)  # Pitch reference (optional)

# Instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  # Time starts at t_start
y = hummingbird.h()
while t < P.t_end:  # Main simulation loop
    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        # Generate reference signals
        psi_d = psi_ref.square(t)  # Desired yaw angle
        theta_d = theta_ref.square(t)  # Desired pitch angle (optional)

        # Get current roll and yaw angles from the hummingbird state
        phi = y[0][0]  # Current roll angle
        psi = y[2][0]  # Current yaw angle

        # Update yaw controller (outer loop)
        phi_d = yaw_controller.update(psi_d, psi)  # Desired roll angle (from yaw controller)

        # Update roll controller (inner loop)
        tau_phi = roll_controller.update(phi_d, phi)  # Roll torque

        # Combine torques to generate control signals (PWM for motors)
        # For now, we're only using the roll torque (tau_phi)
        pwm = np.array([[tau_phi / P.km]])  # Convert torque to PWM (example)
        pwm = saturate(pwm, 0, 1)  # Saturate PWM signals to stay within 0 to 1

        # Propagate the dynamics
        y = hummingbird.update(pwm)  # Update hummingbird state
        t += P.Ts  # Advance time by Ts

    # Update animation and data plots at rate t_plot
    animation.update(t, hummingbird.state)
    dataPlot.update(t, hummingbird.state, pwm, np.array([[phi_d], [psi_d]]))

    # Pause to display the figure during simulation
    plt.pause(0.0001)

# Keep the program from closing until the user presses a button
print('Press key to close')
plt.waitforbuttonpress()
plt.close()