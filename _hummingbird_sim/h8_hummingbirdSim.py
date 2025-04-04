import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlPD import ctrlPD

# instantiate pendulum, controller, and reference classes
hummingbird = HummingbirdDynamics(alpha=0.2)
controller = ctrlPD()
psi_ref = SignalGenerator(amplitude=30.*np.pi/180., frequency=0.02)
theta_ref = SignalGenerator(amplitude=15.*np.pi/180., frequency=0.05)

# instantiate the simulation plots and animation
dataPlot = DataPlotter()
animation = HummingbirdAnimation()

DISTURBANCE_TIME = 2.0  # When to apply step (seconds)
DISTURBANCE_FORCE = 1.0  # 1N force
disturbance_applied = False  # State flag

def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u

t = P.t_start  # time starts at t_start
y = hummingbird.h()
while t < P.t_end:  # main simulation loop

    # Propagate dynamics at rate Ts
    t_next_plot = t + P.t_plot
    while t < t_next_plot:
        r = np.array([[theta_ref.square(t)], [psi_ref.square(t)]])
        pwm, y_ref = controller.update(r, y)
        if not disturbance_applied and t >= DISTURBANCE_TIME:
            # Convert force to PWM and add to both motors
            disturbance_pwm = DISTURBANCE_FORCE / (2 * P.km)
            pwm += disturbance_pwm * np.ones((2, 1))
            pwm = saturate(pwm, 0, 1)
            disturbance_applied = True

        y = hummingbird.update(pwm)  # Propagate the dynamics
        t += P.Ts  # advance time by Ts

    # update animation and data plots at rate t_plot
    animation.update(t, hummingbird.state)
    dataPlot.update(t, hummingbird.state, pwm, y_ref)

    # the pause causes figure to be displayed during simulation
    plt.pause(0.0001)

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
