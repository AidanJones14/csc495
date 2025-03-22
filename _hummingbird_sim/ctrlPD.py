import numpy as np
import hummingbirdParam as P
from ctrlLatRoll import ctrlLatRoll
from ctrlLatYaw import ctrlLatYaw
from ctrlLonPD import ctrlLonPD

class ctrlPD:
    def __init__(self):
        # Initialize the pitch, roll, and yaw controllers
        self.pitch_controller = ctrlLonPD()  
        self.roll_controller = ctrlLatRoll()  
        self.yaw_controller = ctrlLatYaw()  

    def update(self, r, y):
        # Extract reference signals
        theta_ref = r[0][0]  # Desired pitch angle
        psi_ref = r[1][0]  # Desired yaw angle

        # Extract current state
        phi = y[0][0]  # Current roll angle

        # Update yaw controller (outer loop)
        phi_d = self.yaw_controller.update(r, y)

        # Update roll controller (inner loop)
        tau_phi = self.roll_controller.update(phi_d, phi)

        # Update pitch controller
        pwm_pitch, y_ref_pitch = self.pitch_controller.update(r, y) 

        # Combine torques and force
        pwm = pwm_pitch + np.array([[tau_phi / P.d], [-tau_phi / P.d]]) / (2 * P.km)

        # Saturate PWM signals
        pwm = saturate(pwm, 0, 1)

        # Return PWM signals and reference signals
        y_ref = np.array([[phi_d], [theta_ref], [psi_ref]])
        return pwm, y_ref
    
def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u