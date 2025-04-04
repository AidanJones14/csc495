import numpy as np
import hummingbirdParam as P

class ctrlLonPD:
    def __init__(self):
        # rise time
        tr_pitch = 0.3
        # damping ratio
        zeta_pitch = 0.707
        
        # Calculate natural frequency
        wn_pitch = 2.2 / tr_pitch
        
        # gains and b theta value
        b_theta = P.ellT / ((P.m1 * P.ell1**2) + (P.m2 * P.ell2**2) + P.J1y + P.J2y)
        self.kp_pitch = wn_pitch**2 / b_theta
        self.kd_pitch = (2 * zeta_pitch * wn_pitch) / b_theta
        
        # Print gains to terminal
        print('kp_pitch: ', self.kp_pitch)
        print('kd_pitch: ', self.kd_pitch)
        
        # Sample rate of the controller
        self.Ts = P.Ts
        
        # Delayed variables
        self.theta_d1 = 0.0  # Previous pitch angle
        self.theta_dot = 0.0  # Filtered derivative of pitch angle

    def update(self, r: np.ndarray, y: np.ndarray):
        # desired theta
        theta_ref = r[0][0]

        # measured theta
        theta = y[1][0]
        
        # Compute derivative
        theta_dot = (theta - self.theta_d1) / self.Ts        

        # compute feedback force
        force_fl = ((P.m1 * P.ell1) + (P.m2 * P.ell2)) * (P.g / P.ellT) * np.cos(theta)

        # Compute control force
        force_unsat = (self.kp_pitch * (theta_ref - theta)) - (self.kd_pitch * theta_dot)
        total_force = force_fl + force_unsat
        force = saturate(total_force, -P.force_max, P.force_max)
        

        # Zero torque to prevent unwanted rotation in different planes
        torque = 0.0
        

        # Convert force and torque to PWM signals
        pwm = np.array([[force + torque / P.d],
                        [force - torque / P.d]]) / (2 * P.km)
        
        pwm = saturate(pwm, 0, 1)
        
        # Update theta reference
        self.theta_d1 = theta
        
        # Return PWM signals and reference signals
        return pwm, np.array([[0.0], [theta_ref], [0.0]])

def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u