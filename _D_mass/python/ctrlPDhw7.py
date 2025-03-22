import numpy as np
import massParam as P


class ctrlPD:
    def __init__(self):
        # PD gains
        self.kp = 4.5
        self.kd = 12.0
        print('kp: ', self.kp)
        print('kd: ', self.kd)

    def update(self, z_r, state):
        z = state[0][0]
        zdot = state[1][0]
        tau_tilde = self.kp * (z_r - z) - self.kd * zdot
        tau = saturate(tau_tilde, P.F_max)
        return tau


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u







