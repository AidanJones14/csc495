
import numpy as np
import control as cnt
import blockbeamParam as P

class ctrlStateFeedback:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design with Integrator
        #--------------------------------------------------
        # Tuning parameters
        tr_z = 1.2
        tr_th = 0.25
        zeta_z = 0.707
        zeta_th = 0.707
        integrator_pole = -3.0  # slower integrator pole

        # State Space model
        A = np.array([[0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0],
                      [0.0, -P.g, 0.0, 0.0],
                      [-P.m1 * P.g / ((P.m2 * P.length ** 2) / 3.0 + P.m1 * (P.length / 2.0) ** 2), 0.0, 0.0, 0.0]])

        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [P.length / (P.m2 * P.length ** 2 / 3.0 + P.m1 * (P.length / 2.0) ** 2)]])

        C = np.array([[1.0, 0.0, 0.0, 0.0]])  # Output z for integration

        # Augmented system for integrator
        A1 = np.block([
            [A, np.zeros((4, 1))],
            [-C, np.zeros((1, 1))]
        ])
        B1 = np.vstack((B, [[0.0]]))

        # Desired poles
        wn_th = 2.2 / tr_th
        wn_z = 2.2 / tr_z
        des_char_poly = np.convolve(
            np.convolve([1, 2 * zeta_th * wn_th, wn_th ** 2],
                        [1, 2 * zeta_z * wn_z, wn_z ** 2]),
            [1, -integrator_pole]
        )
        des_poles = np.roots(des_char_poly)

        # Check controllability and place poles
        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 5:
            print("System not controllable")
        else:
            K1 = cnt.acker(A1, B1, des_poles)
            self.K = K1[0, :4]
            self.ki = K1[0, 4]

        self.xi = 0.0  # integrator state
        self.limit = 5.0  # saturation limit

    def update(self, z_r, x):
        z = x[0]
        error = z_r - z
        u_unsat = -self.K @ x - self.ki * self.xi
        u = np.clip(u_unsat.item(), -self.limit, self.limit)

        # Anti-windup condition
        if np.abs(u) < self.limit or np.sign(u) == np.sign(u_unsat):
            self.xi += P.Ts * error

        return u
