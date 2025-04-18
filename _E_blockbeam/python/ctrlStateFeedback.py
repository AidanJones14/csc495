import numpy as np
import blockbeamParam as P
import control as cnt


class ctrlStateFeedback:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # tuning parameters
        tr_z = 1.2  # rise time for position
        tr_theta = 0.25  # rise time for angle
        zeta_z = 0.707  # damping ratio position
        zeta_th = 0.707  # damping ratio angle
        # State Space Equations
        # xdot = A*x + B*u
        # y = C*x
        A = np.array([[0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 1.0],
              [0.0, -P.g, 0.0, 0.0],
              [-P.m1 * P.g / ((P.m2 * P.length ** 2) / 3.0 + \
                    P.m1 * (P.length / 2.0) ** 2), 0.0, 0.0, 0.0]])
        
        B = np.array([[0.0],
                      [0.0],
                      [0.0],
                      [P.length / (P.m2 * P.length ** 2 / 3.0 + \
                           P.m1 * P.length ** 2 / 4.0)]])
        
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, 0.0]])

        # gain calculation
        wn_th = 2.2 / tr_theta  # natural frequency for angle
        wn_z = 2.2 / tr_z  # natural frequency for position
        des_char_poly = np.convolve(
                [1, 2 * zeta_z * wn_z, wn_z ** 2],
                [1, 2 * zeta_th * wn_th, wn_th ** 2])
        des_poles = np.roots(des_char_poly)

        # Compute the gains if the system is controllable
        if np.linalg.matrix_rank(cnt.ctrb(A, B)) != 4:
            print("The system is not controllable")
        else:
            self.K = cnt.acker(A, B, des_poles)
            Cr = np.array([[1.0, 0.0, 0.0, 0.0]])
            self.kr = -1.0 / (Cr @ np.linalg.inv(A - B @ self.K) @ B)
        print('K: ', self.K)
        print('kr: ', self.kr)
        self.Ts = P.Ts  # sample rate of controller

    def update(self, z_r, x):
        z = x[0,0]
        x_tilde = x - np.array([[P.ze], [0], [0], [0]])
        zr_tilde = z_r - P.ze # P.ze is the same as C_r*x_e 

        # equilibrium force
        F_e = P.m1*P.g*(P.ze/P.length) + P.m2*P.g/2.0

        # Compute the state feedback controller
        F_tilde = -self.K @ x_tilde + self.kr * zr_tilde
        F_unsat = F_e + F_tilde
        F = saturate(F_unsat[0,0], P.F_max)

        return F


def saturate(u, limit):
    if abs(u) > limit:
        u = limit * np.sign(u)
    return u

