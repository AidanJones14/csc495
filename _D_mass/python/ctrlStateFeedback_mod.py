
import numpy as np
import control as cnt
import massParam as P

class ctrlStateFeedback:
    def __init__(self):
        #  tuning parameters
        tr = 6.0
        zeta = 0.707
        integrator_pole = -2.0  # Place integrator pole far left

        # System matrices
        A = np.array([[0.0, 1.0],
                      [-P.k / P.m, -P.b / P.m]])
        B = np.array([[0.0],
                      [1.0 / P.m]])
        C = np.array([[1.0, 0.0]])
        self.C = C  # Save for later use

        # Augment system with integrator
        A1 = np.block([
            [A, np.zeros((2,1))],
            [-C, np.zeros((1,1))]
        ])
        B1 = np.vstack((B, [[0.0]]))

        des_char_poly = np.convolve([1, 2*zeta*2.2/tr, (2.2/tr)**2],
                                    [1, -integrator_pole])
        des_poles = np.roots(des_char_poly)

        if np.linalg.matrix_rank(cnt.ctrb(A1, B1)) != 3:
            print("The augmented system is not controllable")
        else:
            K1 = cnt.acker(A1, B1, des_poles)
            self.K = K1[0, :2]
            self.ki = K1[0, 2]

        self.xi = 0.0  # Integrator state
        self.u_prev = 0.0
        self.limit = 5.0  # Anti-windup actuator limit

    def update(self, z_r, x):
        z = self.C @ x
        error = z_r - z
        u_unsat = -self.K @ x - self.ki * self.xi
        u = np.clip(u_unsat, -self.limit, self.limit)

        # Anti-windup: only integrate error if not saturated
        if np.abs(u) < self.limit or np.sign(u) == np.sign(u_unsat):
            self.xi += P.Ts * error.item()

        self.u_prev = u
        return u.item()

