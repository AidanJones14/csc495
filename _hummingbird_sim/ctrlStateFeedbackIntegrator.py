import numpy as np
import control as cnt
import hummingbirdParam as P


class ctrlStateFeedbackIntegrator:
    def __init__(self):
        #--------------------------------------------------
        # State Feedback Control Design
        #--------------------------------------------------
        # tuning parameters

        # Compute derived constants
        b_theta = P.ellT / (P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J1y + P.J2y)
        Fe = (P.m1 * P.ell1 + P.m2 * P.ell2) * P.g / P.ellT
        JT = P.m1 * P.ell1**2 + P.m2 * P.ell2**2 + P.J2z + P.m3 * (P.ell3x**2 + P.ell3y**2)
        b_psi = (P.ellT * Fe) / (JT + P.J1z)

        # Store as instance variables
        self.b_theta = b_theta
        self.b_psi = b_psi
        self.Fe = Fe
        self.JT = JT


        # Longitudinal tuning
        wn_th = 2.5          
        zeta_th = 0.7        
        pi_th = -wn_th / 2   

        # Lateral tuning
        wn_phi = 2.2
        zeta_phi = 0.9
  
        wn_psi = 2.3
        zeta_psi = 0.9  
        pi_psi = -wn_psi / 2 

        # Base longitudinal system
        Alon = np.array([[0, 1],
                         [0, 0]])
        Blon = np.array([[0],
                         [self.b_theta]])
        Clon = np.array([[1, 0]])

        # Base lateral system
        Alat = np.array([[0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0],
                         [self.b_psi, 0, 0, 0]])
        Blat = np.array([[0],
                         [0],
                         [1 / P.J1x],
                         [0]])
        Cr_lat = np.array([[0, 1, 0, 0]])

        # Augmented systems
        A1_lon = np.block([
            [Alon, np.zeros((2, 1))],
            [-Clon, np.zeros((1, 1))]
        ])
        B1_lon = np.vstack((Blon, [[0]]))

        A1_lat = np.block([
            [Alat, np.zeros((4, 1))],
            [-Cr_lat, np.zeros((1, 1))]
        ])
        B1_lat = np.vstack((Blat, [[0]]))
        
        # --------------------------------------------------
        # Pole placement using control.place()

        # Desired poles for longitudinal system
        p_lon = np.roots([1, 2 * zeta_th * wn_th, wn_th**2])
        p_lon = np.append(p_lon, pi_th)

        # Desired poles for lateral system
        p_phi = np.roots([1, 2 * zeta_phi * wn_phi, wn_phi**2])  
        p_psi = np.roots([1, 2 * zeta_psi * wn_psi, wn_psi**2])
        p_lat = np.concatenate((p_phi, p_psi, [pi_psi]))

        # Check controllability
        if np.linalg.matrix_rank(cnt.ctrb(A1_lon, B1_lon)) != A1_lon.shape[0]:
            print("Warning: Longitudinal system not controllable")
        if np.linalg.matrix_rank(cnt.ctrb(A1_lat, B1_lat)) != A1_lat.shape[0]:
            print("Warning: Lateral system not controllable")

        # Compute gain vectors
        K1_lon = cnt.place(A1_lon, B1_lon, p_lon)
        K1_lat = cnt.place(A1_lat, B1_lat, p_lat)

        # Fill in the control gains from the computed vectors
        self.k_th = K1_lon[0, 0]
        self.k_thdot = K1_lon[0, 1]
        self.ki_lon = K1_lon[0, 2]

        self.k_phi = K1_lat[0, 0]
        self.k_psi = K1_lat[0, 1]
        self.k_phidot = K1_lat[0, 2]
        self.k_psidot = K1_lat[0, 3]
        self.ki_lat = K1_lat[0, 4]



        # print gains to terminal
        print('K_lon: [', self.k_th, ',', self.k_thdot, ']')
        print('ki_lon: ', self.ki_lon)         
        print('K_lat: [', self.k_phi, ',', self.k_psi, ',', self.k_phidot, ',', self.k_psidot, ']')
        print('ki_lat: ', self.ki_lat)        
        #--------------------------------------------------
        # saturation limits
        theta_max = 30.0 * np.pi / 180.0  # Max theta, rads
        #--------------------------------------------------
        self.Ts = P.Ts
        sigma = 0.05  # cutoff freq for dirty derivative
        self.beta = (2 * sigma - self.Ts) / (2 * sigma + self.Ts)
        self.phi_d1 = 0.
        self.phi_dot = 0.
        self.theta_d1 = 0.
        self.theta_dot = 0.
        self.psi_d1 = 0.
        self.psi_dot = 0.        
        # variables to implement integrator
        self.integrator_th = 0.0  
        self.error_th_d1 = 0.0  
        self.integrator_psi = 0.0  
        self.error_psi_d1 = 0.0 

    def update(self, r: np.ndarray, y: np.ndarray):
        theta_ref = r[0][0]
        psi_ref = r[1][0]
        phi = y[0][0]
        theta = y[1][0]
        psi = y[2][0]
        force_equilibrium = self.Fe   
        # update differentiators
        self.phi_dot = (phi - self.phi_d1) / self.Ts
        self.theta_dot = (theta - self.theta_d1) / self.Ts
        self.psi_dot = (psi - self.psi_d1) / self.Ts

        self.phi_d1 = phi
        self.theta_d1 = theta
        self.psi_d1 = psi

        # integrate error
        error_th = theta_ref - theta
        error_psi = psi_ref - psi
        self.integrator_th += (self.Ts / 2.0) * (error_th + self.error_th_d1)
        self.integrator_psi += (self.Ts / 2.0) * (error_psi + self.error_psi_d1)
        self.error_th_d1 = error_th
        self.error_psi_d1 = error_psi

        # longitudinal control
        force_unsat = self.Fe \
            - self.k_th * theta \
            - self.k_thdot * self.theta_dot \
            - self.ki_lon * self.integrator_th

        force = saturate(force_unsat, -P.force_max, P.force_max)
        # lateral control
        torque_unsat = -self.k_phi * phi \
                - self.k_psi * psi \
                - self.k_phidot * self.phi_dot \
                - self.k_psidot * self.psi_dot \
                - self.ki_lat * self.integrator_psi
 
        torque = saturate(torque_unsat, -P.torque_max, P.torque_max)
        # convert force and torque to pwm signals
        pwm = np.array([[force + torque / P.d],               # u_left
                      [force - torque / P.d]]) / (2 * P.km)   # r_right          
        pwm = saturate(pwm, 0, 1)
        return pwm, np.array([[0], [theta_ref], [psi_ref]])


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        if u > up_limit:
            u = up_limit
        if u < low_limit:
            u = low_limit
    else:
        for i in range(0, u.shape[0]):
            if u[i][0] > up_limit:
                u[i][0] = up_limit
            if u[i][0] < low_limit:
                u[i][0] = low_limit
    return u
