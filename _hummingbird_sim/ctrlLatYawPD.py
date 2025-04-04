import numpy as np
import hummingbirdParam as P 

class ctrlLatYawPD:
    def __init__(self):
        # Design parameters for the yaw loop
        M = 10  
        tr_psi = M * 0.1
        zeta_psi = 0.707
        
        # Calculate natural frequency for yaw
        wn_psi = 2.2 / tr_psi
        
        b_psi = (((P.m1 * P.ell1) + (P.m2 * P.ell2)) * P.g) / ((P.m1 * P.ell1**2) + (P.m2 * P.ell2**2) + P.J2z + (P.m3 * (P.ell3x**2 + P.ell3y**2)) + P.J1z)
        
        # Calculate PD gains for yaw
        self.kp_psi = wn_psi**2 / b_psi
        self.kd_psi = 2 * zeta_psi * wn_psi / b_psi
        
        # Print gains to terminal (for debugging)
        print('kp_psi: ', self.kp_psi)
        print('kd_psi: ', self.kd_psi)
        
        # Sample rate of the controller
        self.Ts = P.Ts
        
        # Delayed variables for derivative calculation
        self.psi_d1 = 0.0  
        self.psi_dot = 0.0  

    def update(self, r: np.ndarray, y: np.ndarray):

        psi_ref = r[1][0]

        psi = y[2][0]

        # Compute derivative of yaw angle
        psi_dot = (psi - self.psi_d1) / self.Ts
        
        # Compute yaw torque using PD control
        tau_psi = (self.kp_psi * (psi_ref - psi)) - (self.kd_psi * psi_dot)
        
        # Saturate the yaw torque to stay within limits
        tau_psi = saturate(tau_psi, -P.torque_max, P.torque_max)
        
        # Update delayed variables
        self.psi_d1 = psi
        
        return tau_psi
    
def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u