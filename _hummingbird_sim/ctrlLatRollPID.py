import numpy as np
import hummingbirdParam as P  # Import parameters for the hummingbird

class ctrlLatRollPID:
    def __init__(self):
        # Design parameters for the roll loop
        tr_phi = 0.1  
        zeta_phi = 0.707  
        
        # Calculate natural frequency for roll
        wn_phi = 2.2 / tr_phi
        
        # Calculate PD gains for roll
        self.kp_phi = P.J1x * wn_phi**2  
        self.kd_phi = 2 * zeta_phi * wn_phi * P.J1x  
        
        # Print gains to terminal (for debugging)
        print('kp_phi: ', self.kp_phi)
        print('kd_phi: ', self.kd_phi)
        
        # Sample rate of the controller
        self.Ts = P.Ts
        
        # Delayed variables for derivative calculation
        self.phi_d1 = 0.0  
        self.phi_dot = 0.0  

    def update(self, phi_ref, phi):
        # Compute derivative of roll angle
        phi_dot = (phi - self.phi_d1) / self.Ts
        
        # Compute roll torque using PD control
        tau_phi = (self.kp_phi * (phi_ref - phi)) - (self.kd_phi * phi_dot)
        
        # Saturate the roll torque to stay within limits
        tau_phi = saturate(tau_phi, -P.torque_max, P.torque_max)
        
        # Update delayed variables
        self.phi_d1 = phi
        
        return tau_phi

def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u