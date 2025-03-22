import numpy as np
import hummingbirdParam as P

class ctrlEquilibrium:
    def __init__(self):
        pass 

    def update(self, x):

        theta = x[1][0]
        #thetadot = x[4][0]
        
        # u_l_e and u_r_e scale is percent duty cycle [0, 1]
        # A reasonable setting for u_l_e and u_r_e would be that, at equilibrium, we allow for
        # half the throttle response in either direction. Hence, u_l_e = u_r_e = 0.5
        u_l_e = 0.5
        u_r_e = 0.5
        # while this is the correct version of km (see equation above 4.9 in hummingbird manual), 
        # you can also use km from the hummingbirdParam file since it's the same if u_l_e + u_r_e = 1
        km = P.g * (P.m1 * P.ell1 + P.m2 * P.ell2) / (P.ellT * (u_l_e + u_r_e))

        # Calculate force and torque
        #force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g*np.cos(theta)/P.ellT # if theta_e isn't 0
        force_equilibrium = (P.m1*P.ell1 + P.m2*P.ell2)*P.g/P.ellT # if theta_e is 0
        force = force_equilibrium
        torque = 0.

        # Convert to PWM
        # Here you could directly plug in force and torque into the pwm using equations 4.10 and 4.11
        # convert force and torque to pwm signals
        pwm = np.array([[force + torque/P.d],
                       [force - torque/P.d]])/(2*km)
        # Or you could continue to use the mixing matrix (notice the 1/2 difference in denominator)
        #pwm = P.mixing @ np.array([[force], [torque]]) / km
        pwm = saturate(pwm, 0, 1) 
        return pwm


def saturate(u, low_limit, up_limit):
    if isinstance(u, float) is True:
        u = np.max((np.min((u, up_limit)), low_limit))
    else:
        for i in range(0, u.shape[0]):
            u[i][0] = np.max((np.min((u[i][0], up_limit)), low_limit))
    return u




