import matplotlib.pyplot as plt
import numpy as np
import hummingbirdParam as P
from signalGenerator import SignalGenerator
from hummingbirdAnimation import HummingbirdAnimation
from dataPlotter import DataPlotter
from hummingbirdDynamics import HummingbirdDynamics
from ctrlEquilibrium import ctrlEquilibrium  

phi_ref = SignalGenerator(amplitude=1.5, frequency=0.05)
theta_ref = SignalGenerator(amplitude=0.5, frequency=0.05)
psi_ref = SignalGenerator(amplitude=0.5, frequency=0.05)

hummingbird = HummingbirdDynamics(alpha=0.0) 

controller = ctrlEquilibrium()

dataPlot = DataPlotter()
animation = HummingbirdAnimation()

t = P.t_start  
while t < P.t_end:  
    t_next_plot = t + P.t_plot  

    while t < t_next_plot:
        u = controller.update(hummingbird.state)  

        y = hummingbird.update(u)

        t += P.Ts

    state = hummingbird.state 
    animation.update(t, state)
    dataPlot.update(t, state, u)

    plt.pause(0.05)  

print('Press key to close')
plt.waitforbuttonpress()
plt.close()