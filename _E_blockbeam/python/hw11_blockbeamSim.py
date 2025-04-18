import matplotlib.pyplot as plt
import blockbeamParam as P
from signalGenerator import signalGenerator
from blockbeamAnimation import blockbeamAnimation
from dataPlotter import dataPlotter
from blockBeamDynamics import blockbeamDynamics
from ctrlStateFeedback_mod2 import ctrlStateFeedback
#from ctrlStateFeedback_mod import ctrlStateFeedback
#from ctrlStateFeedback import ctrlStateFeedback

# instantiate blockbeam, controller, and reference classes
blockbeam = blockbeamDynamics()
controller = ctrlStateFeedback()
reference = signalGenerator(amplitude=0.125, frequency=0.05, y_offset=0.25)
disturbance = signalGenerator(amplitude=0.25, frequency=0.0)

# instantiate the simulation plots and animation
dataPlot = dataPlotter()
animation = blockbeamAnimation()
t = P.t_start  
y = blockbeam.h()  

while t < P.t_end: # main simulation loop
    # Propagate dynamics in between plot samples
    t_next_plot = t + P.t_plot

    while t < t_next_plot:  
        r = reference.square(t)  
        d = 0.0  #disturbance.step(t)  
        n = 0.0  #noise.random(t)  
        x = blockbeam.state
        u = controller.update(r, x)  
        y = blockbeam.update(u + d)  
        t += P.Ts  

    # update animation and data plots
    animation.update(blockbeam.state)
    dataPlot.update(t, blockbeam.state, u, r)
    plt.pause(0.0001)  

# Keeps the program from closing until the user presses a button.
print('Press key to close')
plt.waitforbuttonpress()
plt.close()
