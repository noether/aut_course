from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

# Prey - Predator

# r number of rabits
# f number of foxes
# u_r we kill o introduce rabbits
# u_f we kill o introduce foxes

# \dot r = a r + b rf + u_r
# \dot f = c f + d rf + u_f


# Constant parameters

a = 1.1 # [1/sec]
b = -0.1 # [1/ (sec * fox)]
c = -0.5 # [1 / sec]
d =  0.4 # [1 / (sec * rabbit)]

# Initial conditions
r = 10.0 # [rabbits]
f = 10.0 # [foxes]

X = np.array([r, f]) # Construct the state-vector

# Simulation parameters
Tf = 100 # Final time [s]
DT = 0.005 # Step time [s]

time = np.linspace(0, Tf, Tf/DT) # Time vector for plotting
time_log = np.append(time, Tf+DT) # For the last iteration
it = 0 # Iteration index for logging
plot_frame = 100 # We plot the state every plot_frame iterations

# Data log
X_h = np.zeros((time_log.size, X.size)) # We will store the values of the state vector here

X_h[it,:] = X # We log the first iteration before the loop starts

pl.close("all")
pl.ion()
fig, ax = pl.subplots()

for t in time:

    # Control from linearization
    # A = 
    # B = 
    # C = 
    # D = 
    # K =   # Controller

    u_r = 0 # Control actions
    u_f = 0

    dot_r = a*X[0] + b*X[1]*X[0] + u_r
    dot_f = c*X[1] + d*X[1]*X[0] + u_f

    X[0] = X[0] + dot_r*DT # Euler integration
    X[1] = X[1] + dot_f*DT # Euler integration

    # Animation
    if it%plot_frame == 0:
        ax.clear()
        ax.plot(time[0:it], X_h[0:it,0], 'r')
        ax.plot(time[0:it], X_h[0:it,1], 'b')

        ax.grid()

        pl.pause(0.001)
        pl.show()

    # Next iteration index
    it+=1

    # Log
    X_h[it,:] = X

pl.figure(2)
pl.plot(time_log, X_h[:, 0], label="rabbits")
pl.xlabel("Time [s]")
pl.ylabel("# rabbits")
pl.grid()

pl.figure(3)
pl.plot(time_log, X_h[:, 1], label="foxes")
pl.xlabel("Time [s]")
pl.ylabel("# foxes")
pl.grid()


