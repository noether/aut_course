from scipy import linalg as la
import matplotlib.pyplot as pl
import numpy as np

# We constrain the theta angle between -pi and pi in the simulation
def normalizeAngle(angle):
    newAngle = angle;
    while newAngle <= -np.pi:
        newAngle = newAngle + 2*np.pi;
    while newAngle > np.pi:
        newAngle = newAngle - 2*np.pi;
    return newAngle;

# Constant parameters

m = 0.5 # Mass [Kg]
l = 2 # Length [m]
g = 9.8 # Gravity acceleration [m/s/s]
b = 1 # Friction coefficient [m*l*l/s]

# Initial conditions
theta = 0.01 # [rad]
dot_theta = 0 # [rad/s]

X = np.array([theta, dot_theta]) # Construct the state-vector

# Simulation parameters
Tf = 300 # Final time [s]
DT = 0.01 # Step time [s]

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

    T = 0 # Control action

    dot_dot_theta = 1.0/(m*l*l) * (m*g*l*X[0] - b*X[1] + T) # Dynamics

    X[1] = X[1] + dot_dot_theta*DT # Euler integration
    X[0] = X[0] + X[1]*DT # Euler integration

    # Animation
    if it%plot_frame == 0:
        ax.clear()
        ax.arrow(0, 0, l*np.sin(X[0]), l*np.cos(X[0]))
        ax.set_xlim(-l*2, l*2)
        ax.set_ylim(-l*2, l*2)

        ax.grid()

        pl.text(-0.7*l*2, 0.8*l*2,\
           "Time %.3f"%t, horizontalalignment='center', \
           verticalalignment='center')
        pl.pause(0.001)
        pl.show()

    # Next iteration index
    it+=1

    # We normalize the angle theta
    X[0] = normalizeAngle(X[0])

    # Log
    X_h[it,:] = X

pl.figure(2)
pl.plot(time_log, X_h[:, 0], label="$\theta$")
pl.xlabel("Time [s]")
pl.ylabel("Angle [rad]")
pl.grid()

pl.figure(3)
pl.plot(time_log, X_h[:, 1], label="$\dot\theta$")
pl.xlabel("Time [s]")
pl.ylabel("Angular velocity [rad/s]")
pl.grid()


