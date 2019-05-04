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
theta = 0.02 # [rad]
dot_theta = -0.1 # [rad/s]

X = np.array([[theta], [dot_theta]]) # Construct the state-vector

# Operational point
theta_star = 0
theta_dot_star = 0
x_star = np.array([[theta_star],[theta_dot_star]])

u_star = 0

# Simulation parameters
Tf = 30 # Final time [s]
DT = 0.001 # Step time [s]

time = np.linspace(0, Tf, Tf/DT) # Time vector for plotting
time_log = np.append(time, Tf+DT) # For the last iteration
it = 0 # Iteration index for logging
plot_frame = 100 # We plot the state every plot_frame iterations

# Data log
X_h = np.zeros((X.size, time_log.size)) # We will store the values of the state vector here

X_h[:,it] = X.T # We log the first iteration before the loop starts

pl.close("all")
pl.ion()
fig, ax = pl.subplots()

for t in time:
    # Add small disturbances if you want, to check that the controller is working
    # If the disturbance is BIG, then k_12 and k_11 must be also big, or far from the limit conditions of stability, i.e, we need big lambdas :P.
    #X[0][0] = X[0][0] + 0.001*(0.5 - np.random.randn(1))

    # Controller for C = I
    # K = [k11 k12]
    k12 = b - 1
    k11 = -g*l*m - 10    # Stable
    # k11 = -g*l*m + 0.25  # Inestable

    B = np.array([[0],[1]])
    C = np.eye(2)
    K = np.array([k11, k12])
    delta_X = X - x_star
    delta_u = K.dot(C).dot(delta_X) # delta_u = KC delta_X

    T = u_star + delta_u # Control action

    dot_dot_theta = 1.0/(m*l*l) * (m*g*l*X[0][0] - b*X[1][0] + T) # Dynamics

    X[1][0] = X[1][0] + dot_dot_theta*DT # Euler integration
    X[0][0] = X[0][0] + X[1][0]*DT # Euler integration

    # Animation
    if it%plot_frame == 0:
        ax.clear()
        ax.arrow(0, 0, l*np.sin(X[0][0]), l*np.cos(X[0][0]))
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
    X_h[:,it] = X.T

pl.figure(2)
pl.plot(time_log, X_h[0, :], label="$\theta$")
pl.xlabel("Time [s]")
pl.ylabel("Angle [rad]")
pl.grid()

pl.figure(3)
pl.plot(time_log, X_h[0, :], label="$\dot\theta$")
pl.xlabel("Time [s]")
pl.ylabel("Angular velocity [rad/s]")
pl.grid()


