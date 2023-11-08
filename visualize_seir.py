from seird import SEIRD
import matplotlib.pyplot as plt

# Define initial conditions
# One needs some infection in the model in order to model the spread
init_conditions = [0.99, 0.01, 0, 0, 0]
model = SEIRD(init_conditions)

# Set parameters
# Basic Reproduction Rate (R0)
R0 = 1.5

# Incubation Period (in days)
i = 3

# Recovery Rate (as percentage)
r = 0.8

# Mortality Rate
theta = 0.7

# Time Stepsize for model
dt = 1

# Run model simulation steps
df = model.simulate(R0, i, r, theta, dt)

# Visualize results
plt.figure()
df[["Susceptible", "Exposed", "Infected", "Recovered", "Dead"]].plot()
plt.legend(loc="best")
plt.show()
