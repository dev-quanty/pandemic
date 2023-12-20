import matplotlib.pyplot as plt
import numpy as np
from ODESolver import solve
from models import *


# Test SIR Model
y0 = np.array([0.99999873, 1.27e-6, 0])
t = np.arange(0, 150)
args = (1 / 2, 1 / 3)
method = "RK4"
result = solve(sir, y0, t, method, args)

fig, ax = plt.subplots()
ax.set_title("SIR\n" + method)
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Ratio")
ax.plot(t, result[:, 0], label="Susceptibles")
ax.plot(t, result[:, 1], label="Infected")
ax.plot(t, result[:, 2], label="Removed")
ax.legend()
plt.show()


# Test SEIRQHFD Model
y0 = np.array([0.99999873, 1.27e-6, 0, 0, 0, 0, 0, 0, 0])
t = np.arange(0, 200)
args = (1/2, 1/2, 1/8, 1/3, 1/10, 1/10, 1/10, 1/7, 1/7, 1/14, 1/2, 1/2, 2/3, 1/3)
method = "RK4"
#result = solve(seirqhfd, y0, t, method, args)

# Test Radau Method
y0 = np.array([0.99999873, 1.27e-6, 0, 0, 0, 0, 0, 0, 0])
t = np.arange(0, 200)
args = (1 / 2, 1 / 3)
method = "Radau"
result = solve(sir, y0, t, method, args)
print(result)


fig, ax = plt.subplots()
ax.set_title("SEIRQHFD\n" + method)
ax.set_xlabel("Time (Days)")
ax.set_ylabel("Ratio")
ax.plot(t, result[:, 0], label="Susceptibles")
ax.plot(t, result[:, 2], label="Infected")
ax.plot(t, result[:, 4], label="QuarantineInfected")
ax.plot(t, result[:, 5], label="Hospital")
ax.plot(t, result[:, 6], label="Recovered")
ax.plot(t, result[:, 8], label="Buried")
ax.legend()
plt.show()
