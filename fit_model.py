import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.optimize
from scipy.integrate import odeint

## read data

time = []
S_pop = []
E_pop = []
I_pop = []

with open("population_data.csv") as file:
    reader = csv.reader(file, delimiter=',')

    # skip header
    next(reader)

    for row in reader:
        time.append(float(row[0]))
        S_pop.append(float(row[1]))
        E_pop.append(float(row[2]))
        I_pop.append(float(row[3]))

f, (ax1, ax2, ax3) = plt.subplots(3)

line1 = ax1.scatter(time, S_pop, c="b")
line2 = ax2.scatter(time, E_pop, c="r")
line3 = ax3.scatter(time, I_pop, c="y")

ax1.set_ylabel("S")
ax2.set_ylabel("E")
ax2.set_ylabel("I")
ax3.set_xlabel("Time")


#
# plt.show()


def sim(variables, t, params):
    # set parameters
    i = 0.35
    h_s = 0.1
    f = 0.2
    e = 0.2
    y_i = 0.02

    # Compartments
    S = variables[0]
    E = variables[1]
    I = variables[2]
    # R = variables[3]
    # D = variables[4]

    k_i = params[0]
    k_h = params[1]
    k_f = params[2]
    d = params[3]

    dsdt = -k_i * i * S * I
    dedt = k_i * i * S * I - e * E
    didt = e * E - y_i * I

    return ([dsdt, dedt, didt])


def loss_function(params, time, S_pop, E_pop, I_pop):
    y0 = [S_pop[0], E_pop[0], I_pop[0]]

    t = np.linspace(time[0], time[-1], num=len(time))

    output = odeint(sim, y0, t, args=(params,))

    loss = 0

    for i in range(len(time)):
        data_S = S_pop[i]
        model_S = output[i, 0]

        data_E = E_pop[i]
        model_E = output[i, 1]

        data_I = E_pop[i]
        model_I = output[i, 2]

        res = (data_S - model_S) ** 2 + (data_E - model_E) ** 2 + (data_I - model_I) ** 2

        loss += res

    return (loss)


params0 = np.array([1, 1, 1, 1])
minimum = scipy.optimize.fmin(loss_function, params0, args=(time, S_pop, E_pop, I_pop))

print(minimum)

k_i_fit = minimum[0]
k_h_fit = minimum[1]
k_f_fit = minimum[2]
d_fit = minimum[3]

params = [k_i_fit, k_h_fit, k_f_fit, d_fit]

y0 = [S_pop[0], E_pop[0], I_pop[0]]

t = np.linspace(time[0], time[-1], num=1000)

output = odeint(sim, y0, t, args=(params,))

line1, = ax1.plot(t, output[:, 0], color="b")
line2, = ax2.plot(t, output[:, 1], color="r")
line3, = ax3.plot(t, output[:, 2], color="y")

ax1.set_ylabel("S")
ax2.set_ylabel("E")
ax3.set_ylabel("I")
ax3.set_xlabel("Time")

plt.show()
