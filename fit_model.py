import matplotlib.pyplot as plt
import numpy as np
import csv
import scipy.optimize
from scipy.integrate import odeint

# read data
time = []
S_pop = []
E_pop = []
I_pop = []
Q_e_pop = []
Q_pop = []
H_pop = []
F_pop = []
R_pop = []
D_pop = []

# add data to population arrays
with open("population_data.csv") as file:
    reader = csv.reader(file, delimiter=',')

    # skip header
    next(reader)

    for row in reader:
        time.append(float(row[0]))
        S_pop.append(float(row[1]))
        E_pop.append(float(row[2]))
        I_pop.append(float(row[3]))
        Q_e_pop.append(float(row[4]))
        Q_pop.append(float(row[5]))
        H_pop.append(float(row[6]))
        F_pop.append(float(row[7]))
        R_pop.append(float(row[8]))
        D_pop.append(float(row[9]))

# define Plots
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9)

line1 = ax1.scatter(time, S_pop, c="b")
line2 = ax2.scatter(time, E_pop, c="r")
line3 = ax3.scatter(time, I_pop, c="y")
line4 = ax4.scatter(time, Q_e_pop, c="c")
line5 = ax5.scatter(time, Q_pop, c="m")
line6 = ax6.scatter(time, H_pop, c="pink")
line7 = ax7.scatter(time, F_pop, c="grey")
line8 = ax8.scatter(time, R_pop, c="g")
line9 = ax9.scatter(time, D_pop, c="k")

ax1.set_ylabel("S")
ax2.set_ylabel("E")
ax3.set_ylabel("I")
ax4.set_ylabel("Q_e")
ax5.set_ylabel("Q")
ax6.set_ylabel("H")
ax7.set_ylabel("F")
ax8.set_ylabel("R")
ax9.set_ylabel("D")
ax9.set_xlabel("Time")


# Simulation method
def sim(variables, t, params):
    # set parameters - should be calculated with real-world Data
    e = 0.2
    q_e = 0.2
    q_i = 0.2
    y_i = 0.02
    y_q = 0.02
    y_h = 0.02
    h = 0.1
    p_i = p_q = p_h = 0.8

    # Compartments
    S = variables[0]
    E = variables[1]
    I = variables[2]
    Q_e = variables[3]
    Q = variables[4]
    H = variables[5]
    F = variables[6]
    R = variables[7]
    D = variables[8]

    k_i = params[0]
    k_h = params[1]
    k_f = params[2]
    d = params[3]

    # the ODE
    dsdt = -k_i * S * I - k_h * S * H - k_f * S * F
    dedt = k_i * S * I + k_h * S * H + k_f * S * F - e * E - q_e * E
    didt = e * E - y_i * I - h * I - q_i * I
    dq_edt = q_e * E - e * Q_e
    dqdt = e * Q_e + q_i * I - y_q * Q - h * Q
    dhdt = h * I + h * Q - y_h * H
    dfdt = p_i * y_i * I + p_q * y_q * Q + p_h * y_h * H - d * F
    drdt = (1 - p_i) * y_i * I + (1 - p_q) * y_q * Q + (1 - p_h) * y_h * H
    dddt = d * F

    return ([dsdt, dedt, didt, dq_edt, dqdt, dhdt, dfdt, drdt, dddt])


# Calculate loss for parameter estimation
def loss_function(params, time, S_pop, E_pop, I_pop, Q_e_pop, Q_pop, H_pop, F_pop, R_pop, D_pop):
    y0 = [S_pop[0], E_pop[0], I_pop[0], Q_e_pop[0], Q_pop[0], H_pop[0], F_pop[0], R_pop[0], D_pop[0]]

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

        data_Q_e = S_pop[i]
        model_Q_e = output[i, 3]

        data_Q = E_pop[i]
        model_Q = output[i, 4]

        data_H = E_pop[i]
        model_H = output[i, 5]

        data_F = S_pop[i]
        model_F = output[i, 6]

        data_R = E_pop[i]
        model_R = output[i, 7]

        data_D = E_pop[i]
        model_D = output[i, 8]

        res = abs(data_S - model_S) + abs(data_E - model_E) + abs(data_I - model_I) \
              + abs(data_Q_e - model_Q_e) + abs(data_Q - model_Q) + abs(data_H - model_H) \
              + abs(data_F - model_F) + abs(data_R - model_R) + abs(data_D - model_D)

        loss += res

    return (loss)


# optimize
params0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
minimum = scipy.optimize.fmin(loss_function, params0,
                              args=(time, S_pop, E_pop, I_pop, Q_e_pop, Q_pop, H_pop, F_pop, R_pop, D_pop))

# plot curves
k_i_fit = minimum[0]
k_h_fit = minimum[1]
k_f_fit = minimum[2]
d_fit = minimum[3]
print("k_i = " + str(k_i_fit))
print("k_h = " + str(k_h_fit))
print("k_f = " + str(k_f_fit))
print("d = " + str(d_fit))

params = [k_i_fit, k_h_fit, k_f_fit, d_fit]

y0 = [S_pop[0], E_pop[0], I_pop[0], Q_e_pop[0], Q_pop[0], H_pop[0], F_pop[0], R_pop[0], D_pop[0]]

t = np.linspace(time[0], time[-1], num=1000)

output = odeint(sim, y0, t, args=(params,))

line1, = ax1.plot(t, output[:, 0], color="b")
line2, = ax2.plot(t, output[:, 1], color="r")
line3, = ax3.plot(t, output[:, 2], color="y")
line4, = ax4.plot(t, output[:, 3], color="c")
line5, = ax5.plot(t, output[:, 4], color="m")
line6, = ax6.plot(t, output[:, 5], color="pink")
line7, = ax7.plot(t, output[:, 6], color="grey")
line8, = ax8.plot(t, output[:, 7], color="g")
line9, = ax9.plot(t, output[:, 8], color="k")

ax1.set_ylabel("S")
ax2.set_ylabel("E")
ax3.set_ylabel("I")
ax4.set_ylabel("Q_e")
ax5.set_ylabel("Q")
ax6.set_ylabel("H")
ax7.set_ylabel("F")
ax8.set_ylabel("R")
ax9.set_ylabel("D")
ax3.set_xlabel("Time")

plt.show()
