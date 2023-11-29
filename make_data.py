import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.integrate import odeint

# define starting population
y0 = [10, 1, 0, 0, 0, 0, 0, 0, 0]

t = np.linspace(0, 100, num=1001)

# variables that our fitting should approximate
k_i = 0.4
k_h = 0.1
k_f = 0.1
d = 0.4

params = [k_i, k_h, k_f, d]


def sim(variables, t, params):
    # set parameters - should be calculated with real-world Data
    i = 0.35
    h_s = 0.1
    f = 0.2
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

    dsdt = -k_i * i * S * I - k_h * h_s * S * H - k_f * f * S * F
    dedt = k_i * i * S * I + k_h * h_s * S * H + k_f * f * S * F - e * E - q_e * E
    didt = e * E - y_i * I - h * I - q_i * I
    dq_edt = q_e * E - e * Q_e
    dqdt = e * Q_e + q_i * I - y_q * Q - h * Q
    dhdt = h * I + h * Q - y_h * H
    dfdt = p_i * y_i * I + p_q * y_q * Q + p_h * y_h * H - d * F
    drdt = (1 - p_i) * y_i * I + (1 - p_q) * y_q * Q + (1 - p_h) * y_h * H
    dddt = d * F

    return ([dsdt, dedt, didt, dq_edt, dqdt, dhdt, dfdt, drdt, dddt])


y = odeint(sim, y0, t, args=(params,))

data_t = []
data_S = []
data_E = []
data_I = []
data_Q_e = []
data_Q = []
data_H = []
data_F = []
data_R = []
data_D = []

for i in range(t.shape[0]):
    print(t[i])
    print(y[i, 0])
    print(y[i, 1])
    print(y[i, 2])
    print(y[i, 3])
    print(y[i, 4])
    print(y[i, 5])
    print(y[i, 6])
    print(y[i, 7])
    print(y[i, 8])
    print()

    if i % 20 == 0:
        data_t.append(t[i])
        data_S.append(np.random.normal(loc=y[i, 0], scale=0.2, ))
        data_E.append(np.random.normal(loc=y[i, 1], scale=0.2, ))
        data_I.append(np.random.normal(loc=y[i, 2], scale=0.2, ))
        data_Q_e.append(np.random.normal(loc=y[i, 3], scale=0.2, ))
        data_Q.append(np.random.normal(loc=y[i, 4], scale=0.2, ))
        data_H.append(np.random.normal(loc=y[i, 5], scale=0.2, ))
        data_F.append(np.random.normal(loc=y[i, 6], scale=0.2, ))
        data_R.append(np.random.normal(loc=y[i, 7], scale=0.2, ))
        data_D.append(np.random.normal(loc=y[i, 8], scale=0.2, ))

print(y.shape)
print(t.shape)

for row in data_t:
    print(row)

# making CSV
f = open('population_data.csv', "w")

f.write("year,data_S,data_E,data_I\n")

for i in range(len(data_t)):
    f.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (data_t[i], data_S[i], data_E[i], data_I[i], data_Q_e[i], data_Q[i],
                                                 data_H[i], data_F[i], data_R[i], data_D[i]))

f.close()

# Visualize
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9)

line1 = ax1.scatter(data_t, data_S, c="b")
line2 = ax2.scatter(data_t, data_E, c="r")
line3 = ax3.scatter(data_t, data_I, c="y")
line4 = ax4.scatter(data_t, data_Q_e, c="c")
line5 = ax5.scatter(data_t, data_Q, c="m")
line6 = ax6.scatter(data_t, data_H, c="pink")
line7 = ax7.scatter(data_t, data_F, c="grey")
line8 = ax8.scatter(data_t, data_R, c="g")
line9 = ax9.scatter(data_t, data_D, c="k")

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
plt.savefig("population_data.pdf")

plt.show()
