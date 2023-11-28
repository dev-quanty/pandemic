import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# define starting population
y0 = [10, 1, 0, 0, 0]

t = np.linspace(0, 100, num=1001)

# variables that our fitting should approximate
k_i = 1.1
k_h = 0.4
k_f = 0.1
d = 0.4

params = [k_i, k_h, k_f, d]


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
    R = variables[3]
    D = variables[4]

    k_i = params[0]
    k_h = params[1]
    k_f = params[2]
    d = params[3]

    dsdt = -k_i * i * S * I
    dedt = k_i * i * S * I - e * E
    didt = e * E - y_i * I

    return ([dsdt, dedt, didt, 0, 0])


y = odeint(sim, y0, t, args=(params,))

data_t = []
data_S = []
data_E = []
data_I = []

for i in range(t.shape[0]):
    print(t[i])
    print(y[i, 0])
    print(y[i, 1])
    print(y[i, 2])
    print()

    if i % 20 == 0:
        data_t.append(t[i])
        data_S.append(np.random.normal(loc=y[i, 0], scale=0.2, ))
        data_E.append(np.random.normal(loc=y[i, 1], scale=0.2, ))
        data_I.append(np.random.normal(loc=y[i, 1], scale=0.2, ))

print(y.shape)
print(t.shape)

for row in data_t:
    print(row)

# sys.exit()

f, (ax1, ax2, ax3) = plt.subplots(3)

line1 = ax1.scatter(data_t, data_S, c="b")
line2 = ax2.scatter(data_t, data_E, c="r")
line3 = ax3.scatter(data_t, data_I, c="y")

ax1.set_ylabel("S")
ax2.set_ylabel("E")
ax3.set_ylabel("I")
ax3.set_xlabel("Time (years)")
plt.savefig("population_data.pdf")

plt.show()

# making CSV
f = open('population_data.csv', "w")

f.write("year,data_S,data_E,data_I\n")

for i in range(len(data_t)):
    f.write("%s,%s,%s,%s\n" % (data_t[i], data_S[i], data_E[i], data_I[i]))

f.close()
