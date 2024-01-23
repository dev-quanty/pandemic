import numpy as np
from ODESolver import solve
from models import sir, seird, seirqhfd
import matplotlib.pyplot as plt


def solve_model(args, method):
    y0 = np.array([0.999, 0.005, 0, 0, 0, 0, 0, 0, 0])
    t_initial = np.arange(0, 30)
    result_initial = solve(seirqhfd, y0, t_initial, method, args=(0.10, 0.16, 0.54, 0.10, 0.00003, 0.000003,
                                                                  0.14, 0.06, 0.06, 0.67, 0.67, 0.52))
    y0 = result_initial[-1, :]
    t_method = np.arange(30, 400)
    result_method = solve(seirqhfd, y0, t_method, method, args)

    result = np.concatenate((result_initial, result_method), axis=0)
    t = np.concatenate((t_initial, t_method))
    return t, result


def plot_model(args, title, method='impliciteuler'):
    y0 = np.array([0.999, 0.005, 0, 0, 0, 0, 0, 0, 0])
    t_initial = np.arange(0, 30)
    result_initial = solve(seirqhfd, y0, t_initial, method, args)

    y0 = result_initial[-1, :]
    t_method = np.arange(30, 400)
    result_method = solve(seirqhfd, y0, t_method, method, args)

    result = np.concatenate((result_initial, result_method), axis=0)
    t = np.concatenate((t_initial, t_method))

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Ratio")
    ax.plot(t, result[:, 0], label="Susceptibles")
    ax.plot(t, result[:, 2], label="Infected")
    ax.plot(t, result[:, 4], label="QuarantineInfected")
    ax.plot(t, result[:, 5], label="Hospital")
    ax.plot(t, result[:, 6], label="Recovered")
    ax.plot(t, result[:, 8], label="Buried")
    ax.axvline(x=t_method[0], color='gray', linestyle='--', label='Start of Extended Phase')
    ax.legend()
    plt.show()

def compare_scenarios(scenarios, method='impliciteuler'):
    fig, ax = plt.subplots()

    for scenario in scenarios:
        args, label = scenario['args'], scenario['label']
        t, result = solve_model(args, method)
        ax.plot(t, result[:, 2], label=label)  # Plotting only the infected for simplicity

    ax.set_title("Comparison of Different Scenarios")
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Ratio of Infected")
    ax.axvline(x=30, color='gray', linestyle='--', label='Start of Extended Phase')
    ax.legend()
    plt.show()


# Define scenarios
ki = 0.10
kh = 0.16
kf = 0.54
e = 0.1
qe = 0.00003
qi = qe
h = 0.14
yi = 0.06
yh = yi
pi = 0.67
ph = pi
d = 0.52
scenarios = [
    {'args': (ki * 1.5, kh * 1.5, kf * 1.5, e, qe, qi, h, yi, yh, pi, ph, d), 'label': 'Poor Hygiene'},
    {'args': (ki, kh, kf, e, qe, qi, h, yi, yh, pi, ph, d), 'label': 'Moderate Hygiene'},
    {'args': (ki - 0.02, kh * 0.5, kf * 0.5, e, qe, qi, h, yi, yh, pi, ph, d), 'label': 'Good Hygiene'}
]

compare_scenarios(scenarios)
