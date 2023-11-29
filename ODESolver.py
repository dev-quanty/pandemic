import numpy as np
from scipy.optimize import fsolve


def solve(func, y0, t, method="RK4", args=()):
    """
    Solve the ODE-System given by the differential equation system in f.

    Args:
        func: Function implementing the differential equation system with function call func(y, t, args)
        y0: Vector of initial values
        t: Vector of time points in ascending order in equidistant steps
        method: Method to solve the ODE-System. Default is 'RK4'
        args: Arguments used in the ODE-System

    Returns:
        result: Path of given variables at time steps t
    """
    if method.lower() == "expliciteuler":
        return forwardEuler(func, y0, t, args)
    elif method.lower() == "impliciteuler":
        return backwardEuler(func, y0, t, args)
    elif method.lower() == "rk4":
        return RK4(func, y0, t, args)
    else:
        raise ValueError(f"Method {method} not implemented.")
    pass


def forwardEuler(func, y0, t, args):
    result = np.zeros((np.size(t), np.size(y0)))
    result[0] = y0
    dt = t[1] - t[0]
    for i in range(np.size(t) - 1):
        dydt = func(result[i], t[i], args)
        result[i + 1] = result[i] + dt * dydt
    return result


def backwardEuler(func, y0, t, args):
    result = np.zeros((np.size(t), np.size(y0)))
    result[0, :] = y0
    dt = t[1] - t[0]
    for i in range(np.size(t) - 1):
        dydt_eq = lambda y: y - (result[i] + dt * func(y, t[i+1], args))
        result[i + 1] = fsolve(dydt_eq, result[i])
    return result


def RK4(func, y0, t, args):
    result = np.zeros((np.size(t), np.size(y0)))
    result[0] = y0
    dt = t[1] - t[0]
    for i in range(np.size(t) - 1):
        k1 = func(result[i], t[i], args)
        k2 = func(result[i] + (k1 * dt) / 2, t[i] + dt / 2, args)
        k3 = func(result[i] + (k2 * dt) / 2, t[i] + dt/ 2, args)
        k4 = func(result[i] + k3 * dt, t[i] + dt, args)
        result[i + 1] = result[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return result


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    def sir(y, t, args):
        beta, gamma = args
        S, I, R = y
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I
        dRdt = gamma * I
        return np.array([dSdt, dIdt, dRdt])

    y0 = np.array([0.99999873, 1.27e-6, 0])
    t = np.arange(0, 151)
    args = (1/2, 1/3)
    method = "RK4"

    result = solve(sir, y0, t, method, args)

    fig, ax = plt.subplots()
    ax.set_title(method)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Ratio")
    ax.plot(t, result[:, 0], label="Susceptibles")
    ax.plot(t, result[:, 1], label="Infected")
    ax.plot(t, result[:, 2], label="Removed")
    ax.legend()
    plt.show()
