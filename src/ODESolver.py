from functools import partial
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve
from autograd import make_jvp, jacobian


def solve(func, y0, t, method="RK4", args=(), **kwargs):
    """
    Solve the ODE-System given by the differential equation system in f.

    Args:
        func: Function implementing the differential equation system with function call func(y, t, args)
        y0: Vector of initial values
        t: Vector of time points in ascending order in equidistant steps
        method: Method to solve the ODE-System. Default is 'RK4'
        args: Arguments used in the ODE-System
        kwargs: Further arguments for the solve method

    Returns:
        result: Path of given variables at time steps t
    """
    ode_fun = lambda t, y: func(y, t, args)
    t0 = t[0]
    t_bound = t[-1]

    if method.lower() == "expliciteuler":
        return forwardEuler(func, y0, t, args, **kwargs)
    elif method.lower() == "impliciteuler":
        return backwardEuler(func, y0, t, args, **kwargs)
    elif method.lower() == "rk4":
        return RK4(func, y0, t, args, **kwargs)
    elif method in ["Radau", "BDF", "LSODA", "DOP853", "RK45", "RK23"]:
        return solve_ivp(ode_fun, (t0, t_bound), y0, method=method, t_eval=t).y.T
    else:
        raise ValueError(f"Method {method} not implemented.")
    pass


def forwardEuler(func, y0, t, args, **kwargs):
    result = np.zeros((np.size(t), np.size(y0)))
    result[0] = y0
    dt = t[1] - t[0]
    for i in range(np.size(t) - 1):
        dydt = func(result[i], t[i], args)
        result[i + 1] = result[i] + dt * dydt
    return result


def backwardEuler(func, y0, t, args, **kwargs):
    result = np.zeros((np.size(t), np.size(y0)))
    result[0, :] = y0
    dt = t[1] - t[0]
    for i in range(np.size(t) - 1):
        dydt_eq = lambda y: y - (result[i] + dt * func(y, t[i+1], args))
        result[i + 1] = fsolve(dydt_eq, result[i])
    return result


def RK4(func, y0, t, args, **kwargs):
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
