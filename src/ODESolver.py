import autograd.numpy as np
import scipy as sp
from autograd import make_jvp, jacobian
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp



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
    elif method.lower() == "radau":
        return RKV_Randau(func,y0,t,args)
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


def RKV_Randau(func, y0, t, args):
    # Define a wrapper function to include the extra 'args'
    def func_wrapper(t, y):
        return func(y, t, args)

    # Solve the ODE using the Radau method
    sol = solve_ivp(func_wrapper, [t[0], t[-1]], y0, method='Radau', t_eval=t, vectorized=False)

    result = sol.y.T  # Transpose to align with original format
    return result

def IRK(func, y0, t0, te, dt, A, b, c, tol):
    """
    y_{n+1} = y_{n} + h * sum_{i = 1}^{s}(b_{i]*f(Y^i) Y^i approx y(tn+c,h)
    Yi = yn + h * sum über j bis s aijf(y^1j) , i = 1,...s
    unknowns y n+1 , {Y^i} s mal
    s number of stages,bi quadratur wieghts ,aij internal coefficients,ci quadratur points
    butcher tableau
    equation solved with Newton´s method is
    G(Y_i) = Y_i - y_n - h sum_{j = 1}^s a_ij * f(t_n + c_i * h,Y_i) = 0
    G´(Y_i) = I - h sum_{j = 1} aijf´(t_n + c_i * h,Y_i)
    """

    M = 10  # newton iterations
    J = jacobian(func, t0, y0)
    m = len(y0)  # m number of equations
    s = len(b)  # s = number of stages
    stageDer = np.array(s * [func(t0, y0)])
    stageVal = newton_solve(func, J, t0, y0, dt, stageDer, tol, s, m, A, c, M)
    reshaped_stage_values = stageVal.reshape(s, m)
    weighted_sums = []
    for i in range(te):
        component_stage_values = reshaped_stage_values[:, i]
        weighted_sum = np.dot(b, component_stage_values)
        weighted_sums.append(weighted_sum)
    return np.array(weighted_sums)


def newton_solve(func, J, t0, y0, dt, initVal, tol, s, m, A, c, M):
    JJ = np.eye(s, m) - dt * np.kron(A, J)
    LU = sp.linalg.lu_factor(JJ)
    for i in range(M):
        initVal = newton_step(func, t0, y0, initVal, LU)
        if np.norm(initVal) < tol:
            break
        elif i == M - 1:
            raise ValueError("did not converge")
    return initVal


def newton_step(func, t0, y0, dt, initVal, s, m, A, c, lrFactor):
    """
       Takes one Newton step by solving
           G’(Y_i)(Y^(n+1)_i-Y^(n)_i)=-G(Y_i)
       where
           G(Y_i) = Y_i - y_n - h*sum(a_{ij}*Y’_j) for j=1,...,s
       Paramets:
       ---------------
       :param A:
       :param self:
       :param t: timesteps
       :param y0: 1xm vector ,last solution of y_n
       :param initVal: guess for Newton
       :param lrFactor: lr-Zerlegung
       :return:The difference Y^(n+1)_i-Y^(n)_i
       """
    d = sp.linalg.lu_solve(lrFactor, F(func, initVal.flatten(), t0, y0, dt, s, m, A, c))
    return initVal.flatten() + d


def F(func, stageDer, t0, y0, dt, s, m, A, c):
    """
        Returns the subtraction Y’_{i}-f(t_{n}+c_{i}*h, Y_{i}),
        where Y are the stage values, Y’ the stage derivatives and f the function of
        the IVP y’=f(t,y) that should be solved by the RK-method.
    """
    stageDer_new = np.empty((s, m))
    for i in range(s):
        stageVal = y0 + np.array([dt * np.dot(A[i, :], stageDer.reshape(s, m)[:, j]) for j in range(m)])
        stageDer_new[i, :] = func(t0 + c[i] * dt, stageVal)
    return stageDer - stageDer_new.reshape(-1)