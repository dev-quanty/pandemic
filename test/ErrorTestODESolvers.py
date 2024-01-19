import numpy as np
import matplotlib.pyplot as plt
from src.ODESolver import forwardEuler, backwardEuler, RK4
from scipy.linalg import norm


def ode_sys(XY, t, args):
    x = XY[0]
    y = XY[1]
    dx_dt = - x + y
    dy_dt = 4. * x - y
    return np.array([dx_dt, dy_dt])


def sysAnalytical(t_space):
    return np.exp(t_space) + np.exp(-3. * t_space), 2. * np.exp(t_space) - 2. * np.exp(-3. * t_space)


def testODEwith2D():
    # parameters
    t_space = np.linspace(0., 10., 100)
    x_init = 2.
    y_init = 0.

    # analytical solution
    x_an_sol, y_an_sol = sysAnalytical(t_space)

    # function solution
    solRK4 = RK4(ode_sys, np.array([x_init, y_init]), t_space, 0)
    solForEuler = forwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)
    solBackEuler = backwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)

    # Plot solution
    plt.figure()
    plt.plot(t_space, x_an_sol, '--k', linewidth=2, label='analytical x')
    plt.plot(t_space, y_an_sol, '--', color='dimgrey', linewidth=2, label='analytical y')
    plt.plot(t_space, solRK4.T[0], color='darkviolet', linewidth=1, label='RK4 x')
    plt.plot(t_space, solRK4.T[1], color='magenta', linewidth=1, label='RK4 y')
    plt.plot(t_space, solForEuler.T[0], color='darkred', linewidth=1, label='forwardEuler x')
    plt.plot(t_space, solForEuler.T[1], color='crimson', linewidth=1, label='forwardEuler y')
    plt.plot(t_space, solBackEuler.T[0], color='mediumblue', linewidth=1, label='backwardEuler x')
    plt.plot(t_space, solBackEuler.T[1], color='dodgerblue', linewidth=1, label='backwardEuler y')
    plt.title('System of 2 ODEs 1st order solved with different methods')
    plt.xlabel('t')
    plt.ylabel('value of equation')
    plt.legend()
    plt.show()

    # Plot error
    plt.figure()
    plt.plot(t_space, solRK4.T[0] - x_an_sol, color='darkviolet', linewidth=1, label='RK4 x')
    plt.plot(t_space, solRK4.T[1] - y_an_sol, color='magenta', linewidth=1, label='RK4 y')
    plt.plot(t_space, solForEuler.T[0] - x_an_sol, color='darkred', linewidth=1, label='forwardEuler x')
    plt.plot(t_space, solForEuler.T[1] - y_an_sol, color='crimson', linewidth=1, label='forwardEuler y')
    plt.plot(t_space, solBackEuler.T[0] - x_an_sol, color='mediumblue', linewidth=1, label='backwardEuler x')
    plt.plot(t_space, solBackEuler.T[1] - y_an_sol, color='dodgerblue', linewidth=1, label='backwardEuler y')
    plt.title('Error of ODE solvers with System of 2 ODEs 1st order')
    plt.xlabel('t')
    plt.ylabel('value of equation')
    plt.legend()
    plt.show()


def testODEkonsistenz():
    values = [1, 5, 10, 25, 50, 75, 100]
    n = len(values)
    errRK4 = np.zeros(n)
    errForEuler = np.zeros(n)
    errBackEuler = np.zeros(n)
    for i in range(n):
        tau = values[i]

        # parameters
        t_space = np.linspace(0., 10., 10 * tau)
        x_init = 2.
        y_init = 0.

        # analytical solution
        x_an_sol, y_an_sol = sysAnalytical(t_space)

        # function solution
        solRK4 = RK4(ode_sys, np.array([x_init, y_init]), t_space, 0)
        solForEuler = forwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)
        solBackEuler = backwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)

        errRK4[i] = norm(solRK4.T[0] - x_an_sol + solRK4.T[1] - y_an_sol, ord=np.inf)
        errForEuler[i] = norm(solForEuler.T[0] - x_an_sol + solForEuler.T[1] - y_an_sol, ord=np.inf)
        errBackEuler[i] = norm(solBackEuler.T[0] - x_an_sol + solBackEuler.T[1] - y_an_sol, ord=np.inf)

    # Plot error
    plt.figure()
    plt.yscale("log")
    plt.plot(values, errRK4, color='magenta', linewidth=1, label='RK4')
    plt.plot(values, errForEuler, color='crimson', linewidth=1, label='forwardEuler')
    plt.plot(values, errBackEuler, color='mediumblue', linewidth=1, label='backwardEuler')
    plt.title('Konsistenzfehler')
    plt.xlabel('tau - Zwischenschritte pro Schritt')
    plt.ylabel('size of error of numerical calculation')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # testODEwith2D()
    testODEkonsistenz()
