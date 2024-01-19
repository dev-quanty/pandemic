import numpy as np
import matplotlib.pyplot as plt
from src.ODESolver import forwardEuler, backwardEuler, RK4


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
    t_space = np.linspace(0., 5., 100)
    x_init = 2.
    y_init = 0.

    # analytical solution
    x_an_sol, y_an_sol = sysAnalytical(t_space)

    # function solution
    solRK4 = RK4(ode_sys, np.array([x_init, y_init]), t_space, 0)
    solForEuler = forwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)
    solBackEuler = backwardEuler(ode_sys, np.array([x_init, y_init]), t_space, 0)

    # Plot
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


if __name__ == "__main__":
    testODEwith2D()