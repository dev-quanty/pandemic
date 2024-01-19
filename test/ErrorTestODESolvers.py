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
    # Parameters
    t_space = np.linspace(0., 5., 100)
    x_init = 2.
    y_init = 0.

    # analytical solution
    x_an_sol, y_an_sol = sysAnalytical(t_space)

    # Function solution
    RK4_sol = RK4(ode_sys, np.array([x_init, y_init]), t_space, 0)
    x_num_sol = RK4_sol.T[0]
    y_num_sol = RK4_sol.T[1]

    # Plot
    plt.figure()
    plt.plot(t_space, x_an_sol, '--', linewidth=2, label='analytical x')
    plt.plot(t_space, y_an_sol, '--', linewidth=2, label='analytical y')
    plt.plot(t_space, x_num_sol, linewidth=1, label='numerical x')
    plt.plot(t_space, y_num_sol, linewidth=1, label='numerical y')
    plt.title('System of 2 ODEs 1st order solved with different method')
    plt.xlabel('t')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    testODEwith2D()
