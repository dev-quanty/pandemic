from scipy.optimize import fmin
from scipy.interpolate import interp1d
import numpy as np
from ODESolver import solve
from toms178 import hooke


def minloss(data, model, parameters, method="fmin", solver="RK4", **kwargs):
    """
    Minimize the loss between the data and the model simulation. Interpolates between simulation and true data for
    error calculation, if inputted time steps not equidistant.

    Args:
        data: TxN+1 array with the time points in the first column, first row corresponds to t=0
        model: Model function with N differential equations
        parameters: List of model parameters, contains NoneType if parameter should be fitted
        method: Method to minimize the loss (fmin or hooke)
        solver: Method for the ODE Solver

    Returns:
        parameters: All parameters (inputted and fitted)
        loss: MSE of final model
    """
    t = data[:, 0] - data[0, 0]
    y = data[:, 1:]
    dt = min(0.01, np.min(t[1:] - t[:-1]))
    T = np.max(t)
    t_est = np.linspace(0, T, int(T // dt) + 1, endpoint=True)
    nparams = len([p for p in parameters if p is None])

    def fillParams(missingParams):
        params = []
        i = 0
        for p in parameters:
            if p is None:
                params.append(missingParams[i])
                i += 1
            else:
                params.append(p)
        return params

    def loss(missingParams):
        params = fillParams(missingParams)
        y0 = y[0, :]
        y_est = solve(model, y0, t_est, solver, params)
        interpolator = interp1d(t_est, y_est.T, kind="cubic")
        y_fitted = interpolator(t).T
        return np.linalg.norm(y_fitted - y)

    missingParams = [0] * nparams

    if method == "fmin":
        minimum = fmin(loss, missingParams)
    elif method == "hooke":
        rho = kwargs.get("rho", 0.1)
        eps = kwargs.get("eps", 1e-6)
        maxiter = kwargs.get("maxiter", 100000)
        f = lambda X, _: loss(X)
        _, minimum = hooke(nparams, missingParams, rho, eps, maxiter, f)
    else:
        raise ValueError("Method not implemented")

    fittedLoss = loss(minimum)
    return fillParams(minimum), fittedLoss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from models import sir

    # Create true data
    y0 = [9, 1, 0]
    params = [0.321, 1.677]
    t = np.linspace(0, 6, 5000)
    y = solve(sir, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y]), sir, [None, None], "hooke", "RK4")
    ytest = solve(sir, y0, t, args=fittedparams)

    fig, ax = plt.subplots()
    ax.set_title(f"True Parameters: {params}\nFitted: {fittedparams}")
    ax.plot(t, y[:, 0], label="S [True]", c="black")
    ax.plot(t, ytest[:, 0], label="S [Test]", c="blue")
    ax.plot(t, y[:, 1], label="I [True]", c="red")
    ax.plot(t, ytest[:, 1], label="I [Test]", c="orange")
    plt.show()