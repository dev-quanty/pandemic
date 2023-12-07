from scipy.optimize import fmin
from scipy.interpolate import interp1d
import numpy as np
from ODESolver import solve


def minloss(data, model, method, parameters):
    """
    Minimize the loss between the data and the model simulation. Interpolates between simulation and true data for
    error calculation, if inputted time steps not equidistant.

    Args:
        data: TxN+1 array with the time points in the first column, first row corresponds to t=0
        model: Model function with N differential equations
        method: Method for the ODE Solver
        parameters: List of model parameters, contains NoneType if parameter should be fitted

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
        y_est = solve(model, y0, t_est, method, params)
        interpolator = interp1d(t_est, y_est.T, kind="cubic")
        y_fitted = interpolator(t).T
        return np.linalg.norm(y_fitted - y)

    missingParams = [0] * nparams
    minimum = fmin(loss, missingParams)
    fittedLoss = loss(minimum)
    return fillParams(minimum), fittedLoss


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from models import sir

    # Create true data
    y0 = [0.9, 0.1, 0]
    params = [1/2, 1/3]
    t = np.linspace(0, 50, 5000)
    y = solve(sir, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y]), sir, "RK4", [None, None])
    ytest = solve(sir, y0, t, args=fittedparams)

    fig, ax = plt.subplots()
    ax.set_title(f"True Parameters: {params}\nFitted: {fittedparams}")
    ax.plot(t, y[:, 0], label="S [True]", c="black")
    ax.plot(t, ytest[:, 0], label="S [Test]", c="blue")
    ax.plot(t, y[:, 1], label="I [True]", c="red")
    ax.plot(t, ytest[:, 1], label="I [Test]", c="orange")
    plt.show()