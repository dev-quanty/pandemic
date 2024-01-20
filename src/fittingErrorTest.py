import numpy as np
import matplotlib.pyplot as plt
from fitting import minloss
from models import sir, seird, seirqhfd
from ODESolver import solve


def fittingErrorSIvsSIR():
    # Create true data
    y0 = [9, 1, 0]
    params = [0.321, 1.677]
    t = np.linspace(0, 10, 100)
    y = solve(sir, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y[:, [0, 1]]]), sir, [None, None], "hooke", "RK4",
                                 letters="SI")
    ytest = solve(sir, y0, t, args=fittedparams)

    plt.close()
    plt.figure()
    plt.title(f"Gewählte Parameter: {params}\nFitted: {fittedparams}\nFehler: {np.linalg.norm(y - ytest)}")
    plt.plot(t, y[:, 0], color="blue", label='S [Wahr]')
    plt.plot(t, ytest[:, 0], color="cornflowerblue", label='S [Fit]')
    plt.plot(t, y[:, 1], color="darkgoldenrod", label='I [Wahr]')
    plt.plot(t, ytest[:, 1], color="gold", label='I [Fit]')
    plt.plot(t, y[:, 2], color="green", label='R [Wahr]')
    plt.plot(t, ytest[:, 2], color="lime", label='R [Fit]')
    plt.xlabel('t - Zeitabschnitte')
    plt.ylabel('Population')
    plt.legend()
    plt.show()

    # Create true data
    y0 = [9, 2, 0]
    params = [0.321, 1.677]
    t = np.linspace(0, 10, 100)
    y = solve(sir, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y[:, [0, 1, 2]]]), sir, [None, None], "hooke", "RK4",
                                 letters="SIR")
    ytest = solve(sir, y0, t, args=fittedparams)

    plt.figure()
    plt.title(f"Gewählte Parameter: {params}\nFitted: {fittedparams}\nFehler: {np.linalg.norm(y - ytest)}")
    plt.plot(t, y[:, 0], color="blue", label='S [Wahr]')
    plt.plot(t, ytest[:, 0], color="cornflowerblue", label='S [Fit]')
    plt.plot(t, y[:, 1], color="darkgoldenrod", label='I [Wahr]')
    plt.plot(t, ytest[:, 1], color="gold", label='I [Fit]')
    plt.plot(t, y[:, 2], color="green", label='R [Wahr]')
    plt.plot(t, ytest[:, 2], color="lime", label='R [Fit]')
    plt.xlabel('t - Zeitabschnitte')
    plt.ylabel('Population')
    plt.legend()
    plt.show()


def fittingErrorSEIRD():
    # Create true data
    y0 = [99, 0, 1, 0, 0]
    params = [0.321, 0.111, 1.677, 0.667]
    t = np.linspace(0, 10, 100)
    y = solve(seird, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y[:, [1, 2, 4]]]), seird, [0.321, 0.111, None, None],
                                 "hooke", "RK4", letters="EID")
    ytest = solve(seird, y0, t, args=fittedparams)

    plt.close()
    plt.figure()
    plt.title(f"Gewählte Parameter: {params}\nFitted: {fittedparams}\nFehler: {np.linalg.norm(y - ytest)}")
    plt.plot(t, y[:, 0], color="blue", label='S [Wahr]')
    plt.plot(t, ytest[:, 0], color="cornflowerblue", label='S [Fit]')
    plt.plot(t, y[:, 1], color="teal", label='E [Wahr]')
    plt.plot(t, ytest[:, 1], color="turquoise", label='E [Fit]')
    plt.plot(t, y[:, 2], color="darkgoldenrod", label='I [Wahr]')
    plt.plot(t, ytest[:, 2], color="gold", label='I [Fit]')
    plt.plot(t, y[:, 3], color="green", label='R [Wahr]')
    plt.plot(t, ytest[:, 3], color="lime", label='R [Fit]')
    plt.plot(t, y[:, 4], color="black", label='D [Wahr]')
    plt.plot(t, ytest[:, 4], color="dimgrey", label='D [Fit]')
    plt.xlabel('t - Zeitabschnitte')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    fittingErrorSIvsSIR()
    fittingErrorSEIRD()
