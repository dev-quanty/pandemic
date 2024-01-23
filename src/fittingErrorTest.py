import numpy as np
import matplotlib.pyplot as plt
from fitting import minloss
from models import sir, seird, seirqhfd
from ODESolver import solve
import warnings


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
    plt.title(
        f"Gewählte Parameter: {params}\nFitted: {fittedparams}\nFehler pro Messung: {np.linalg.norm(y - ytest) / 100}")
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


def fittingErrorSEIRDQHF():
    # suppress warnings
    warnings.filterwarnings('ignore')
    # Create true data
    y0 = [9, 0, 1, 0, 0, 0, 0, 0, 0]
    params = [0.3, 0.3, 0.03, 0.111, 0.6, 0.18, 0.3, 0.7,
              0.9, 0.6, 0.3, 0.18]
    t = np.linspace(0, 50, 100)
    y = solve(seirqhfd, y0, t, args=params)

    # Fit all parameters
    fittedparams, loss = minloss(np.hstack([t.reshape(-1, 1), y[:, [2, 3, 4, 5, 7, 8]]]), seirqhfd,
                                 [0.3, 0.3, 0.03, 0.111, None, 0.18, None, None, None, None, None, 0.18],
                                 "hooke", "RK4", letters="I,QE,QI,H,F,D")
    ytest = solve(seirqhfd, y0, t, args=fittedparams)

    plt.close()
    plt.figure()
    plt.title(
        f"Gewählte Parameter: {params}\nFitted: {fittedparams}\nFehler pro Messung: {np.linalg.norm(y - ytest) / 100}")
    plt.plot(t, y[:, 0], color="blue", label='S [Wahr]')
    plt.plot(t, ytest[:, 0], color="cornflowerblue", label='S [Fit]')
    plt.plot(t, y[:, 1], color="teal", label='E [Wahr]')
    plt.plot(t, ytest[:, 1], color="turquoise", label='E [Fit]')
    plt.plot(t, y[:, 2], color="darkgoldenrod", label='I [Wahr]')
    plt.plot(t, ytest[:, 2], color="gold", label='I [Fit]')
    plt.plot(t, y[:, 3], color="cadetblue", label='QE [Wahr]')
    plt.plot(t, ytest[:, 3], color="paleturquoise", label='QE [Fit]')
    plt.plot(t, y[:, 4], color="darkkhaki", label='QI [Wahr]')
    plt.plot(t, ytest[:, 4], color="khaki", label='QI [Fit]')
    plt.plot(t, y[:, 5], color="chocolate", label='H [Wahr]')
    plt.plot(t, ytest[:, 5], color="peru", label='H [Fit]')
    plt.plot(t, y[:, 6], color="green", label='R [Wahr]')
    plt.plot(t, ytest[:, 6], color="lime", label='R [Fit]')
    plt.plot(t, y[:, 7], color="maroon", label='F [Wahr]')
    plt.plot(t, ytest[:, 7], color="lightcoral", label='F [Fit]')
    plt.plot(t, y[:, 8], color="black", label='D [Wahr]')
    plt.plot(t, ytest[:, 8], color="dimgrey", label='D [Fit]')
    plt.xlabel('t - Zeitabschnitte')
    plt.ylabel('Population')
    plt.legend(loc='upper right')
    plt.show()


if __name__ == "__main__":
    fittingErrorSIvsSIR()
    fittingErrorSEIRD()
    fittingErrorSEIRDQHF()
