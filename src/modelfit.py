import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import sir, seird, seirqhfd
from src.ODESolver import solve
from src.fitting import minloss


# Static variables
DATA_DIR = "../Data/Processed/"
COUNTRY = "Liberia"
MODEL = seird
METHOD = "fmin"
SOLVER = "impliciteuler"

# Mappings
N_COUNTRY = {
    "Liberia": 10_000
}
PARAMETERS = {
    "ki": None,
    "kh": None,
    "kf": None,
    "e": None,
    "qe": None,
    "qi": None,
    "h": None,
    "yi": None,
    "yh": None,
    "pi": 0.666,
    "ph": None,
    "d": None
}

# Read data
df = pd.read_excel(os.path.join(DATA_DIR, COUNTRY + ".xlsx"))
N = N_COUNTRY.get(COUNTRY)

# Add time index
start_dt = df.loc[0, "Date"]
df["t"] = (df["Date"] - start_dt).dt.days

# Prepare data and fit models
if MODEL == sir:
    # Data preparation
    df.rename(columns={"E-I": "I", "D": "R"}, inplace=True)
    df["I"] = df["I"]
    df["R"] = df["R"] / PARAMETERS.get("pi")
    y0 = np.array([N - df.loc[0, "I"], df.loc[0, "I"], 0])
    y = df[["t", "I", "R"]].to_numpy()
    stop = df[["t"]].to_numpy().max()
    t = np.linspace(start=0, stop=stop, num=stop+1, endpoint=True)
    letters = "IR"

    # Choose parameters
    ki = PARAMETERS.get("ki")
    yi = PARAMETERS.get("yi")
    args = [ki, yi]

    # Fit model
    fittedparams, loss = minloss(y, MODEL, args, method=METHOD, solver=SOLVER, y0=y0, letters=letters)
    y_fitted = solve(MODEL, y0, t, args=fittedparams)

    # Plot model
    fig, ax = plt.subplots(nrows=3)
    ax[0].plot(t, y_fitted[:, 0], label="S [Fitted]", c="blue", linestyle="dashed")
    ax[1].scatter(y[:, 0], y[:, 1], label="I [Data]", c="black", marker="s")
    ax[1].plot(t, y_fitted[:, 1], label="I [Fitted]", c="blue", linestyle="dashed")
    ax[2].scatter(y[:, 0], y[:, 2], label="R [Data]", c="black", marker="s")
    ax[2].plot(t, y_fitted[:, 2], label="R [Fitted]", c="blue", linestyle="dashed")

    for axis in fig.axes:
        axis.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    print(fittedparams)

elif MODEL == seird:
    # Data preparation
    df.rename(columns={"E-I": "E"}, inplace=True)
    df["E"] = df["E"]
    df["D"] = df["D"]
    y0 = np.array([N - df.loc[0, "E"] - df.loc[0, "D"], df.loc[0, "E"], 0, 0, df.loc[0, "D"]])
    y = df[["t", "E", "D"]].to_numpy()
    stop = df[["t"]].to_numpy().max()
    t = np.linspace(start=0, stop=stop, num=stop+1, endpoint=True)
    letters = "ED"

    # Choose parameters
    ki = PARAMETERS.get("ki")
    e = PARAMETERS.get("e")
    yi = PARAMETERS.get("yi")
    pi = PARAMETERS.get("pi")
    args = [ki, e, yi, pi]

    # Fit model
    fittedparams, loss = minloss(y, MODEL, args, method=METHOD, solver=SOLVER, y0=y0, letters=letters)
    y_fitted = solve(MODEL, y0, t, args=fittedparams)

    # Plot model
    fig, ax = plt.subplots(nrows=5)
    ax[0].plot(t, y_fitted[:, 0], label="S [Fitted]", c="blue", linestyle="dashed")
    ax[1].scatter(y[:, 0], y[:, 1], label="E [Data]", c="black", marker="s")
    ax[1].plot(t, y_fitted[:, 1], label="E [Fitted]", c="blue", linestyle="dashed")
    ax[2].plot(t, y_fitted[:, 2], label="I [Fitted]", c="blue", linestyle="dashed")
    ax[3].plot(t, y_fitted[:, 3], label="R [Fitted]", c="blue", linestyle="dashed")
    ax[4].scatter(y[:, 0], y[:, 2], label="D [Data]", c="black", marker="s")
    ax[4].plot(t, y_fitted[:, 4], label="D [Fitted]", c="blue", linestyle="dashed")

    for axis in fig.axes:
        axis.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

else:
    raise ValueError("Missing correct model")
