import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.models import sir, seird, seirqhfd, seirqhfd_shared_weights
from src.ODESolver import solve
from src.fitting import minloss


# Static variables
DATA_DIR = "../Data/Processed/"
COUNTRY = "Liberia"
MODEL = seird
METHOD = "fmin"
SOLVER = "Radau"

# Mappings
N_COUNTRY = {
    "Liberia": 6_000
}
PARAMETERS = {
    "ki": None,
    "kh": None,
    "kf": None,
    "e": 0.0954,
    "qe": None,
    "qi": None,
    "h": 0.2257,
    "yi": None,
    "yh": None,
    "pi": 0.666,
    "ph": 0.666,
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
    df["I"] = 100* df["I"] / N
    df["R"] = 100 * df["R"] / (N * PARAMETERS.get("pi", 0.666))
    start_value = 100 - df.loc[0, "I"] - df.loc[0, "R"]
    y0 = np.array([start_value, df.loc[0, "I"], df.loc[0, "R"]])
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
    for col in ["E", "D"]:
        df[col] = 100 * df[col] / N
    start_value = 100 - df.loc[0, "E"] - df.loc[0, "D"]
    y0 = np.array([start_value, df.loc[0, "E"], 0, 0, df.loc[0, "D"]])
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
    print(fittedparams)

elif MODEL == seirqhfd:
    # Data preparation
    df.rename(columns={"E-I": "E", "QE-QI": "QE", "D[F]": "D", "D": "D_Old"}, inplace=True)
    for col in ["E", "H", "F", "D"]:
        df[col] = 100 * df[col] / N
    start_value = 100 - df.loc[0, "E"] - df.loc[0, "H"] - df.loc[0, "F"] - df.loc[0, "D"]
    y0 = np.array([start_value, df.loc[0, "E"], 0, 0, 0, df.loc[0, "H"], 0, df.loc[0, "F"], df.loc[0, "D"]])
    y = df[["t", "E", "H", "F", "D"]].to_numpy()
    stop = df[["t"]].to_numpy().max()
    t = np.linspace(start=0, stop=stop, num=stop+1, endpoint=True)
    letters = ["E", "H", "F", "D"]

    # Choose parameters
    ki = PARAMETERS.get("ki")
    kh = PARAMETERS.get("kh")
    kf = PARAMETERS.get("kf")
    e = PARAMETERS.get("e")
    qe = PARAMETERS.get("qe")
    qi = PARAMETERS.get("qi")
    h = PARAMETERS.get("h")
    yi = PARAMETERS.get("yi")
    yh = PARAMETERS.get("yh")
    pi = PARAMETERS.get("pi")
    ph = PARAMETERS.get("ph")
    d = PARAMETERS.get("d")
    args = [ki, kh, kf, e, qe, qi, h, yi, yh, pi, ph, d]

    # Fit model
    fittedparams, loss = minloss(y, MODEL, args, method=METHOD, solver=SOLVER, y0=y0, letters=letters)
    y_fitted = solve(MODEL, y0, t, args=fittedparams)

    # Plot model
    fig, ax = plt.subplots(nrows=6)
    ax[0].scatter(y[:, 0], y[:, 1], label="E [Data]", c="black", marker="s")
    ax[0].plot(t, y_fitted[:, 1], label="E [Fitted]", c="blue", linestyle="dashed")
    ax[1].plot(t, y_fitted[:, 2], label="I [Fitted]", c="blue", linestyle="dashed")
    ax[2].plot(t, y_fitted[:, 3], label="QE [Fitted]", c="blue", linestyle="dashed")
    ax[3].scatter(y[:, 0], y[:, 2], label="H [Data]", c="black", marker="s")
    ax[3].plot(t, y_fitted[:, 5], label="H [Fitted]", c="blue", linestyle="dashed")
    ax[4].scatter(y[:, 0], y[:, 3], label="F [Data]", c="black", marker="s")
    ax[4].plot(t, y_fitted[:, 7], label="F [Fitted]", c="blue", linestyle="dashed")
    ax[5].scatter(y[:, 0], y[:, 4], label="D [Data]", c="black", marker="s")
    ax[5].plot(t, y_fitted[:, 8], label="D [Fitted]", c="blue", linestyle="dashed")

    for axis in fig.axes:
        axis.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    print(fittedparams)

elif MODEL == seirqhfd_shared_weights:
    # Data preparation
    df.rename(columns={"E-I": "E", "QE-QI": "QE", "D[F]": "D", "D": "D_Old"}, inplace=True)
    for col in ["E", "H", "F", "D"]:
        df[col] = 100 * df[col] / N
    start_value = 100 - df.loc[0, "E"] - df.loc[0, "H"] - df.loc[0, "F"] - df.loc[0, "D"]
    y0 = np.array([start_value, df.loc[0, "E"], 0, 0, 0, df.loc[0, "H"], 0, df.loc[0, "F"], df.loc[0, "D"]])
    y = df[["t", "E", "H", "F", "D"]].to_numpy()
    stop = df[["t"]].to_numpy().max()
    t = np.linspace(start=0, stop=stop, num=stop+1, endpoint=True)
    letters = ["E", "H", "F", "D"]

    # Choose parameters
    ki = PARAMETERS.get("ki")
    kh = PARAMETERS.get("kh")
    kf = PARAMETERS.get("kf")
    e = PARAMETERS.get("e")
    qe = PARAMETERS.get("qe")
    h = PARAMETERS.get("h")
    yi = PARAMETERS.get("yi")
    pi = PARAMETERS.get("pi")
    d = PARAMETERS.get("d")
    args = [ki, kh, kf, e, qe, h, yi, pi, d]

    # Fit model
    fittedparams, loss = minloss(y, MODEL, args, method=METHOD, solver=SOLVER, y0=y0, letters=letters)
    y_fitted = solve(MODEL, y0, t, args=fittedparams)

    # Plot model
    fig, ax = plt.subplots(nrows=6)
    ax[0].scatter(y[:, 0], y[:, 1], label="E [Data]", c="black", marker="s")
    ax[0].plot(t, y_fitted[:, 1], label="E [Fitted]", c="blue", linestyle="dashed")
    ax[1].plot(t, y_fitted[:, 2], label="I [Fitted]", c="blue", linestyle="dashed")
    ax[2].plot(t, y_fitted[:, 3], label="QE [Fitted]", c="blue", linestyle="dashed")
    ax[3].scatter(y[:, 0], y[:, 2], label="H [Data]", c="black", marker="s")
    ax[3].plot(t, y_fitted[:, 5], label="H [Fitted]", c="blue", linestyle="dashed")
    ax[4].scatter(y[:, 0], y[:, 3], label="F [Data]", c="black", marker="s")
    ax[4].plot(t, y_fitted[:, 7], label="F [Fitted]", c="blue", linestyle="dashed")
    ax[5].scatter(y[:, 0], y[:, 4], label="D [Data]", c="black", marker="s")
    ax[5].plot(t, y_fitted[:, 8], label="D [Fitted]", c="blue", linestyle="dashed")

    for axis in fig.axes:
        axis.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    print(fittedparams)

else:
    raise ValueError("Missing correct model")
