import numpy as np


def sir(y, t, args):
    ki, yi = args
    S, I, R = y
    dSdt = -ki * S * I
    dIdt = ki * S * I - yi * I
    dRdt = yi * I
    return np.array([dSdt, dIdt, dRdt])


def seird(y, t, args):
    ki, e, yi, pi = args
    S, E, I, R, D = y
    dSdt = -ki * S * I
    dEdt = ki * S * I - e * E
    dIdt = e * E - yi * I
    dRdt = yi * (1 - pi) * I
    dDdt = yi * pi * I
    return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])


def seirqhfd(y, t, args):
    ki, kh, kf, e, qe, qi, h, yi, yh, pi, ph, d = args
    S, E, I, QE, QI, H, R, F, D = y
    dSdt = -ki * S * I - kh * S * H - kf * S * F
    dEdt = ki * S * I + kh * S * H + kf * S * F - e * E - qe * E
    dIdt = e * E - qi * I - yi * I - h * I
    dQEdt = qe * E - e * QE
    dQIdt = e * QE + qi * I - yi * QI - h * QI
    dHdt = h * I + h * QI - yh * H
    dRdt = yi * (1 - pi) * I + yi * (1 - pi) * QI + yh * (1 - ph) * H
    dFdt = yi * pi * I + yi * pi * QI + yh * ph * H - d * F
    dDdt = d * F
    return np.array([dSdt, dEdt, dIdt, dQEdt, dQIdt, dHdt, dRdt, dFdt, dDdt])


def seirqhfd_shared_weights(y, t, args):
    ki, kh, kf, e, qe, h, yi, pi, d = args
    S, E, I, QE, QI, H, R, F, D = y
    dSdt = -ki * S * I - kh * S * H - kf * S * F
    dEdt = ki * S * I + kh * S * H + kf * S * F - e * E - qe * E
    dIdt = e * E - qe * I - yi * I - h * I
    dQEdt = qe * E - e * QE
    dQIdt = e * QE + qe * I - yi * QI - h * QI
    dHdt = h * I + h * QI - yi * H
    dRdt = yi * (1 - pi) * I + yi * (1 - pi) * QI + yi * (1 - pi) * H
    dFdt = yi * pi * I + yi * pi * QI + yi * pi * H - d * F
    dDdt = d * F
    return np.array([dSdt, dEdt, dIdt, dQEdt, dQIdt, dHdt, dRdt, dFdt, dDdt])


def letter_index(model, letters):
    func = model.__name__
    if func == "sir":
        model_letters = list("SIR")
    elif func == "seird":
        model_letters = list("SEIRD")
    elif func == "seirqhfd" or func == "seirqhfd_shared_weights":
        model_letters = list("SEIQQHRFD")
        model_letters[3] = "QE"
        model_letters[4] = "QI"
    else:
        model_letters = []
    return [model_letters.index(l.upper()) for l in letters if l.upper() in model_letters], len(model_letters)
