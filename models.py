import numpy as np


def sir(y, t, args):
    beta, gamma = args
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])


def seird(y, t, args):
    beta, alpha, gamma, theta = args
    S, E, I, R, D = y
    dSdt = -beta * S * I
    dEdt = beta * S * I - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * (1 - theta) * I
    dDdt = gamma * theta * I
    return np.array([dSdt, dEdt, dIdt, dRdt, dDdt])


def seirqhfd(y, t, args):
    ki, kh, kf, e, qe, qi, h, yi, yq, yh, pi, pq, ph, d = args
    S, E, I, QE, QI, H, R, F, D = y
    dSdt = -ki * S * I - kh * S * H - kf * S * F
    dEdt = ki * S * I + kh * S * H + kf * S * F - e * E - qe * E
    dIdt = e * E - qi * I - yi * I - h * I
    dQEdt = qe * E - e * QE
    dQIdt = e * QE + qi * I - yq * QI - h * QI
    dHdt = h * I + h * QI - yh * H
    dRdt = yi * (1 - pi) * I + yq * (1 - pq) * QI + yh * (1 - ph) * H
    dFdt = yi * pi * I + yq * pq * QI + yh * ph * H - d * F
    dDdt = d * F
    return np.array([dSdt, dEdt, dIdt, dQEdt, dQIdt, dHdt, dRdt, dFdt, dDdt])
