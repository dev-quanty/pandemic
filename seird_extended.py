import pandas as pd

class SEIRQHFD:
    def __init__(self, initial_values):
        self.initial_values = initial_values
        self.names = ["Susceptible", "Exposed", "Infected", "Recovered", "QuarantineExposed", "QuarantineInfected",
                      "Hospital", "Funeral", "Buried"]

    def simulate(self, ki, kh, kf, e, qe, qi, h, yi, yq, yh, pi, pq, ph, d, dt, tol):
        dSdt = lambda St, It, Ht, Ft: -ki * St * It - kh * St * Ht - kf * St * Ft
        E = lambda St, Et, It, Ht, Ft: ki * St * It + kh * St * Ht + kf * St * Ft - e * Et - qe * Et
        I = lambda Et, It: e * Et - yi * It - h * It - qi * It
        QE = lambda Et, QEt: qe * Et - e * QEt
        QI = lambda It, QEt, QIt: e * QEt + qi * It - yq * QIt - h * QIt
        H = lambda It, QIt, Ht: h * It + h * QIt - yh * Ht
        F = lambda It, QIt, Ht, Ft: pi * yi * It + pq * yq * QIt + ph * yh * Ht - d * Ft
        R = lambda It, QIt, Ht: (1 - pi) * yi * It + (1 - pq) * yq * QIt + (1 - ph) * yh * Ht
        D = lambda Ft: d * Ft

        df = pd.DataFrame([self.initial_values], columns=self.names)
        df["T"] = 0

