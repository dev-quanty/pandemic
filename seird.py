import pandas as pd
import numpy as np


np.seterr(all='raise')
class SEIRD:
    def __init__(self, initial_values):
        self.initial_values = initial_values
        self.names = ["Susceptible", "Exposed", "Infected", "Recovered", "Dead"]

    def simulate(self, R0, i, r, theta, dt, tol=1e-5, method="implicit"):
        """
        Simulates the SEIR model using the Forward Euler method

        Args:
            R0: Basic Reproduction Rate
            i: Incubation Period (days)
            r: Disease Duration until recovered (days)
            theta: Mortality Rate (%)
            dt: Step size (days)
            tol: Stopping Criterion
            method: Explicit or Implicit Euler

        Returns:
            df: Simulated components
        """
        alpha = 1 / i
        beta = r * R0
        gamma = r

        df = pd.DataFrame([self.initial_values], columns=self.names)
        df["t"] = 0

        j = 0
        while True:
            St, Et, It, Rt, Dt, t = df.iloc[j]
            t1 = t + dt
            if method.lower() == "explicit":
                St1 = St - dt * beta * It * St
                Et1 = Et + dt * (beta * It * St - alpha * Et)
                It1 = It + dt * (alpha * Et - gamma * It)
                Rt1 = Rt + dt * gamma * (1 - theta) * It
                Dt1 = Dt + dt * gamma * theta * It
            else:
                St1 = St / (1 + dt * beta * It)
                Et1 = (Et + dt * beta * It * St1) / (1 + dt * alpha)
                It1 = (It + dt * alpha * Et1) / (1 + dt * gamma)
                Rt1 = Rt + dt * gamma * (1 - theta) * It1
                Dt1 = Dt + dt * gamma * theta * It1
            df.loc[j+1] = [St1, Et1, It1, Rt1, Dt1, t1]
            j += 1
            try: linnorm = np.linalg.norm(np.array([St1, Et1, It1, Rt1, Dt1]) - np.array([St, Et, It, Rt, Dt]))
            except: linnorm = 0
            if linnorm <= tol and j > 20:
                break
        return df
