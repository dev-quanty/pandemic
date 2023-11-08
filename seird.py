import pandas as pd
import numpy as np


class SEIRD:
    def __init__(self, initial_values):
        self.initial_values = initial_values
        self.names = ["Susceptible", "Exposed", "Infected", "Recovered", "Dead"]

    def simulate(self, R0, i, r, theta, dt, tol=1e-5):
        """
        Simulates the SEIR model using the Forward Euler method

        Args:
            R0: Basic Reproduction Rate
            i: Incubation Period (days)
            r: Disease Duration until recovered (days)
            theta: Mortality Rate
            dt: Step size (days)
            tol: Stopping Criterion

        Returns:
            df: Simulated components
        """
        alpha = dt * 1 / i
        beta = dt * r * R0
        gamma = dt * r

        df = pd.DataFrame([self.initial_values], columns=self.names)
        df["t"] = 0

        j = 0
        while True:
            St, Et, It, Rt, Dt, t = df.iloc[j]
            t1 = t + dt
            St1 = St - beta * It * St
            Et1 = Et + beta * It * St - alpha * Et
            It1 = It + alpha * Et - gamma * It
            Rt1 = Rt + gamma * (1 - theta) * It
            Dt1 = Dt + gamma * theta * It
            df.loc[j+1] = [St1, Et1, It1, Rt1, Dt1, t1]
            j += 1
            if np.linalg.norm(np.array([St1, Et1, It1, Rt1, Dt1]) - np.array([St, Et, It, Rt, Dt])) <= tol and j > 20:
                break
        return df
