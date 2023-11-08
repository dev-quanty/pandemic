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
            S0, E0, I0, R0, D0, t0 = df.iloc[j]
            t1 = t0 + dt
            S1 = S0 - beta * I0 * S0
            E1 = E0 + beta * I0 * S0 - alpha * E0
            I1 = I0 + alpha * E0 - gamma * I0
            R1 = R0 + gamma * (1 - theta) * I0
            D1 = D0 + gamma * theta * I0
            df.loc[j+1] = [S1, E1, I1, R1, D1, t1]
            j += 1
            if np.linalg.norm(np.array([S1, E1, I1, R1, D1]) - np.array([S0, E0, I0, R0, D0])) <= tol and j > 20:
                break
        return df
