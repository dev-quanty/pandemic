import pandas as pd
import numpy as np

# configure error-handeling
np.seterr(all='raise')

''' 
In this section we define the model to use.
This function receives the inputs of a specific time and returns the calculated derivatives at the point t.
The parameters are added in the numpy-array param_SEIRD.
The previous SEIRD-values (or the values at t = t_0) are added into a numpy-array called last_SEIRD.
'''

# param_SEIRD = [k_i, e, y_i, p_i]
parameters = np.array(0.6, 0.3, 0.7, 0.2)

# last_SEIRD = [S, E, I, R, D]
last_SEIRD = np.array(100, 1, 0, 0, 0)


def SEIRD(t, parameters):  # [S, E, I, R, D, k_i, e, y_i, p_i]
    # unpack last_SEIRD
    S = parameters[0]
    E = parameters[1]
    I = parameters[2]

    # unpack paramSEIRD
    k_i = parameters[6]
    e = parameters[7]
    y_i = parameters[8]
    p_i = parameters[9]

    # define derivative functions
    dSdt = - k_i * S * I
    dEdt = k_i * S * I - e * E
    dIdt = e * E - y_i * I
    dRdt = (1 - p_i) * y_i * I
    dDdt = p_i * y_i * I

    return np.array(dSdt, dEdt, dIdt, dRdt, dDdt, 0, 0, 0, 0)


'''
general implementation of forward-euler numerical approximation of the next step
f = function dependent on t and u(t), calculates the derivative
s0 = starting conditions
'''


def forwardEuler(f, s0, step_nbr, step_size):
    s = np.zeros(step_nbr, s0)
    s[0] = s0

    for t in range(1, step_nbr):
        s[t + 1] = s[t] + step_size * f(t * step_nbr, s[t])

    return np.array(s)


""" 
TODO:
- TEST forwardEuler on SEIRD
- adjust Visualize
- adjust simulate
- discuss with Team
"""


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
            elif method.lower() == "implicit":
                St1 = St / (1 + dt * beta * It)
                Et1 = (Et + dt * beta * It * St1) / (1 + dt * alpha)
                It1 = (It + dt * alpha * Et1) / (1 + dt * gamma)
                Rt1 = Rt + dt * gamma * (1 - theta) * It1
                Dt1 = Dt + dt * gamma * theta * It1
            else:
                Sk1 = -beta * It * St
                Ek1 = beta * It * St - alpha * Et
                Ik1 = alpha * Et - gamma * It
                Rk1 = gamma * (1 - theta) * It
                Dk1 = gamma * theta * It

                Sk2 = -beta * (St + (Sk1 * dt) / 2) * (It + (Ik1 * dt) / 2)
                Ek2 = beta * (St + (Sk1 * dt) / 2) * (It + (Ik1 * dt) / 2) - alpha * (Et + (Ek1 * dt) / 2)
                Ik2 = alpha * (Et + (Ek1 * dt) / 2) - gamma * (It + (Ik1 * dt) / 2)
                Rk2 = (1 - theta) * gamma * (It + (Ik1 * dt) / 2)
                Dk2 = theta * gamma * (It + (Ik1 * dt) / 2)

                Sk3 = -beta * (St + (Sk2 * dt) / 2) * (It + (Ik2 * dt) / 2)
                Ek3 = beta * (St + (Sk2 * dt) / 2) * (It + (Ik2 * dt) / 2) - alpha * (Et + (Ek2 * dt) / 2)
                Ik3 = alpha * (Et + (Ek2 * dt) / 2) - gamma * (It + (Ik2 * dt) / 2)
                Rk3 = (1 - theta) * gamma * (It + (Ik2 * dt) / 2)
                Dk3 = theta * gamma * (It + (Ik2 * dt) / 2)

                Sk4 = -beta * (St + Sk3 * dt) * (It + Ik3 * dt)
                Ek4 = beta * (St + Sk3 * dt) * (It + Ik3 * dt) - alpha * (Et + Ek3 * dt)
                Ik4 = alpha * (Et + Ek3 * dt) - gamma * (It + Ik3 * dt)
                Rk4 = (1 - theta) * gamma * (It + Ik3 * dt)
                Dk4 = theta * gamma * (It + Ik3 * dt)

                St1 = St + dt * (Sk1 + 2 * Sk2 + 2 * Sk3 + Sk4) / 6
                Et1 = Et + dt * (Ek1 + 2 * Ek2 + 2 * Ek3 + Ek4) / 6
                It1 = It + dt * (Ik1 + 2 * Ik2 + 2 * Ik3 + Ik4) / 6
                Rt1 = Rt + dt * (Rk1 + 2 * Rk2 + 2 * Rk3 + Rk4) / 6
                Dt1 = Dt + dt * (Dk1 + 2 * Dk2 + 2 * Dk3 + Dk4) / 6

            df.loc[j+1] = [St1, Et1, It1, Rt1, Dt1, t1]
            j += 1
            try: linnorm = np.linalg.norm(np.array([St1, Et1, It1, Rt1, Dt1]) - np.array([St, Et, It, Rt, Dt]))
            except: linnorm = 0
            if linnorm <= tol and j > 20:
                break
        return df
