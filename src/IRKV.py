import autograd.numpy as np


class Einschrittverfahren(object):

    def __init__(self, func, t0, te, y0, A, b, c, N, tol):
        self.func = func
        self.y0 = y0
        self.t0 = 0
        self.interval = [t0, te]
        self.grid = np.linspace(t0, te, N + 2)  # N internal points
        self.dt = (te - t0) / (N + 1)
        self.A = A
        self.b = b
        self.c = c
        self.N = N
        self.tol = tol
        self.s = len(b)
        self.m = len(y0)

    def step(self):
        ti = self.grid[0]
        yi = self.y0
        y, t = [yi], [ti]
        tim1 = ti
        test = 0
        for ti in self.grid[1:]:
            yi = yi + self.dt * self.phi(tim1, yi)
            tim1 = ti
            y.append(yi)
            t.append(ti)
        return np.array(y), np.array(t)

    def solve(self):
        self.solution = list(self.step())
