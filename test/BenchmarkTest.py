import numpy as np
import perfplot
from tabulate import tabulate
from src.ODESolver import forwardEuler, backwardEuler, RK4
from src.models import sir, seird, seirqhfd

test_ode = "seirqhfd"

def test_brick():
    def printPretty(out):
        data = []
        headers = ["n"] + out.labels
        for n, t in zip(out.n_range, out.timings_s.T):
            lst = [str(n)] + [str(tt) for tt in t]
            data.append(lst)

        print(tabulate(data, headers=headers, tablefmt='psql'))

    # The function - preferably ODE - to calculate --- This can be replaced with functions from models.py
    def func(y, t, args):
        res = 0
        for i in args:
            res += i + y
            res /= 20
        return res

    def setup(n):
        t = np.linspace(0, 1, n)
        if test_ode == "sir":
            y0 = np.random.rand(3)
            args = np.random.rand(2)
            return sir, y0, t, args
        elif test_ode == "seird":
            y0 = np.random.rand(5)
            args = np.random.rand(4)
            return seird, y0, t, args
        elif test_ode == "seirqhfd":
            y0 = np.random.rand(9)
            args = np.random.rand(12)
            return seirqhfd, y0, t, args
        else:
            y0 = np.random.rand(n)
            args = np.random.rand(n)
            return func, y0, t, args

    # Define your kernels
    kernels = [forwardEuler, backwardEuler, RK4]
    labels = ["Forward Euler", "Backward Euler", "RK4"]

    # Generate the performance plot
    out = perfplot.bench(
        setup=setup,
        kernels=kernels,
        labels=labels,
        n_range=[2 ** k + 10 for k in range(5)],
        xlabel="iteration amount",
        equality_check=None
    )

    # Print and save the output
    printPretty(out)
    out.save("benchPlot.png", transparent=True, bbox_inches="tight")

if __name__ == "__main__":
    test_brick()