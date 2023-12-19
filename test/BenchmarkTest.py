# TODO: Testing ODESolver.py functions
import numpy as np
import perfplot


def test_brick():
    out = perfplot.bench(
        setup=lambda n: n,
        kernels=[
            # functions as names separated with comma
            lambda n: list(np.arange(n))
        ],
        labels=[],  # function names as strings, separated with comma
        n_range=[2 ** k for k in range(10)],  # log-scale
        xlabel="size of input",
        equality_check=None
    )
    print(out)
    # out.show()
    # out.save("perf.png", transparent=True, bbox_inches="tight")


if __name__ == "__main__":
    test_brick()
