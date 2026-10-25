"""Time split operator steps for typical grid sizes, precisions and orders.

Run as ``python benchmarks/benchmark.py``; ``--help`` lists the options. The
time per step is the best of several runs of ``--steps`` steps, so it shows
what the machine can do rather than the noise of other processes.
"""

import argparse
import os
import platform
import time

import numpy as np
import scipy.fft

import so_schroedinger as so


def make_grid(samples, dtype):
    """A harmonic oscillator with a moving Gaussian; ``samples`` is an int for
    1d and a tuple ``(n, n)`` for 2d."""

    if np.ndim(samples) == 0:
        return so.Grid1d((-40.0, 40.0), samples, so.HarmOscPot(1.0), so.Gaussian1d(0.0, 1.0, 1.0), dtype=dtype)
    psi0 = so.Gaussian2d(3.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    return so.Grid2d(((-20.0, 20.0), (-20.0, 20.0)), samples, so.HarmOscPot2d(1.0), psi0, dtype=dtype)


def grid_sizes(quick):
    return [2**11, (256, 256), (512, 512)] if quick else [2**11, 2**14, (512, 512), (1024, 1024)]


def time_step(samples, dtype, order, workers, steps, repeat):
    """Best time per step in seconds."""

    grid = make_grid(samples, dtype)
    prop = so.SplitOperator(order=order, workers=workers)
    prop.propagate(grid, 0.001, 2)  # warm up the caches and FFT plans
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        prop.propagate(grid, 0.001, steps)
        best = min(best, (time.perf_counter() - t0) / steps)
    return best


def format_time(seconds):
    return f"{seconds * 1e3:8.3f} ms" if seconds >= 1e-3 else f"{seconds * 1e6:8.1f} µs"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--steps", type=int, default=20, help="steps per timing run (default 20)")
    parser.add_argument("--repeat", type=int, default=5, help="timing runs, the best counts (default 5)")
    parser.add_argument("--workers", type=int, default=-1, help="FFT threads, -1 for all cores (default)")
    parser.add_argument("--quick", action="store_true", help="smaller grids only")
    parser.add_argument("--pyfftw", action="store_true", help="use pyFFTW as the scipy.fft backend")
    args = parser.parse_args()

    backend = "scipy.fft"
    if args.pyfftw:
        import pyfftw.interfaces.cache
        import pyfftw.interfaces.scipy_fft

        pyfftw.interfaces.cache.enable()
        scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
        backend = "pyFFTW"

    print(f"so_schroedinger {so.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}, {backend}")
    print(f"{platform.processor() or platform.machine()}, {os.cpu_count()} cores, workers={args.workers}\n")
    columns = [(np.complex128, 2), (np.complex64, 2), (np.complex128, 4)]
    header = ["grid".ljust(12)] + [f"{np.dtype(d).name}, order {o}".rjust(22) for d, o in columns]
    print("".join(header))
    for samples in grid_sizes(args.quick):
        label = f"1d {samples}" if np.ndim(samples) == 0 else f"2d {samples[0]}x{samples[1]}"
        row = [label.ljust(12)]
        for dtype, order in columns:
            t = time_step(samples, dtype, order, args.workers, args.steps, args.repeat)
            row.append(format_time(t).rjust(22))
        print("".join(row), flush=True)


if __name__ == "__main__":
    main()
