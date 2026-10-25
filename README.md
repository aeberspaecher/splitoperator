# so_schroedinger

Split operator time propagation for the time-dependent Schrödinger equation,
written in pure Python on top of NumPy's FFT.

The propagator uses the symmetrized (Strang) splitting

    exp(-i H dt) ≈ exp(-i V dt/2) · exp(-i T dt) · exp(-i V dt/2)

where the kinetic part is applied in momentum space. Complex time steps
`dt = -i τ` give imaginary time evolution, which relaxes to the ground state.
All quantities are in atomic units (ħ = m = 1).

## Installation

```bash
pip install .            # or: pip install -e ".[test]" for development
```

Requires Python ≥ 3.10, NumPy, SciPy and Matplotlib.

## Example

```python
import numpy as np
import so_schroedinger as so

psi0 = so.Gaussian1d(x0=-10.0, sigma=2.0, p0=2.0)
barrier = so.GaussBarrier1d(x0=0.0, V0=2.0, sigma=0.5)
grid = so.Grid1d(size=(-50.0, 50.0), gridsamples=2**11, potential=barrier, psi0=psi0)

prop = so.SplitOperator()
prop.propagate(grid, dt=0.01, nSteps=1000)

print(grid.t, grid.psiNorm, grid.expectationX())
```

## Two dimensions

`Grid2d` works like `Grid1d`; potentials and wavepackets are evaluated with
`r = (X, Y)`:

```python
size = ((-4.5, 4.5), (-2.5, 2.5))
stadium = so.StadiumPotential2d(a=2.0, R=2.0, V0=400.0, w=0.04)
psi0 = so.Gaussian2d(x0=0.0, y0=0.0, sigmax=0.4, px=8.0)
grid = so.Grid2d(size, (192, 108), stadium, psi0)
so.SplitOperator().propagate(grid, dt=0.004, nSteps=1000)
anim = so.animate2d(grid, so.SplitOperator(), dt=0.004, nFrames=50, stepsPerFrame=20)
```

2d potentials: `HarmOscPot2d`, `GaussBump2d`, `StadiumPotential2d`,
`RandomPotential2d` (Gaussian random field with a given correlation
function) and `BoundaryAbsorber2d`. `analyzers.EigenstateFilter` builds
eigenstates by energy filtering on the fly, without storing the whole time
series.

## Animations

```python
from IPython.display import HTML

anim = so.animate1d(grid, so.SplitOperator(), dt=0.02, nFrames=150, stepsPerFrame=10)
HTML(anim.to_jshtml())        # in a Jupyter notebook
anim.save("barrier.gif")      # or .mp4 (needs ffmpeg)
```

The propagation is done before the animation is drawn, so an animation can be
rendered or saved several times. See `examples/animate_barrier.py` for a
standalone script.

## Notebooks

The `notebooks/` folder contains illustrated examples:

1. `01_free_wavepacket.ipynb`: spreading of a free Gaussian, compared with the
   analytic solution
2. `02_tunneling.ipynb`: tunneling through a rectangular barrier, transmission
   versus energy compared with the plane wave formula
3. `03_harmonic_oscillator.ipynb`: coherent states, ground state from
   imaginary time evolution, spectrum and eigenstates from the
   autocorrelation function
4. `04_branched_flow.ipynb`: a wave in a weak, smooth random potential
   splits into branches; quantum density compared with classical rays
5. `05_stadium_billiard.ipynb`: chaotic dynamics in the Bunimovich stadium
   versus the circle, spectrum and eigenstates scarred by a periodic orbit

Install the extras with `pip install ".[notebooks]"` and start
`jupyter lab notebooks/`.

## Contents

- `so_schroedinger.grid`: periodic 1d and 2d grids holding ψ and V
- `so_schroedinger.potentials`: barriers, wells, ramps, oscillators, random
  potentials with a prescribed correlation function, absorbing boundaries
- `so_schroedinger.wavepackets`: Gaussian wavepackets (1d, 2d) and coherent states
- `so_schroedinger.timeprop`: the split operator propagator
- `so_schroedinger.analyzers`: autocorrelation spectra, expectation values,
  eigenfunctions from time series
- `so_schroedinger.animation`: Matplotlib animations of the propagation

## Tests

```bash
pytest
```

The tests check the propagator against analytic results: norm conservation,
spreading of a free Gaussian, a coherent state in a harmonic oscillator and
the oscillator ground state energy from imaginary time evolution.

## License

BSD 2-Clause, see [LICENSE](LICENSE).
