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

## Contents

- `so_schroedinger.grid`: periodic 1d grid holding ψ and V
- `so_schroedinger.potentials`: barriers, wells, ramps, oscillators, random
  potentials with a prescribed correlation function, absorbing boundaries
- `so_schroedinger.wavepackets`: Gaussian wavepackets and coherent states
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
