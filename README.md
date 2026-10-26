# so_schroedinger

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aeberspaecher/splitoperator/main?urlpath=lab/tree/notebooks)
[![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://aeberspaecher.github.io/splitoperator/lite/lab/index.html)
[![Docs](https://img.shields.io/badge/docs-API-blue)](https://aeberspaecher.github.io/splitoperator/)

Split operator time propagation for the time-dependent Schrödinger equation,
written in pure Python on top of NumPy's FFT.

The propagator uses the symmetrized (Strang) splitting

    exp(-i H dt) ≈ exp(-i V dt/2) · exp(-i T dt) · exp(-i V dt/2)

where the kinetic part is applied in momentum space. Time-dependent
potentials enter as V(t) in the first and V(t+dt) in the second half step,
which keeps the scheme second order. `SplitOperator(order=4)` or `order=6`
composes 3 or 9 Strang steps (Yoshida's triple jump) into a fourth or sixth
order step for real time steps. Complex time steps
`dt = -i τ` give imaginary time evolution, which relaxes to the ground state.
All quantities are in atomic units (ħ = m = 1).

## Installation

```bash
pip install so_schroedinger
```

or from a checkout, `pip install .` (`pip install -e ".[test]"` for
development). Requires Python ≥ 3.10, NumPy, SciPy and Matplotlib. The API
documentation is at https://aeberspaecher.github.io/splitoperator/.

## Example

```python
import numpy as np
import so_schroedinger as so

psi0 = so.Gaussian1d(x0=-10.0, sigma=2.0, p0=2.0)
barrier = so.GaussBarrier1d(x0=0.0, V0=2.0, sigma=0.5)
grid = so.Grid1d(size=(-50.0, 50.0), grid_samples=2**11, potential=barrier, psi0=psi0)

prop = so.SplitOperator()
prop.propagate(grid, dt=0.01, n_steps=1000)

print(grid.t, grid.psi_norm, grid.expectation_x())
```

## Speed

The FFTs use `scipy.fft` on all CPU cores; `SplitOperator(workers=1)` limits
them to one thread. Other FFT libraries plug in through SciPy's backend
mechanism, e.g. pyFFTW:

```python
import pyfftw.interfaces.scipy_fft
import scipy.fft

scipy.fft.set_global_backend(pyfftw.interfaces.scipy_fft)
```

For time-independent potentials, `propagate()` merges the potential half
steps of consecutive steps, which is exact and saves one array multiplication
per step. On a 4-core machine a step on a 512×512 grid takes about 3.4 ms
(11.5 ms in version 0.4.0) and a step on 2048 points in 1d about 48 µs (86 µs).

Grids take `dtype=np.complex64` for single precision, which halves the memory
of the wavefunction, the potential and recorded frames, and makes the 512×512
step about 1.5 times faster (2.7 ms against 4.2 ms here). Results then agree
with double precision to a few parts in 10⁶, but the norm drifts by about
5·10⁻⁸ per step from rounding (3·10⁻⁴ after 6000 steps), so long runs that
need it exact should use `SplitOperator(renormalize_every=...)` or stay in
double precision.

```python
grid = so.Grid2d(((-20, 20), (-20, 20)), 1024, pot, psi0, dtype=np.complex64)
```

`python benchmarks/benchmark.py` times steps for 1d and 2d grids in both
precisions and at orders 2 and 4 (`--quick` for the smaller grids only,
`--pyfftw` to use pyFFTW, `--help` for the rest).

## Two dimensions

`Grid2d` works like `Grid1d`; potentials and wavepackets are evaluated with
`r = (X, Y)`:

```python
size = ((-4.5, 4.5), (-2.5, 2.5))
stadium = so.StadiumPotential2d(a=2.0, R=2.0, V0=400.0, w=0.04)
psi0 = so.Gaussian2d(x0=0.0, y0=0.0, sigma_x=0.4, px=8.0)
grid = so.Grid2d(size, (192, 108), stadium, psi0)
so.SplitOperator().propagate(grid, dt=0.004, n_steps=1000)
anim = so.animate2d(grid, so.SplitOperator(), dt=0.004, n_frames=50, steps_per_frame=20)
```

2d potentials: `HarmOscPot2d`, `GaussBump2d`, `StadiumPotential2d`,
`RandomPotential2d` (Gaussian random field with a given correlation
function) and `BoundaryAbsorber2d`.

## Eigenstates

Imaginary time evolution relaxes any state to the ground state; projecting
out the states found before gives the excited ones:

```python
grid = so.Grid1d((-15.0, 15.0), 2**9, so.HarmOscPot(1.0), psi0)
E0 = so.ground_state(grid)  # grid.psi is now the ground state
energies, states = so.lowest_eigenstates(grid, 5)  # 0.5, 1.5, 2.5, ...
```

`grid.energy()` gives ⟨H⟩, split into `grid.expectation_kinetic()` and
`grid.expectation_potential()`.

## Absorbing boundaries

The grid is periodic, so a wave that leaves on one side comes back on the
other. An absorber passed as `absorber=` to the grid removes it near the edges:

```python
size = (-50.0, 50.0)
absorber = so.ManolopoulosAbsorber1d(E_min=0.5, x0=size[0], x1=size[1])
grid = so.Grid1d(size, 2**11, potential, psi0, absorber=absorber)
```

- `ManolopoulosAbsorber1d`: transmission-free absorber that reflects less
  than 1% of all waves above `E_min`, which also sets its width; the default
  choice
- `PowerAbsorber1d`: −iη uⁿ in a layer of given width (quadratic by default)
- `ExpAbsorber1d`: exponential imaginary part, optionally with a real part
- `MaskAbsorber1d`: multiplies ψ by cos^(1/8) in the layer once per time step
- `BoundaryAbsorber2d`: quadratic layer along the edges of a 2d grid

Notebook 12 measures their reflection and transmission versus energy.

## Analyzers

`so_schroedinger.analyzers` collects data during the propagation. Each
analyzer is called with the grid, `analyzer(grid)`, so it can be passed as
`callback` (or in a list of callbacks) to `propagate()` or in the `analyzers` list of `animate1d()` and
`animate2d()`. All of them work in 1d and 2d and do not plot anything.

```python
obs = so.analyzers.Observables()
corr = so.analyzers.AutoCorrelation()
obs(grid)  # record the initial state
corr(grid)
prop.propagate(grid, dt=0.01, n_steps=4096, callback=[obs, corr])

obs.t, obs.norm, obs.position, obs.momentum, obs.width  # arrays
E, spectrum = corr.spectrum()  # peaks at the eigenenergies
```

- `Observables`: norm, ⟨x⟩, ⟨p⟩, Δx and ⟨H⟩ versus time
- `AutoCorrelation`: C(t) = ⟨ψ₀|ψ(t)⟩ and the spectrum from it
- `EigenstateFilter`: eigenstates for given energies, filtered on the fly
- `EigenfunctionBuilder`: stores ψ(t) to build eigenstates for energies
  chosen afterwards
- `IntegratedDensity`: ∫|ψ|² dt
- `Flux`: probability flux through a point (1d) or line (2d), and its time
  integral, e.g. the transmission through a barrier
- `Wigner`, `Husimi`: snapshots of the phase space functions below (1d)

## Phase space

`so.wigner(grid)` and `so.husimi(grid, sigma=...)` return the Wigner
function W(x, p) and the Husimi function Q(x, p) of a 1d wavefunction as
`x, p, W` with `W[i, j]` at `(x[i], p[j])`. Both are normalized to
∫∫ W dx dp = 1; W can be negative where parts of the wavefunction
interfere, Q is the Wigner function smoothed over a coherent state and never
negative.

```python
x, p, W = so.wigner(grid, x_lim=(-6, 6), p_lim=(-5, 5))
so.plotting.plot_phase_space(x, p, W)  # diverging colors if W < 0 somewhere
```

## Plotting

`so_schroedinger.plotting` turns the usual pictures into one-liners. Each
function draws into `ax` (default: the current axes), labels the axes and
returns what it drew:

```python
so.plotting.plot1d(grid)  # |ψ|² over the scaled potential
so.plotting.plot2d(grid, kind="phase", potential_levels=3)  # also "density", "abs", "real"
so.plotting.plot_momentum(grid)  # |ψ(k)|², 1d or 2d
so.plotting.plot_eigenstates(grid, states, energies)  # |φₙ|² + Eₙ over V
so.plotting.plot_phase_space(*so.wigner(grid))  # W(x, p) or Q(x, p)
```

## Animations

```python
from IPython.display import HTML

anim = so.animate1d(grid, so.SplitOperator(), dt=0.02, n_frames=150, steps_per_frame=10)
HTML(anim.to_jshtml())  # in a Jupyter notebook
anim.save("barrier.gif")  # or .mp4 (needs ffmpeg)
```

The propagation is done before the animation is drawn, so an animation can be
rendered or saved several times. See `examples/animate_barrier.py` for a
standalone script.

For long runs on large grids, `live=True` propagates frame by frame while the
animation is drawn instead. An interactive window (`plt.show()` in a script,
`%matplotlib widget` in Jupyter) then starts at once, and no frames are kept
in memory; such an animation can be drawn only once. See
`examples/live_stadium.py`.

## Notebooks

The `notebooks/` folder contains illustrated examples. They run in the
browser without installing anything: on Binder (all notebooks, with the
package preinstalled; the first start takes a few minutes), on Google
Colab (one notebook at a time; its first cell installs the package) or in
JupyterLite, where Python itself runs in the browser (Pyodide). JupyterLite
starts in seconds and needs no server, but it is slower than native Python
and the animation cells take a while.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/aeberspaecher/splitoperator/main?urlpath=lab/tree/notebooks)
[![JupyterLite](https://jupyterlite.rtfd.io/en/latest/_static/badge.svg)](https://aeberspaecher.github.io/splitoperator/lite/lab/index.html)

| Notebook | Content | |
|---|---|---|
| `01_free_wavepacket.ipynb` | spreading of a free Gaussian, compared with the analytic solution | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/01_free_wavepacket.ipynb) |
| `02_tunneling.ipynb` | tunneling through a rectangular barrier, transmission versus energy compared with the plane wave formula, and sliders for barrier height, width and packet energy | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/02_tunneling.ipynb) |
| `03_harmonic_oscillator.ipynb` | coherent states, ground state from imaginary time evolution (ground and excited states), spectrum and eigenstates from the autocorrelation function, error versus time step for splittings of order 2, 4 and 6 | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/03_harmonic_oscillator.ipynb) |
| `04_branched_flow.ipynb` | a wave in a weak, smooth random potential splits into branches; quantum density compared with classical rays | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/04_branched_flow.ipynb) |
| `05_stadium_billiard.ipynb` | chaotic dynamics in the Bunimovich stadium versus the circle, spectrum and eigenstates scarred by a periodic orbit | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/05_stadium_billiard.ipynb) |
| `06_talbot_carpet.ipynb` | quantum carpet and exact and fractional revivals in an infinite well, and how finite walls spoil the revival | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/06_talbot_carpet.ipynb) |
| `07_anderson_localization.ipynb` | a wavepacket stops spreading in a weak 1d random potential; exponential tails and the localization length versus energy compared with the Born approximation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/07_anderson_localization.ipynb) |
| `08_double_slit.ipynb` | a wavepacket passes a double slit in 2d; the pattern on a screen compared with the Huygens–Fresnel and Fraunhofer predictions, and one slit at a time | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/08_double_slit.ipynb) |
| `09_double_well.ipynb` | tunneling between the wells of a double well: the splitting from eigenstates, the tunneling oscillation and the spectrum, Rabi oscillations, and coherent destruction of tunneling by a fast drive | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/09_double_well.ipynb) |
| `10_kicked_rotor.ipynb` | the kicked rotor as a split operator step: classical chaos in the standard map, dynamical localization against classical diffusion, the exponential momentum distribution, and noise that destroys localization | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/10_kicked_rotor.ipynb) |
| `11_phase_space.ipynb` | Wigner and Husimi functions of coherent, squeezed, excited and cat states; in the harmonic oscillator the Wigner function moves like a classical ensemble, in a quartic oscillator it filaments, interferes and partly revives, unlike the classical one | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/11_phase_space.ipynb) |
| `12_absorbers.ipynb` | reflection and transmission of absorbing boundaries versus energy: the transmission-free Manolopoulos absorber, quadratic and exponential absorbers, and a cos^(1/8) mask | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/12_absorbers.ipynb) |
| `13_limacon_cavity.ipynb` | a dielectric microcavity shaped like a limaçon (ε = 0.43, n = 3.3) as a Schrödinger well: whispering gallery modes from the autocorrelation spectrum, their quality factors, and their directional far field measured with `FarField` and checked against a Hankel extrapolation | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aeberspaecher/splitoperator/blob/main/notebooks/13_limacon_cavity.ipynb) |

To run them locally, install the extras with `pip install ".[notebooks]"` and
start `jupyter lab notebooks/`.

## Contents

- `so_schroedinger.grid`: periodic 1d and 2d grids holding ψ and V
- `so_schroedinger.potentials`: barriers, wells, ramps, oscillators, random
  potentials with a prescribed correlation function, absorbing boundaries
- `so_schroedinger.wavepackets`: Gaussian wavepackets (1d, 2d) and coherent states
- `so_schroedinger.timeprop`: the split operator propagator
- `so_schroedinger.eigenstates`: ground and excited states from imaginary time
- `so_schroedinger.analyzers`: expectation values, autocorrelation spectra,
  eigenstates, time integrated density, flux, Wigner and Husimi snapshots
- `so_schroedinger.phasespace`: Wigner and Husimi functions in 1d
- `so_schroedinger.animation`: Matplotlib animations of the propagation
- `so_schroedinger.plotting`: one-line plots of wavefunctions, momentum
  distributions and eigenstates

## Tests

```bash
pytest
```

The tests check the propagator against analytic results: norm conservation,
spreading of a free Gaussian, a coherent state in a harmonic oscillator and
the oscillator ground state energy from imaginary time evolution.

## Documentation

The API documentation is built with MkDocs from the docstrings:

```bash
pip install -e ".[docs]"
mkdocs serve
```

On every push to `main`, the `pages` workflow publishes it on GitHub Pages
together with the JupyterLite notebooks (under `lite/`).

## Releases

The changes are listed in [CHANGELOG.md](https://github.com/aeberspaecher/splitoperator/blob/main/CHANGELOG.md).
To release, bump the version in `pyproject.toml` and
`so_schroedinger/__init__.py`, add the changelog entry, and publish a GitHub
release with a tag `vX.Y.Z`; the `release` workflow then uploads the sdist
and wheel to PyPI with trusted publishing.

## Code style

The code follows PEP 8: `snake_case` for functions, methods, arguments and
variables, `CapWords` for classes. Physics symbols keep their usual capitals
(`V0`, `E0`, `R`, `delta_E`, ...). Linting and formatting use
[ruff](https://docs.astral.sh/ruff/), configured in `pyproject.toml`:

```bash
ruff check .
ruff format .
```

## License

BSD 2-Clause, see [LICENSE](https://github.com/aeberspaecher/splitoperator/blob/main/LICENSE).
