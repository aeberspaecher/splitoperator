# Changelog

## Unreleased

- Phase space: `so.wigner()` and `so.husimi()` for 1d grids, the analyzers
  `Wigner` and `Husimi`, `so.plotting.plot_phase_space()`, and a phase space
  notebook comparing quantum and classical flow.
- Absorbers: `ManolopoulosAbsorber1d` (transmission-free), `PowerAbsorber1d`
  and `MaskAbsorber1d` (cos^(1/8) mask), and a notebook that measures
  reflection and transmission of the absorbers versus energy.

- New notebooks: double slit in 2d; double well with Rabi oscillations and
  coherent destruction of tunneling; dynamical localization in the kicked rotor.

## 0.5.0

First release on PyPI.

- Speed: FFTs through `scipy.fft` on all cores (pluggable, e.g. pyFFTW), and
  merged potential half steps in `propagate()`; about 3× faster in 2d.
- `SplitOperator(order=4)` and `order=6`: higher-order splittings by
  composing Strang steps (Yoshida's triple jump).
- `dtype=np.complex64` for grids: single precision, half the memory.
- Time-dependent potentials are propagated to second order (V(t) and
  V(t+dt) in the two half steps).
- Energy `<H>` on grids and in `Observables`; `ground_state()` and
  `lowest_eigenstates()` from imaginary time evolution.
- `so.plotting`: one-line plots of wavefunctions, potentials, momentum
  distributions and eigenstates.
- Live mode for the animations (`live=True`), which steps while drawing.
- New notebooks: Talbot carpet, Anderson localization; sliders in the
  tunneling notebook; splitting orders in the harmonic oscillator notebook.
- Notebooks run on Binder, Colab and in the browser with JupyterLite.
- API documentation with MkDocs, a benchmark script, and a PyPI release
  workflow.

## 0.4.0

The analyzers take the grid instead of `(psi, V, t)`, no longer plot, and
return arrays.

## 0.3.0

The former camelCase API was renamed to PEP 8 names, e.g. `psiNorm` →
`psi_norm`, `expectationX()` → `expectation_x()`, `nSteps` → `n_steps`,
`nFrames`/`stepsPerFrame` → `n_frames`/`steps_per_frame`.
