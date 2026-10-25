#!/usr/bin/env python
#-*- coding:utf-8 -*-

"""Animations for time-propagated wavepackets.
"""

#import matplotlib
#matplotlib.use('WXAgg') # blitting!
import matplotlib.pyplot as plt
from mpltools.animation import Animation
import numpy as np
from timeprop import *
from grid import *
from analyzer import *
from potentials import *
from wavepackets import *

from scipy.signal import correlate


class AnimateProp(Animation):
    """Animate wavepacket propagation.
    """

    def __init__(self, grid, tmax, dt, plotEvery=1, warnNorm=False,
                 analyzers=None, scalePot=1.0, updatePotential=False):
        """Create an animation object.

        Parameters
        ----------
        grid : Grid instance
            The grid to be used in iteration.
        tmax : float
            Final time.
        dt : float
            Time step.
        plotEvery : int, optional
            Plot after this many time steps.
        warnNorm : boolean, optional
            True if warnings about state norm loss should be printed.
        analyzers : list, optional
            List of Analyzer instances.
        scalePot : float, optional
            Scale the potential by this factor.
        """

        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.tight_layout()
        self.xVals = grid.xVals
        self.grid = grid
        self.tmax, self.dt = tmax, dt
        self.plotEvery = int(plotEvery)
        self.warnNorm = warnNorm
        if(np.imag(dt) != 0.0):  # TODO: investigate this once imaginary time evolution has arrived
            self.num_frames = -int(tmax/np.imag(dt))  # TODO: do we need a minus sign here?
        else:
            self.num_frames = int(tmax/dt)
        self.initNorm = grid.psiNorm
        self.psiInit = grid.psi.copy()  # save psi_0 for plotting
        self.analyzers = analyzers
        self.scalePot = float(scalePot)
        self.updatePotential = updatePotential

    def init_background(self):
        """Draw potential landspace and initial wavepacket.
        """

        if(not self.updatePotential):
            self.ax.plot(self.xVals, self.scalePot*np.real(grid.V), lw=1, c="blue")
            self.ax.plot(self.xVals, self.scalePot*np.imag(grid.V), lw=0.5, ls="--", c="blue")
            self.ax.fill_between(self.xVals, self.scalePot*np.real(grid.V), 0.0, facecolor="blue", alpha=0.3)
        self.ax.plot(grid.xVals, np.abs(self.psiInit)**2, c="gray", lw=1, ls="-", label=r"$\vert\psi_0\vert^2$", alpha=0.7)

    def update(self):

        stepcount = 0

        if(self.updatePotential):
            self.line, = self.ax.plot(self.grid.xVals, np.abs(self.grid.psi)**2, c="red", lw=1.5, ls="-", label=r"$\vert\psi(t)\vert^2$")
            self.line2, = self.ax.plot(self.xVals, self.scalePot*np.real(grid.V), lw=1, c="blue")
        else:
            self.line, = self.ax.plot(self.grid.xVals, np.abs(self.grid.psi)**2, c="red", lw=1.5, ls="-", label=r"$\vert\psi(t)\vert^2$")

        self.ax.legend()

        while(self.grid.t < self.tmax):

            if(self.analyzers is not None):  # if analyzers are given, hand data to them
                for i in range(len(self.analyzers)):
                    self.analyzers[i].addData(grid.psi, grid.V, grid.t)

            stepcount += 1
            prop.step(grid, dt=dt)

            if(self.warnNorm):
                if(abs(self.grid.psiNorm - self.initNorm) > maxDeviation):
                    print("At t = %s, the norm is (now %s) not conserved anymore! Maybe consider a smaller time step."%(self.grid.t, self.grid.psiNorm))

            if(stepcount % self.plotEvery == 0):
                if(self.updatePotential):
                    self.line.set_ydata(np.abs(self.grid.psi)**2)
                    self.line2.set_ydata(self.scalePot*np.real(grid.V))
                    yield self.line, self.line2
                else:
                    self.line.set_ydata(np.abs(self.grid.psi)**2)
                    yield self.line,


#-----------------------------------------------------------------------
if(__name__ == '__main__'):

    # all times, lengths and energies are in atomic units (m=hbar=epsilon_0=e=1)
    samples = 2**12  # use powers of 2
    tmax = 20
    dt = 0.0005
    maxDeviation = 1E-4  # max allowed deviation from initial state's norm
    size = [-50, +50]
    renormSteps = 0

    # Initial wavefunctions:
    #psi0 = CoherentState(x0=0.0, omega=3.0, p0=20.0)
    #psi02 = CoherentState(x0=0.0, omega=3.0, p0=-20.0)
    psi0 = Gaussian1dEnergy(x0=0, E0=10, deltaE=10.0, direction=+1)
    #psi0 = Gaussian1d(x0=0.0, sigma=1.0, p0=+2)
    psi0 = CoherentState(x0=0.0, omega=3.0, p0=20.0)
    #psi0 = WavepacketSuperposition1d([psi01, psi02])

    # Potentials:
    #pot2 = WellPotential1d(pos=0.0, width=30.0, V0=0.0, Vwall=500.0, d=-0.5, Vext=0.0)
    #pot = BoxPotential(pos=0.0, width=5.0, height=+600)
    #pot = ConstantPotential(0)
    pot = RandomPotential1d(lambda x: 1/np.sqrt(2*np.pi*1)*np.exp(-x**2/(2*1**2)), x=np.linspace(size[0], size[1], samples), V0=500)
    #pot1 = SmoothedBoxPotential(pos=-3.0, width=0.1, height=+200.0, exponent=6)
    #pot = SmoothedBoxPotential(pos=+8.0, width=0.1, height=+200.0, exponent=6)
    #pot = HarmOscPot(3)
    #pot2 = SquareWell1d(x0=0.0, omega=10.0, width=5.0, V0=0, well=True)
    #pot = InterpolatingPotential1d([(-2,-10), (+2,-30), (0,0), (-10, 60), (+10, 60)])
    #pot = PotentialSuperposition([pot0, pot1, pot2])
    #potEck1 = EckartPotential1d(x0=6, V0=140, alpha=2.0)
    #potEck2 = EckartPotential1d(x0=-6, V0=140, alpha=2.0)
    #potGauss3 = GaussBarrier1d(x0=0, V0=-100, sigma=0.5)
    #potGauss1 = GaussBarrier1d(x0=-10, V0=100, sigma=0.4)
    #potGauss2 = GaussBarrier1d(x0=+10, V0=100, sigma=0.20)
    #potRamp1 = MovingRamp1d(x0=-7, V0=400, alpha=0.01, uphill=False, motionType="sin",
                            #frequency=10.0, motionAmpl=1.0, phase=0)
    #potRamp2 = MovingRamp1d(x0=+7, V0=400, alpha=0.01, uphill=True, motionType="sin",
                            #frequency=10.0, motionAmpl=1.0, phase=0.0)
    #pot = PotentialSuperposition1d([potRamp1, potRamp2])
    #pot = GaussBarrier1d(x0=0.0, V0=-100, sigma=5)
    #pot = MicrocyvityRadialPotential(R=5, n=3.3, k=1, m=10)
    #pot = PotentialSuperposition1d([pot1, pot2])

    # Absorbing potential:
    #absorber = None
    absorber = ExpAbsorber1d(Areal=-0, Aimag=-500, lreal=1.0, limag=2.0, x0=size[0], x1=size[1])

    # Grid:
    grid = Grid1d(size=size, gridsamples=samples, potential=pot, psi0=psi0, absorber=absorber)

    # Test plot of the potentials autocorrelation function:
    plt.plot(grid.xVals, np.real(correlate(np.real(grid.V), np.real(grid.V), mode="same")), lw=2, c="b")
    plt.title("Autocorrelation function of potential")
    plt.show()

    # recompute reasonable time step from potential elevation:
    print("dt_old = %s"%dt)
    print("\Delta V = %s"%(np.max(np.real(grid.V)) - np.min(np.real(grid.V))))
    dt = grid.suggestTimeStep()  # -1j*... for imaginary time evolution!
    print("dt_new = %s"%dt)

    prop = SplitOperator(renormalizeEveryNSteps=renormSteps)

    # normalize the initial state:
    grid.renormalizePsi()

    initNorm = grid.psiNorm
    print("Norm of initial state = %s"%initNorm)

    # uncomment to train anfft with current FFT length / grid size - also import fft from anfft!
    #bla = fft(np.random.rand(grid.gridSamples), measure=True)

    # pick an analyzer:
    #analyzer = None
    #analyer = AutoCorrelation(grid.xVals, grid.psi, grid.V)
    corrAnalyzer = AutoCorrelation(grid.xVals, grid.psi, grid.V)
    observAnalyzer = Observables(grid.xVals, grid.psi, grid.V)
    eigenAnalyzer = EigenfunctionBuilder(grid.xVals, grid.psi, grid.V)
    analyzers = [corrAnalyzer, observAnalyzer]

    anim = AnimateProp(grid, tmax, dt, plotEvery=7, warnNorm=(absorber is None),
                       analyzers=analyzers, scalePot=1/np.max(np.abs(np.real(grid.V))),
                       updatePotential=pot.isTimeDependent)
    anim.animate(blit=False, interval=40.0)  # increase interval in case the animation sucks, goal should be 40ms
    # Note: `save` and `show` don't play nicely together. Use one at a time.
    #osc.save('movie.avi', fps=30, bitrate=200)
    plt.show()

    print("Norm of final state = %s"%grid.psiNorm)

    if(analyzers is not None):
        print("Processing data for analysis")
        for i in range(len(analyzers)):
            analyzers[i].processData()
        print("\tdone.")

    corrAnalyzer.plotCorr()
    corrAnalyzer.plotSpectrum()
    observAnalyzer.plotData()
    #eigenAnalyzer.plotEigenfunction(-67.6)

    # plot groundstate WFs - test imaginary time evolution:
    #plt.plot(grid.xVals, np.abs(grid.psi)**2, lw=2, c="b", label="Imag. time")
    ##plt.plot(grid.xVals, np.abs(np.sqrt(2)*np.sin(np.pi/1.0*grid.xVals))**2, lw=2, c="r", ls="--", label="Analyt.") # Well around x=0.5 with width = 1.0
    #plt.plot(grid.xVals, np.abs((3/np.pi)**0.25 * np.exp(-3*(grid.xVals**2)/2))**2, lw=2, c="r", ls="--", label="Analyt.") # HarmOsc
    #plt.legend()
    #plt.show()
