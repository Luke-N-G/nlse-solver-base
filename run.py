# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:43:33 2023

@author: d/dt Lucas

-----------------------RUN.PY-----------------------------
Example for NLSE-solver
Soliton raman-induces self frequency shift (RIFS),
Nonlinear Fiber Optics, G. Agrawal, 4th edition, page 163.
----------------------------------------------------------

"""
#%%--------------Imports------------------
import numpy as np
from solvers.solvegnlse import SolveNLS
from common.commonfunc import Pulses, Sim, Fibra, Tools, Data, Units, fftshift, FT, IFT
from common.plotter import plotinst, plotcmap, plt

#Helper classes
pulse = Pulses()
tls = Tools()
units = Units()

#SIMULATION PARAMETERS
N = int(2**14)  #Number of temporal points
Tmax = 70       #Maximum temporal value: ps

c = 299792458 * (1e9)/(1e12)      #Speed of light: nm/ps
lambda0 = 1550                    #Central wavelength: nm
omega0  = 2*np.pi*c/lambda0       #Central angular frequency (rad/ps)

#FIBER PARAMETERS
L     = 400                       #Lfib:   m
b2    = -20e-3                    #Beta2:  ps^2/m
b3    = 0                         #Beta3:  ps^3/m
gam   = 1.4e-3                    #Gamma:  1/Wm
gam1  = gam/omega0*0              #Gamma1: 0
alph  = 0                         #alpha:  1/m
TR    = 3e-3                      #TR:     fs
fR    = 0.18                      #fR:     adimensional (0.18)

#PULSE PARAMETERS (Gaussian Pulse)
amp    = 1                        #Amplitude:  sqrt(W), Po = amp**2
ancho  = .4                       #Wdith T0:  ps
offset = 0                        #Offset:    ps
chirp  = 0                        #Chirp:     1/m
orden  = 1                        #Order

#Set object with given parameters:
sim   = Sim(N, Tmax)
fibra = Fibra(L, gamma=gam, gamma1=gam1, alpha=alph, lambda0=lambda0, betas=[b2,b3], TR=TR, fR=fR)

#Initialize pulse
#Supergaussian:
gauss_pulse = pulse.Sgaussian(sim.tiempo, amp, ancho, offset, chirp, orden)
#Soliton:
soliton = pulse.soliton(sim.tiempo, ancho, fibra.beta2, fibra.gamma, order = 1)


#%%------RUNNING THE SIMULATION------

#Propagating the soliton
A = Data(SolveNLS, sim, fibra, soliton, raman=True, z_locs=100)
# raman = True uses the full Raman response (otherwise, it uses an approximation)
# zlocs gives us the number of points in which the solution is evaluated (use at least 100 for good colormap plots)


#%%--------------Plots----------------

plotinst(A, Wlim=[-2,2], Tlim=[-7,7])
plotcmap(A, Wlim=[-2,2], Tlim=[-7,7])
