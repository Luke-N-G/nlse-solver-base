# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:43:33 2023

@author: d/dt Lucas
"""
#%%--------------Imports------------------
import numpy as np
from solvers.solvegnlse import SolveNLS
from common.commonfunc import SuperGauss, Soliton, Sim, Fibra, Pot, fftshift, FT, IFT
from common.plotter import plotinst, plotcmap, plt


#Parametros para la simulación
N = int(2**14)  #puntos
Tmax = 70       #ps

c = 299792458 * (1e9)/(1e12)      #Vel. luz: nm/ps

lambda0 = 1550                    #Longitud de onda central: nm
omega0  = 2*np.pi*c/lambda0

#Parametros para la fibra
L     = 400                       #Lfib:   m
b2    = -20e-3                    #Beta2:  ps^2/m
b3    = 0                         #Beta3:  ps^3/m
gam   = 1.4e-3                    #Gamma:  1/Wm
gam1  = gam/omega0*0              #Gamma1: 0
alph  = 0                         #alpha:  1/m
TR    = 3e-3*0                    #TR:     fs
fR    = 0.18*1                    #fR:     adimensional (0.18)

#Parámetros pulso gaussiano:
amp    = 1                        #Amplitud:  sqrt(W), Po = amp**2
ancho  = .4                       #Ancho T0:  ps
offset = 0                        #Offset:    ps
chirp  = 0                        #Chirp:     1/m
orden  = 1                        #Orden

#Cargo objetos con los parámetros:
sim   = Sim(N, Tmax)
fibra = Fibra(L, b2, b3, gam, gam1, alph, lambda0, TR, fR)

#Calculamos el pulso inicial
#Supergaussiano:
pulso = SuperGauss(sim.tiempo, amp, ancho, offset, chirp, orden)
#Solitón:
soliton = Soliton(sim.tiempo, ancho, fibra.beta2, fibra.gamma, orden = 1)


#%%------Corriendo la simulación------

zlocs, AW, AT = SolveNLS(sim, fibra, soliton, raman=True, z_locs=100)


#%%--------------Plots----------------

plotinst(sim, fibra, AT, AW, Wlim=[-1,1], Tlim=[-7,7])
plotcmap(sim, fibra, zlocs, AT, AW, Wlim=[-1,1], Tlim=[-7,7])