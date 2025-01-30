# -*- coding: utf-8 -*-
"""
Created on Thursday June 12 11:05:09 2024

@author: d/dt Lucas
"""

from ..common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #Para resolver eq. diferenciales ordinarias
from functools import partial          #Permite evaluar parcialmente una función g(x,y) = partial(f(x,y,z), z = 2)
from tqdm import tqdm

'''Función dB/dz = f(B,z)
Requiere: z y B libres, operador lineal D, frecuencia w, gamma y TR'''

def dBdz(z,B,D,w,gamma,heavi):
    A_w = B * np.exp(D*z)
    A_t = IFT(A_w)
    if gamma:
        op_nolin = FT( heavi * A_t + 1j * gamma * Pot(A_t) * A_t)
    else:
        op_nolin = FT( heavi * A_t )
    return np.exp(-D*z) * op_nolin


""" SOLVE_BARRIERNLSE
Solve_barrierNLSE: Función para simular la evolución con la NLSE, pero considerando un cambio
de índice de refracción temporal, modelado a través de una función de Heaviside
sim:         Parámetros de la simulación
fib:         Parámetros de la fibra
pulso_0:     Pulso de entrada
delta_beta1: Diferencia de beta 1 entre el pulso incidente y la pared
TB:          Tiempo correspondiente al cambio de índice de refracción
betab:       Altura de la pared
z_locs:      Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
pbar:        Booleano, por defecto True. Barra de progreso de la simulación.
"""
def Solve_barrierNLSE(sim: Sim, fib: Fibra, pulso_0, delta_beta1, TB, betab, z_locs=None, pbar=True):
    #Calculamos el espectro inicial, es lo que vamos a evolucionar.
    espectro_0 = FT(pulso_0)

    #Calculamos el operador lineal
        
    D_w = 1j* delta_beta1 * (2*np.pi*sim.freq) + 1j * fib.beta2/2 * (2*np.pi*sim.freq)**2 + 1j * fib.beta3/6 * (2*np.pi*sim.freq)**3 - fib.alpha/2
    D_w = np.array(D_w)
    
    if fib.betas:
        D_w = 1j*delta_beta1 * (2*np.pi*sim.freq)
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/np.math.factorial(i+2) * (2*np.pi*sim.freq)**(i+2)
        D_w = np.array(D_w)
    
    #Construimos preliminares: Función de Heaviside
    heavi = 1j * betab * np.heaviside(sim.tiempo - TB,1)

    #Introducimos todos los parametros en la función, quedando f(z, B) = dB/dz
    f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gamma=fib.gamma, heavi=heavi)


    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    #Usamos solve_ivp: Buscamos solución entre 0 y L
    
    if pbar:
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, heavi)
            dBdz_with_progress.prev_z = 0
    
            if z_locs:
                t_eval = np.linspace(0,fib.L,z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)
    else:
        if z_locs:
            t_eval = np.linspace(0,fib.L,z_locs)
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, heavi), [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, heavi), [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)

    
    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    print(sol["message"])

    #Armamos array de arrays A(w) y A(t).
    
    ysol  = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range( len(zlocs) ):
        A_w[j,:] = A_w[j,:] * np.exp(D_w * zlocs[j])
    A_t = np.array([IFT(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w

#zlocs: Puntos donde esta calculada la solución
#A_w  : Matriz con la solución en el espectro [posición, frequencia]
#A_t  : Matriz con la solución en tiempo [posición, tiempo], se puede obtener a través de IFT(A_w)