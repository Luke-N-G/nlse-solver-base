# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:46:16 2024

@author: d/dT Lucas
"""

from common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #Para resolver eq. diferenciales ordinarias
from functools import partial          #Permite evaluar parcialmente una función g(x,y) = partial(f(x,y,z), z = 2)
from tqdm import tqdm

'''Función dB/dz = f(B,z)
Requiere: z y B libres, operador lineal D, frecuencia w, gamma y ventana
A cada step restaura los bordes de simulación. 
'''

def dBdz(z,B,D,w,gamma,window):
    A_w = B * np.exp(D*z)
    A_t = IFT(A_w)
    #A_t[0:window] = A_t[window+1]
    #A_t[-window:] = A_t[-window-1]
    if gamma:
        op_nolin = FT( 1j * gamma * Pot(A_t) * A_t)
    return np.exp(-D*z) * op_nolin


""" SOLVE_DARK
solveNLS: Función para simular la evolución con la NLSE.
sim:      Parámetros de la simulación.
fib:      Parámetros de la fibra.
pulso_0:  Pulso de entrada.
window:   Int, por defecto 0. Número de puntos a restaurar en los bordes de la ventana temporal.
z_locs:   Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
pbar:     Booleano, por defecto True. Barra de progreso de la simulación.
"""

def Solve_dark(sim: Sim, fib: Fibra, pulso_0, window=0, z_locs=None, pbar=True):
    
    #Calculamos el espectro inicial, es lo que vamos a evolucionar.
    espectro_0 = FT(pulso_0)

    #Calculamos el operador lineal
    D_w = 1j * fib.beta2/2 * (2*np.pi*sim.freq)**2 + 1j * fib.beta3/6 * (2*np.pi*sim.freq)**3 - fib.alpha/2
    D_w = np.array(D_w)
    
    if fib.betas:
        D_w = 0
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/np.math.factorial(i+2) * (2*np.pi*sim.freq)**(i+2)
        D_w = np.array(D_w)
    
    #Introducimos todos los parametros en la función, quedando f(z, B) = dB/dz
    f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gamma=fib.gamma, window=window)

    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    #Usamos solve_ivp: Buscamos solución entre 0 y L
    
    if pbar:
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, window)
            dBdz_with_progress.prev_z = 0
    
            if z_locs:
                t_eval = np.linspace(0,fib.L,z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)
    else:
        if z_locs:
            t_eval = np.linspace(0,fib.L,z_locs)
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, window), [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, fib.gamma, window), [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)

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

    return zlocs, A_w, A_t #Nos devuelve: zlocs = Posiciones donde calculamos la solución, A_w = Matriz con la evolución del espectro, A_t = Matriz con la evolución del pulso

