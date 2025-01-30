# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:00:09 2023

@author: d/dt Lucas
"""

from ..common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #Para resolver eq. diferenciales ordinarias
from functools import partial          #Permite evaluar parcialmente una función g(x,y) = partial(f(x,y,z), z = 2)

'''Función dB/dz = f(B,z)
Requiere: z y B libres, operador lineal D, frecuencia w, gamma y TR'''
def dBdz(z, B, D, w, gammaw_eff, r, r_c): #Todo lo que esta del lado derecho de dB/dz = ..., en el espacio de frecuencias.
    A_w = B * np.exp(D*z)  #Convertimos de B(w) -> A(w)
    B_w = r * A_w          #Cuidado con la notación! B es la envolvente, B_w 
    C_w = r_c * A_w    
    op_nolin = 1j * gammaw_eff * FT( np.conj(IFT(C_w)) * IFT(B_w)**2  ) + 1j * np.conj(gammaw_eff) * FT( IFT(C_w)**2 * np.conj(IFT(B_w))  ) 
    
    return np.exp(-D*z) * op_nolin

""" SOLVE_PCNLSE
solve_pcNLSE: Función para simular la evolución con la pcNLSE
sim:          Parámetros de la simulación
fib:          Parámetros de la fibra
pulso_0:      Pulso de entrada
raman:        Booleano, por defecto False. Si False, se usa aproximación de pulso ancho, si True respuesta completa.
z_locs:       Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
"""
def Solve_pcNLSE(sim: Sim, fib: Fibra, pulso_0, z_locs=None):
    #Calculamos el espectro inicial, es lo que vamos a evolucionar.
    espectro_0 = FT(pulso_0)

    #Calculamos el operador lineal
    D_w = 1j * fib.beta2/2 * (2*np.pi*sim.freq)**2 + 1j * fib.beta3/6 * (2*np.pi*sim.freq)**3 - fib.alpha/2
    
    if fib.betas:
        D_w = 0
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/np.math.factorial(i+2) * (2*np.pi*sim.freq)**(i+2)
        D_w = np.array(D_w)
    
    
    #Calculamos parámetros preliminares de la pcNLSE
    gammaw = fib.gamma + fib.gamma1 * (2*np.pi*sim.freq) #gamma(w), se podría extender
    r = ( gammaw / (2*np.pi*sim.freq + fib.omega0) )**(1/4)
    r_c = np.conj(r)
    gammaw_eff = (1/2)*( gammaw * (2*np.pi*sim.freq + fib.omega0)**3 )**(1/4)
    
    #Introducimos todos los parametros en la función, quedando f(z, B) = dB/dz
    f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gammaw_eff = gammaw_eff, r = r, r_c = r_c)

    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-5
    atol = 1e-8

    #Usamos solve_ivp: Buscamos solución entre 0 y L
    if z_locs: #Si le damos número a zlocs, armamos un array con esa cantidad de puntos, donde guardamos la solución en dicho paso
        t_eval = np.linspace(0,fib.L,z_locs)
        sol = solve_ivp(f_B, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
    else:
        sol = solve_ivp(f_B, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    print(sol["message"])

    #Armar array de arrays A(w) y A(t).
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