# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:00:09 2023

@author: d/dt Lucas
"""

from common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #Para resolver eq. diferenciales ordinarias
from functools import partial          #Permite evaluar parcialmente una función g(x,y) = partial(f(x,y,z), z = 2)
from time import time
from tqdm import tqdm

'''Función dB/dz = f(B,z)
Requiere: z y B libres, operador lineal D, frecuencia w, gamma y TR'''
def dBdz(z, B, D, w, gammaw_eff, gammaw_eff_c, r, r_c, hR_W, fR): #Todo lo que esta del lado derecho de dB/dz = ..., en el espacio de frecuencias.
    A_w = B * np.exp(D*z)  #Convertimos de B(w) -> A(w)
    B_w = r * A_w          #Cuidado con la notación! B es la envolvente, B_w y B_t son los términos de la pcGNLSE
    C_w = r_c * A_w
    B_t = IFT(B_w)
    C_t = IFT(C_w)  
    
    op_nolin = 1j * gammaw_eff * FT( np.conj(C_t) * B_t**2  ) + 1j * gammaw_eff_c * FT( C_t**2 * np.conj(B_t) ) + \
        1j * gammaw_eff_c *2* fR * FT( B_t * IFT( hR_W  * FT( Pot(B_t) ) ) - B_t * Pot(B_t)  )
    op_nolin = np.array(op_nolin)
    
    return np.exp(-D*z) * op_nolin

def Raman(T, tau1=12.2e-3, tau2=32e-3, fR=0.18): #Agrawal pag.38: t1 = 12.2s fs, t2 = 32 fs
    hR = np.zeros( len(T) )
    hR[T>=0] = (tau1**2+tau2**2)/(tau1*tau2**2) * np.exp(-T[T>=0]/tau2) * np.sin(T[T>=0]/tau1) #Definimos el hR(T)
    hR[T<0]  = 0
    
    hR = fftshift(hR)  #Shifteamos para que la respuesta empiece al principio de la ventana temporal
    hR = hR/np.sum(hR) #Normalizamos, tal que int(hR) = 1    
    hR_W = FT(hR)      #Pasamos el hR_W a frecuencia

    return hR_W

""" SOLVE_PCGNLSE
solve_pcGNLSE: Función para simular la evolución con la NLSE
sim:           Parámetros de la simulación
fib:           Parámetros de la fibra
pulso_0:       Pulso de entrada
raman:         Booleano, por defecto False. Si False, se usa aproximación de pulso ancho, si True respuesta completa.
z_locs:        Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
pbar:          Booleano, por defecto True. Barra de progreso de la simulación.
"""
def Solve_pcGNLSE(sim: Sim, fib: Fibra, pulso_0, z_locs=None, tau1=12.2e-3, pbar=True):
    
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
    
    #Calculamos parámetros preliminares de la pcNLSE
    gammaw = fib.gamma + fib.gamma1 * (2*np.pi*sim.freq) #gamma(w), se podría extender
    gammaw = np.array(gammaw, dtype=np.complex128) #Pedimos que sea un array complejo, para que no se rompan las raices de más adelante.
    gammaw_eff = (1/2)*( gammaw * (2*np.pi*sim.freq + fib.omega0)**3 )**(1/4)
    gammaw_eff_c = np.conj(gammaw_eff)
    r = ( gammaw / (2*np.pi*sim.freq + fib.omega0) )**(1/4)
    r_c = np.conj(r)
    
    #Calculamos la respuesta Raman
    hR_W = Raman(sim.tiempo, fR = fib.fR, tau1=tau1) 
    

    #Introducimos todos los parametros en la función, quedando f(z, B) = dB/dz
    f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gammaw_eff = gammaw_eff, gammaw_eff_c = gammaw_eff_c, r = r, r_c = r_c, hR_W = hR_W, fR = fib.fR)


    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    if pbar: #Por si queremos la barra de progreso
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2*np.pi*sim.freq, gammaw_eff, gammaw_eff_c, r, r_c, hR_W, fib.fR)
            dBdz_with_progress.prev_z = 0

            if z_locs:
                t_eval = np.linspace(0,fib.L,z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)
                
    else: #Sin la barra de progreso
        if z_locs:
            t_eval = np.linspace(0,fib.L,z_locs)
            #Llamamos a solve_ivp, nos definimos una función f(z,B) con el resto de los parametros ya evaluados
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, gammaw_eff,
                            gammaw_eff_c, r, r_c, hR_W, fib.fR), 
                            [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, 
                            t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2*np.pi*sim.freq, gammaw_eff,
                            gammaw_eff_c, r, r_c, hR_W, fib.fR), 
                            [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)

# =============================================================================
#     #Usamos solve_ivp: Buscamos solución entre 0 y L
#     if z_locs: #Si le damos número a zlocs, armamos un array con esa cantidad de puntos, donde guardamos la solución en dicho paso
#         t_eval = np.linspace(0,fib.L,z_locs)
#         sol = solve_ivp(f_B, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol, t_eval=t_eval)
#     else:
#         sol = solve_ivp(f_B, [0, fib.L], y0 = espectro_0, rtol = rtol, atol = atol)
# =============================================================================

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    print(sol["message"])


    #Armamos arrays 2D A(W) y A(T)
    ysol  = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range( len(zlocs) ):
        A_w[j,:] = A_w[j,:] * np.exp(D_w * zlocs[j])
    A_t = np.array([IFT(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w, A_t

#zlocs: Puntos donde esta calculada la solución
#A_w  : Matriz con la solución en el espectro [posición, frequencia]
#A_t  : Matriz con la solución en tiempo [posición, tiempo]
