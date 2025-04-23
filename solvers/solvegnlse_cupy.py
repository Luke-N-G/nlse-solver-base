# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:39:28 2023

@author: d/dt Lucas
"""

from ..common.commonfunc import FT, IFT, fftshift, Tools, Sim, Fibra, disp_op
import numpy as np
import cupy as cp
from scipy.integrate import solve_ivp  #Para resolver eq. diferenciales ordinarias
from functools import partial          #Permite evaluar parcialmente una función g(x,y) = partial(f(x,y,z), z = 2)
from tqdm import tqdm


tls = Tools()

#Cupy FT
def FT_cu(pulso):
    if pulso.ndim == 1:
        return cp.fft.ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return cp.fft.ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")
        
#Cupy IFT
def IFT_cu(espectro):
    if espectro.ndim == 1:
        return cp.fft.fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return cp.fft.fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

'''Función dB/dz = f(B,z)
Requiere: z y B libres, operador lineal D, frecuencia w, gamma y TR'''
def dBdz(z,B,D,w,gamma,TR): #Todo lo que esta del lado derecho de dB/dz = ..., en el espacio de frecuencias.
    A_w = B * np.exp(D*z)
    A_t = IFT( A_w )
    op_nolin = FT( 1j*gamma*tls.pot(A_t)*A_t - 1j*gamma*TR*IFT( -1j*w*FT(tls.pot(A_t)) )*A_t)
    return np.exp(-D*z) * op_nolin

def dBdz_raman_cupy(z,B,D,w,gamma,gamma1,RW): #RW es la respuesta Raman en frecuencia (lo que calcula la función Raman(T))
    B = cp.array(B)
    D = cp.array(D)
    w = cp.array(w)
    gamma = cp.array(gamma)
    gamma1 = cp.array(gamma1)
    RW = cp.array(RW)
    A_w = B * cp.exp(D*z)
    A_t = cp.fft.fft(A_w) / len(A_w)
    op_nolin = 1j*(gamma + gamma1*w) * FT_cu(A_t * IFT_cu( RW * FT_cu(np.abs(A_t)**2)  ) )
    retr = cp.exp(-D*z) * op_nolin
    return cp.asnumpy(retr)

def dBdz_raman(z,B,D,w,gamma,gamma1,RW): #RW es la respuesta Raman en frecuencia (lo que calcula la función Raman(T))
    A_w = B * np.exp(D*z)
    A_t = IFT( A_w )
    op_nolin = 1j*(gamma + gamma1*w) * FT(A_t * IFT( RW * FT(np.abs(A_t)**2)  ) )
    return np.exp(-D*z) * op_nolin


def Raman(T, tau1=12.2e-3, tau2=32e-3, fR=0.18): #Agrawal pag.38: t1 = 12.2s fs, t2 = 32 fs
    hR = np.zeros( len(T) )
    hR[T>=0] = (tau1**2+tau2**2)/(tau1*tau2**2) * np.exp(-T[T>=0]/tau2) * np.sin(T[T>=0]/tau1) #Definimos el hR(T)
    hR[T<0]  = 0
    
    hR = fftshift(hR)  #Shifteamos para que la respuesta empiece al principio de la ventana temporal
    hR = hR/np.sum(hR) #Normalizamos, tal que int(hR) = 1    
    hR_W = FT(hR)      #Pasamos el hR_W a frecuencia

    R_W = fR * hR_W + (1-fR)*np.ones( len(T) )
    return R_W

""" SOLVENLS
solveNLS: Función para simular la evolución con la NLSE
sim:      Parámetros de la simulación
fib:      Parámetros de la fibra
pulso_0:  Pulso de entrada
raman:    Booleano, por defecto False. Si False, se usa aproximación de pulso ancho, si True respuesta completa.
z_locs:   Int, opcional. En cuantos puntos de z (entre 0 y L) se requiere la solución. 
pbar:     Booleano, por defecto True. Barra de progreso de la simulación.
cupy:     Booleano, por defecto False. Usa cupy para el operador no lineal.
"""
def SolveNLS_cu(sim: Sim, fib: Fibra, pulso_0, raman=False, z_locs=None, pbar=True, cupy=False):

    #Calculamos el espectro inicial, es lo que vamos a evolucionar.
    espectro_0 = FT(pulso_0)

    #Calculamos el operador lineal
    D_w = disp_op(sim, fib)

    #Introducimos todos los parametros en la función, quedando f(z, B) = dB/dz
    if raman:
      RW = Raman(sim.tiempo, fR = fib.fR)
      f_B = partial(dBdz_raman, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, gamma1 = fib.gamma1, RW = RW)
    else:
      f_B = partial(dBdz, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, TR = fib.TR)

    #Tolerancias para integrar (Tolerancias estandar: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    if pbar:  # Por si queremos la barra de progreso
        with tqdm(total=fib.L, unit="m") as pbar:
            if raman:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    if cupy:
                        return dBdz_raman_cupy(z, B, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, gamma1 = fib.gamma1, RW = RW)
                    else:
                        return dBdz_raman(z, B, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, gamma1 = fib.gamma1, RW = RW)
                dBdz_with_progress.prev_z = 0
            else:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    return dBdz(z, B, D = D_w, w = 2*np.pi*sim.freq, gamma = fib.gamma, TR = fib.TR)
                dBdz_with_progress.prev_z = 0

            # Usamos solve_ivp: Buscamos solución entre 0 y L
            if z_locs:  # Si le pasamos un valor de z_locs: El array de salida tendra z_locs elementos
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:  # Sin la barra de progreso
        if z_locs:  # Si le pasamos un valor de z_locs: El array de salida tendra z_locs elementos
            t_eval = np.linspace(0, fib.L, z_locs)
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    

    zlocs = sol["t"]  #Puntos de z donde tenemos B(w,z)
    ysol  = sol["y"]  #Array, en cada elemento se tiene un subarray [B(w0,z0), B(w0,z1), ..., B(w0,zf)]
    print(sol["message"])

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
