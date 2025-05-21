# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:00:09 2023

@author: d/dt Lucas
"""

try:
    from ..common.commonfunc import FT, IFT, fftshift, Tools, Sim, Fibra, disp_op
except ImportError:
    from common.commonfunc import FT, IFT, fftshift, Tools, Sim, Fibra, disp_op
import numpy as np
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial     
from time import time
from tqdm import tqdm

tls = Tools()

#Differential equation
def dBdz(z, B, D, w, gammaw_eff, gammaw_eff_c, r, r_c, hR_W, fR):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain.

    This function calculates the right-hand side of the equation dB/dz = ..., 
    considering linear and nonlinear effects in the frequency domain.

    Parameters:
        z (float): Propagation distance.
        B (ndarray): Field in the frequency domain at the current step.
        D (ndarray): Linear operator in the frequency domain.
        w (ndarray): Angular frequency array.
        gammaw_eff (float): Effective nonlinear coefficient for the primary term.
        gammaw_eff_c (float): Conjugate of gammaw_eff
        r (ndarray): r factor.
        r_c (ndarray): Conjugate of r.
        hR_W (ndarray): Raman response in the frequency domain.
        fR (float): Fractional Raman contribution.

    Returns:
        ndarray: The derivative of B with respect to z in the frequency domain.
    """
    A_w = B * np.exp(D * z)  # Convert from B(w) -> A(w)
    B_w = r * A_w            # Note: B is the envelope, B_w and B_t are terms of the pcGNLSE
    C_w = r_c * A_w
    B_t = IFT(B_w)
    C_t = IFT(C_w)
    
    op_nolin = 1j * gammaw_eff * FT(np.conj(C_t) * B_t**2) + \
               1j * gammaw_eff_c * FT(C_t**2 * np.conj(B_t)) + \
               1j * gammaw_eff_c * 2 * fR * FT(B_t * IFT(hR_W * FT(tls.pot(B_t))) - B_t * tls.pot(B_t))
    op_nolin = np.array(op_nolin)
    
    return np.exp(-D * z) * op_nolin

#Raman response function
def Raman(T, tau1=12.2e-3, tau2=32e-3, fR=0.18):
    """
    Compute the Raman response in the frequency domain.

    This function calculates the Raman response based on the time-domain response hR(T),
    as described in Agrawal's Nonlinear Fiber Optics (page 38).

    Parameters:
        T (ndarray): Time array.
        tau1 (float): Raman parameter tau1 (default: 12.2 fs).
        tau2 (float): Raman parameter tau2 (default: 32 fs).
        fR (float): Fractional Raman contribution (default: 0.18).

    Returns:
        ndarray: Raman response in the frequency domain (hR_W).
    """
    hR = np.zeros(len(T))
    hR[T >= 0] = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T[T >= 0] / tau2) * np.sin(T[T >= 0] / tau1)  # Define hR(T)
    hR[T < 0] = 0

    hR = fftshift(hR)  # Shift to start at the beginning of the temporal window
    hR = hR / np.sum(hR)  # Normalize such that int(hR) = 1
    hR_W = FT(hR)  # Convert hR to the frequency domain

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

#Solver function for the photon conserving Generalized Nonlinear Schrödinger Equation (pcGNLSE)
def Solve_pcGNLSE(sim: Sim, fib: Fibra, pulso_0, z_locs=None, tau1=12.2e-3, pbar=True,
                  rtol=1e-3, atol=1e-6):
    """
    Simulate the evolution of a pulse using the pcGNLSE (photon conserving Generalized Nonlinear Schrödinger Equation).

    This function solves the pcGNLSE for a given input pulse, considering both linear
    and nonlinear effects, including Raman response. It supports progress tracking
    with a progress bar.

    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse in the time domain.
        z_locs (int, optional): Number of z points (between 0 and L) where the solution is required. Default is None.
        tau1 (float, optional): Raman parameter tau1 (default: 12.2 fs).
        pbar (bool, optional): If True, display a progress bar for the simulation. Default is True.
        rtol (float, optional): Relative tolerance for the ODE solver. Default is 1e-3.
        atol (float, optional): Absolute tolerance for the ODE solver. Default is 1e-6.

    Returns:
        tuple:
            - zlocs (ndarray): Points where the solution is calculated.
            - A_w (ndarray): Matrix with the solution in the frequency domain [position, frequency].

    Notes:
        - The solution in the time domain can be obtained using the inverse Fourier transform (IFT) of A_w.
    """
    # Calculate the initial spectrum, which will be evolved.
    espectro_0 = FT(pulso_0)

    # Calculate the linear operator.
    D_w = disp_op(sim, fib)
    
    # Calculate preliminary parameters for the pcGNLSE.
    gammaw = fib.gamma + fib.gamma1 * (2 * np.pi * sim.freq)  # gamma(w), can be extended.
    gammaw = np.array(gammaw, dtype=np.complex128)  # Ensure it's a complex array to avoid issues with roots.
    gammaw_eff = (1 / 2) * (gammaw * (2 * np.pi * sim.freq + fib.omega0)**3)**(1 / 4)
    gammaw_eff_c = np.conj(gammaw_eff)
    r = (gammaw / (2 * np.pi * sim.freq + fib.omega0))**(1 / 4)
    r_c = np.conj(r)
    
    # Calculate the Raman response.
    hR_W = Raman(sim.tiempo, fR=fib.fR, tau1=tau1) 

    # Introduce all parameters into the function, resulting in f(z, B) = dB/dz.
    f_B = partial(dBdz, D=D_w, w=2 * np.pi * sim.freq, gammaw_eff=gammaw_eff, 
                  gammaw_eff_c=gammaw_eff_c, r=r, r_c=r_c, hR_W=hR_W, fR=fib.fR)

    # Integration tolerances (default values: rtol=1e-3, atol=1e-6).
    if pbar:  # If progress bar is enabled.
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                """
                Wrapper function to compute dB/dz and update the progress bar.

                Parameters:
                    z (float): Current propagation distance.
                    B (ndarray): Current field in the frequency domain.

                Returns:
                    ndarray: Derivative of B with respect to z.
                """
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2 * np.pi * sim.freq, gammaw_eff, gammaw_eff_c, r, r_c, hR_W, fib.fR)
            dBdz_with_progress.prev_z = 0

            if z_locs:  # If z_locs is provided, the output array will have z_locs elements.
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
                
    else:  # Without progress bar.
        if z_locs:  # If z_locs is provided, the output array will have z_locs elements.
            t_eval = np.linspace(0, fib.L, z_locs)
            # Call solve_ivp, defining a lambda function f(z, B) with the remaining parameters already evaluated.
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, gammaw_eff,
                                              gammaw_eff_c, r, r_c, hR_W, fib.fR), 
                            [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, gammaw_eff,
                                              gammaw_eff_c, r, r_c, hR_W, fib.fR), 
                            [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)

    zlocs = sol["t"]  # Points of z where B(w, z) is calculated.
    ysol = sol["y"]  # Array, each element contains a subarray [B(w0, z0), B(w0, z1), ..., B(w0, zf)].
    print(sol["message"])

    # Build 2D arrays A(W) and A(T).
    ysol = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range(len(zlocs)):
        A_w[j, :] = A_w[j, :] * np.exp(D_w * zlocs[j])

    return zlocs, A_w
