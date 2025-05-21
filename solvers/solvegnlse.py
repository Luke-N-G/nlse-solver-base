# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:39:28 2023

@author: d/dt Lucas
"""

try:
    from ..common.commonfunc import FT, IFT, fftshift, Tools, Sim, Fibra, disp_op
except ImportError:
    from common.commonfunc import FT, IFT, fftshift, Tools, Sim, Fibra, disp_op
import numpy as np
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial         
from tqdm import tqdm


tls = Tools()

def dBdz(z, B, D, w, gamma, TR):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain.
    This consideres the approximate Raman effect (TR)

    This function calculates the right-hand side of the equation dB/dz = ..., 
    considering linear and nonlinear effects in the frequency domain.

    Parameters:
        z (float): Propagation distance.
        B (ndarray): Field in the frequency domain at the current step.
        D (ndarray): Linear operator in the frequency domain.
        w (ndarray): Angular frequency array.
        gamma (float): Nonlinear coefficient.
        TR (float): Raman response time.

    Returns:
        ndarray: The derivative of B with respect to z in the frequency domain.
    """
    A_w = B * np.exp(D * z)
    A_t = IFT(A_w)
    op_nolin = FT(1j * gamma * tls.pot(A_t) * A_t - 1j * gamma * TR * IFT(-1j * w * FT(tls.pot(A_t))) * A_t)
    return np.exp(-D * z) * op_nolin

def dBdz_raman(z, B, D, w, gamma, gamma1, RW):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain,
    considering the full Raman effect.

    Parameters:
        z (float): Propagation distance.
        B (ndarray): Field in the frequency domain at the current step.
        D (ndarray): Linear operator in the frequency domain.
        w (ndarray): Angular frequency array.
        gamma (float): Nonlinear coefficient.
        gamma1 (float): Raman coefficient.
        RW (ndarray): Raman response in the frequency domain.

    Returns:
        ndarray: The derivative of B with respect to z in the frequency domain.
    """
    A_w = B * np.exp(D * z)
    A_t = IFT(A_w)
    op_nolin = 1j * (gamma + gamma1 * w) * FT(A_t * IFT(RW * FT(np.abs(A_t)**2)))
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
        ndarray: Raman response in the frequency domain.
    """
    hR = np.zeros(len(T))
    hR[T >= 0] = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T[T >= 0] / tau2) * np.sin(T[T >= 0] / tau1)  # Define hR(T)
    hR[T < 0] = 0

    hR = fftshift(hR)     # Shift to start at the beginning of the temporal window
    hR = hR / np.sum(hR)  # Normalize such that int(hR) = 1
    hR_W = FT(hR)         # Convert hR to the frequency domain

    R_W = fR * hR_W + (1 - fR) * np.ones(len(T))
    return R_W

#Solver functions for Generalized Nonlinear Schrödinger Equation
def SolveNLS(sim: Sim, fib: Fibra, pulso_0, raman=False, z_locs=None, pbar=True,
             rtol=1e-3, atol=1e-6):
    """
    Simulate the evolution of a pulse using the Nonlinear Schrödinger Equation (NLSE).

    This function solves the NLSE for a given input pulse, considering both linear
    and nonlinear effects. It supports the inclusion of the Raman effect and allows
    for progress tracking with a progress bar.

    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse in the time domain.
        raman (bool, optional): If True, use the full Raman response. Default is False (use approximation).
        z_locs (int, optional): Number of z points (between 0 and L) where the solution is required. Default is None.
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

    # Introduce all parameters into the function, resulting in f(z, B) = dB/dz.
    if raman:
        RW = Raman(sim.tiempo, fR=fib.fR)
        f_B = partial(dBdz_raman, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, gamma1=fib.gamma1, RW=RW)
    else:
        f_B = partial(dBdz, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, TR=fib.TR)

    # Integration tolerances (default values: rtol=1e-3, atol=1e-6).
    if pbar:  # If progress bar is enabled.
        with tqdm(total=fib.L, unit="m") as pbar:
            if raman:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    return dBdz_raman(z, B, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, gamma1=fib.gamma1, RW=RW)
                dBdz_with_progress.prev_z = 0
            else:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    return dBdz(z, B, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, TR=fib.TR)
                dBdz_with_progress.prev_z = 0

            # Use solve_ivp: Find the solution between 0 and L.
            if z_locs:  # If z_locs is provided, the output array will have z_locs elements.
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:  # Without progress bar.
        if z_locs:  # If z_locs is provided, the output array will have z_locs elements.
            t_eval = np.linspace(0, fib.L, z_locs)
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)

    zlocs = sol["t"]  # Points of z where B(w, z) is calculated.
    ysol = sol["y"]  # Array, each element contains a subarray [B(w0, z0), B(w0, z1), ..., B(w0, zf)].

    print(sol["message"])

    # Build arrays of A(w) and A(t).
    ysol = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range(len(zlocs)):
        A_w[j, :] = A_w[j, :] * np.exp(D_w * zlocs[j])
    A_t = np.array([IFT(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w
