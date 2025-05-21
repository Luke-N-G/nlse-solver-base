# -*- coding: utf-8 -*-
"""
Created on Thursday June 12 11:05:09 2024

@author: d/dt Lucas
"""
try:
    from ..common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
except ImportError:
    from common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial          
from tqdm import tqdm


def dBdz(z, B, D, w, gamma, heavi):
    """
    Compute the derivative of B with respect to z (dB/dz) as a function of B and z.

    This function calculates the evolution of a field in the frequency domain,
    considering both linear and nonlinear effects.

    Parameters:
        z (float): The propagation distance.
        B (ndarray): The field in the frequency domain at the current step.
        D (ndarray): The linear operator in the frequency domain.
        w (ndarray): The angular frequency array.
        gamma (float): The nonlinear coefficient. If None, nonlinear effects are ignored.
        heavi (ndarray): The Heaviside function or a similar operator applied in the time domain.

    Returns:
        ndarray: The derivative of B with respect to z (dB/dz) in the frequency domain.
    """
    # Get the A field from the B field
    A_w = B * np.exp(D * z)

    # Transform the field to the time domain
    A_t = IFT(A_w)

    # Compute the nonlinear operator in the frequency domain
    if gamma:
        # Include nonlinear effects: Heaviside function and nonlinear term
        op_nolin = FT(heavi * A_t + 1j * gamma * Pot(A_t) * A_t)
    else:
        # Only include the Heaviside function if gamma is not provided
        op_nolin = FT(heavi * A_t)

    # Return the derivative of B with respect to z in the frequency domain
    return np.exp(-D * z) * op_nolin


def Solve_barrierNLSE(sim: Sim, fib: Fibra, pulso_0, delta_beta1, TB, betab, z_locs=None, pbar=True):
    
    """
    Solve_barrierNLSE: Function to simulate the evolution with the NLSE, considering a temporal 
    refractive index change modeled through a Heaviside function.
    
    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse.
        delta_beta1 (float): Difference in beta1 between the incident pulse and the barrier.
        TB (float): Time corresponding to the refractive index change.
        betab (float): Height of the barrier.
        z_locs (int, optional): Number of z points (between 0 and L) where the solution is required. Default is None.
        pbar (bool, optional): If True, display a progress bar for the simulation. Default is True.
    
    Returns:
        tuple:
            - zlocs (ndarray): Points where the solution is calculated.
            - A_w (ndarray): Matrix with the solution in the frequency domain [position, frequency].
            - A_t (ndarray): Matrix with the solution in the time domain [position, time], can be obtained via IFT(A_w).
    """
    
    # Calculate the initial spectrum, which will be evolved.
    espectro_0 = FT(pulso_0)

    # Calculate the linear operator
    D_w = 1j * delta_beta1 * (2 * np.pi * sim.freq) + 1j * fib.beta2 / 2 * (2 * np.pi * sim.freq)**2 + \
          1j * fib.beta3 / 6 * (2 * np.pi * sim.freq)**3 - fib.alpha / 2
    D_w = np.array(D_w)
    
    if fib.betas:
        D_w = 1j * delta_beta1 * (2 * np.pi * sim.freq)
        for i in range(len(fib.betas)):
            D_w = D_w + 1j * fib.betas[i] / np.math.factorial(i + 2) * (2 * np.pi * sim.freq)**(i + 2)
        D_w = np.array(D_w)
    
    # Construct preliminaries: Heaviside function
    heavi = 1j * betab * np.heaviside(sim.tiempo - TB, 1)

    # Integration tolerances (Standard tolerances: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    # Use solve_ivp: Find the solution between 0 and L
    if pbar:
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, heavi)
            dBdz_with_progress.prev_z = 0
    
            if z_locs:
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:
        if z_locs:
            t_eval = np.linspace(0, fib.L, z_locs)
            #Partial evaluation of the function
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, heavi), 
                            [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, heavi), 
                            [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)

    zlocs = sol["t"]  # Points of z where B(w, z) is calculated
    ysol = sol["y"]   # Array, each element contains a subarray [B(w0, z0), B(w0, z1), ..., B(w0, zf)]
    print(sol["message"])

    # Build arrays of A(w) and A(t)
    ysol = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range(len(zlocs)):
        A_w[j, :] = A_w[j, :] * np.exp(D_w * zlocs[j])
    A_t = np.array([IFT(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w