# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:46:16 2024

@author: d/dT Lucas
"""
try:
    from ..common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra, disp_op
except ImportError:
    from common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra
import numpy as np
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial         
from tqdm import tqdm


#Differential equation
def dBdz(z, B, D, w, gamma, window):
    """
    Compute the derivative of B with respect to z (dB/dz) as a function of B and z.

    This function calculates the evolution of a field in the frequency domain,
    considering both linear and nonlinear effects. At each step, it restores the
    simulation window edges if necessary.

    Parameters:
        z (float): The propagation distance.
        B (ndarray): The field in the frequency domain at the current step.
        D (ndarray): The linear operator in the frequency domain.
        w (ndarray): The angular frequency array.
        gamma (float): The nonlinear coefficient. If None, nonlinear effects are ignored.
        window (int): Number of points to restore at the edges of the temporal window.

    Returns:
        ndarray: The derivative of B with respect to z (dB/dz) in the frequency domain.
    """
    A_w = B * np.exp(D * z)
    A_t = IFT(A_w)
    # A_t[0:window] = A_t[window+1]  # Restore the left edge of the temporal window
    # A_t[-window:] = A_t[-window-1]  # Restore the right edge of the temporal window
    if gamma:
        op_nolin = FT(1j * gamma * Pot(A_t) * A_t)
    return np.exp(-D * z) * op_nolin

#Solver function for dark solitons, allowing for window restoration at the edges.
def Solve_dark(sim: Sim, fib: Fibra, pulso_0, window=0, z_locs=None, pbar=True):
    """
    Simulate the evolution of a pulse using the Nonlinear Schrödinger Equation (NLSE).

    This function solves the NLSE for a given input pulse, considering both linear
    and nonlinear effects. It also allows for restoring the edges of the temporal
    window during the simulation.

    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse in the time domain.
        window (int, optional): Number of points to restore at the edges of the temporal window. Default is 0.
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
    D_w = disp_op(sim, fib)

    # Introduce all parameters into the function, resulting in f(z, B) = dB/dz
    f_B = partial(dBdz, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, window=window)

    # Integration tolerances (Standard tolerances: rtol=1e-5, atol=1e-8)
    rtol = 1e-3
    atol = 1e-6

    # Use solve_ivp: Find the solution between 0 and L
    if pbar:
        with tqdm(total=fib.L, unit="m") as pbar:
            def dBdz_with_progress(z, B):
                pbar.update(abs(z - dBdz_with_progress.prev_z))
                dBdz_with_progress.prev_z = z
                return dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, window)
            dBdz_with_progress.prev_z = 0

            if z_locs:
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:
        if z_locs:
            t_eval = np.linspace(0, fib.L, z_locs)
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, window),
                            [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(lambda z, B: dBdz(z, B, D_w, 2 * np.pi * sim.freq, fib.gamma, window),
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