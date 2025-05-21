# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 11:00:09 2023

@author: d/dt Lucas
"""

try:
    from ..common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra, disp_op
except ImportError:
    from common.commonfunc import FT, IFT, fftshift, Pot, Sim, Fibra, disp_op
import numpy as np
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial         

#Differential equation
def dBdz(z, B, D, w, gammaw_eff, r, r_c):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain.

    This function calculates the right-hand side of the equation dB/dz = ...,
    considering linear and nonlinear effects in the frequency domain.

    Parameters:
        z (float): Propagation distance.
        B (ndarray): Field in the frequency domain at the current step.
        D (ndarray): Linear operator in the frequency domain.
        w (ndarray): Angular frequency array.
        gammaw_eff (ndarray): Effective nonlinear coefficient.
        r (ndarray): r factor.
        r_c (ndarray): Conjugate of r.

    Returns:
        ndarray: The derivative of B with respect to z in the frequency domain.
    """
    A_w = B * np.exp(D * z)  # Convert from B(w) -> A(w)
    B_w = r * A_w            # Note: B is the envelope, B_w is a term of the pcNLSE
    C_w = r_c * A_w
    op_nolin = 1j * gammaw_eff * FT(np.conj(IFT(C_w)) * IFT(B_w)**2) + \
               1j * np.conj(gammaw_eff) * FT(IFT(C_w)**2 * np.conj(IFT(B_w)))

    return np.exp(-D * z) * op_nolin


#Solver for the photon conversing Nonlinear Schrödinger Equation
def Solve_pcNLSE(sim: Sim, fib: Fibra, pulso_0, z_locs=None):
    """
    Simulate the evolution of a pulse using the pcNLSE (polarization-coupled Nonlinear Schrödinger Equation).

    This function solves the pcNLSE for a given input pulse, considering both linear
    and nonlinear effects.

    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse in the time domain.
        z_locs (int, optional): Number of z points (between 0 and L) where the solution is required. Default is None.

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

    # Calculate preliminary parameters for the pcNLSE.
    gammaw = fib.gamma + fib.gamma1 * (2 * np.pi * sim.freq)  # gamma(w), can be extended.
    r = (gammaw / (2 * np.pi * sim.freq + fib.omega0))**(1 / 4)
    r_c = np.conj(r)
    gammaw_eff = (1 / 2) * (gammaw * (2 * np.pi * sim.freq + fib.omega0)**3)**(1 / 4)

    # Introduce all parameters into the function, resulting in f(z, B) = dB/dz.
    f_B = partial(dBdz, D=D_w, w=2 * np.pi * sim.freq, gammaw_eff=gammaw_eff, r=r, r_c=r_c)

    # Integration tolerances (default values: rtol=1e-5, atol=1e-8).
    rtol = 1e-5
    atol = 1e-8

    # Use solve_ivp: Find the solution between 0 and L.
    if z_locs:  # If z_locs is provided, create an array with that many points to store the solution at each step.
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