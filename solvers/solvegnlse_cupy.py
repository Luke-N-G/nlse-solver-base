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
import cupy as cp
from scipy.integrate import solve_ivp  #RK45 solver
from functools import partial         
from tqdm import tqdm


tls = Tools()

#CuPy FFT
def FT_cu(pulso):
    """
    Perform the Fourier Transform (FT) using CuPy with the "-iw" convention.

    Parameters:
        pulso (ndarray): Input array representing the time-domain signal.
                         Can be 1D or 2D.

    Returns:
        ndarray: Fourier-transformed signal in the frequency domain.

    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if pulso.ndim == 1:
        return cp.fft.ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return cp.fft.ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

#CuPy IFFT
def IFT_cu(espectro):
    """
    Perform the Inverse Fourier Transform (IFT) using CuPy with the "-iw" convention.

    Parameters:
        espectro (ndarray): Input array representing the frequency-domain signal.
                            Can be 1D or 2D.

    Returns:
        ndarray: Inverse Fourier-transformed signal in the time domain.

    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if espectro.ndim == 1:
        return cp.fft.fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return cp.fft.fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

def dBdz(z, B, D, w, gamma, TR):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain.

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

def dBdz_raman_cupy(z, B, D, w, gamma, gamma1, RW):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain,
    considering the Raman effect, using CuPy for GPU acceleration.

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
    B = cp.array(B)
    D = cp.array(D)
    w = cp.array(w)
    gamma = cp.array(gamma)
    gamma1 = cp.array(gamma1)
    RW = cp.array(RW)
    A_w = B * cp.exp(D * z)
    A_t = cp.fft.fft(A_w) / len(A_w)
    op_nolin = 1j * (gamma + gamma1 * w) * FT_cu(A_t * IFT_cu(RW * FT_cu(np.abs(A_t)**2)))
    retr = cp.exp(-D * z) * op_nolin
    return cp.asnumpy(retr)

def dBdz_raman(z, B, D, w, gamma, gamma1, RW):
    """
    Compute the derivative of B with respect to z (dB/dz) in the frequency domain,
    considering the Raman effect.

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

def Raman(T, tau1=12.2e-3, tau2=32e-3, fR=0.18):
    """
    Compute the Raman response in the frequency domain.

    Parameters:
        T (ndarray): Time array.
        tau1 (float): Raman parameter tau1 (default: 12.2 fs).
        tau2 (float): Raman parameter tau2 (default: 32 fs).
        fR (float): Fractional Raman contribution (default: 0.18).

    Returns:
        ndarray: Raman response in the frequency domain.
    """
    hR = np.zeros(len(T))
    hR[T >= 0] = (tau1**2 + tau2**2) / (tau1 * tau2**2) * np.exp(-T[T >= 0] / tau2) * np.sin(T[T >= 0] / tau1)
    hR[T < 0] = 0

    hR = fftshift(hR)  # Shift to start at the beginning of the temporal window
    hR = hR / np.sum(hR)  # Normalize such that int(hR) = 1
    hR_W = FT(hR)  # Convert hR to the frequency domain

    R_W = fR * hR_W + (1 - fR) * np.ones(len(T))
    return R_W

def SolveNLS_cu(sim: Sim, fib: Fibra, pulso_0, raman=False, z_locs=None, pbar=True, cupy=False):
    """
    Simulate the evolution of a pulse using the Nonlinear Schrödinger Equation (NLSE).

    Parameters:
        sim (Sim): Simulation parameters.
        fib (Fibra): Fiber parameters.
        pulso_0 (ndarray): Input pulse in the time domain.
        raman (bool, optional): If True, use the full Raman response. Default is False.
        z_locs (int, optional): Number of z points (between 0 and L) where the solution is required. Default is None.
        pbar (bool, optional): If True, display a progress bar for the simulation. Default is True.
        cupy (bool, optional): If True, use CuPy for the nonlinear operator. Default is False.

    Returns:
        tuple:
            - zlocs (ndarray): Points where the solution is calculated.
            - A_w (ndarray): Matrix with the solution in the frequency domain [position, frequency].
            - A_t (ndarray): Matrix with the solution in the time domain [position, time], can be obtained via IFT(A_w).
    """
    espectro_0 = FT(pulso_0)
    D_w = disp_op(sim, fib)

    if raman:
        RW = Raman(sim.tiempo, fR=fib.fR)
        f_B = partial(dBdz_raman, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, gamma1=fib.gamma1, RW=RW)
    else:
        f_B = partial(dBdz, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, TR=fib.TR)

    rtol = 1e-3
    atol = 1e-6

    if pbar:
        with tqdm(total=fib.L, unit="m") as pbar:
            if raman:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    if cupy:
                        return dBdz_raman_cupy(z, B, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, gamma1=fib.gamma1, RW=RW)
                    else:
                        return dBdz_raman(z, B, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, gamma1=fib.gamma1, RW=RW)
                dBdz_with_progress.prev_z = 0
            else:
                def dBdz_with_progress(z, B):
                    pbar.update(abs(z - dBdz_with_progress.prev_z))
                    dBdz_with_progress.prev_z = z
                    return dBdz(z, B, D=D_w, w=2 * np.pi * sim.freq, gamma=fib.gamma, TR=fib.TR)
                dBdz_with_progress.prev_z = 0

            if z_locs:
                t_eval = np.linspace(0, fib.L, z_locs)
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
            else:
                sol = solve_ivp(dBdz_with_progress, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)
    else:
        if z_locs:
            t_eval = np.linspace(0, fib.L, z_locs)
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol, t_eval=t_eval)
        else:
            sol = solve_ivp(f_B, [0, fib.L], y0=espectro_0, rtol=rtol, atol=atol)

    zlocs = sol["t"]
    ysol = sol["y"]
    print(sol["message"])

    ysol = np.array(ysol)
    ysol_transposed = ysol.T
    A_w = ysol_transposed
    for j in range(len(zlocs)):
        A_w[j, :] = A_w[j, :] * np.exp(D_w * zlocs[j])
    A_t = np.array([IFT(a_w) for a_w in A_w], dtype=complex)

    return zlocs, A_w