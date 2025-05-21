# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:51:17 2023

@author: d/dt Lucas
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline
try:
    from ..common.commonfunc import Tools, fftshift, Sim, Fibra, Data, Adapt_Vector
except ImportError:
    from common.commonfunc import Tools, fftshift, Sim, Fibra, Data, Adapt_Vector

"""
Function to plot the initial and final pulse and spectrum from simulation data.
"""

def plotinst(data:Data, Tlim=None, Wlim=None, Plim=None, PSDlim=None,
             wavelength=False, zeros=None, save=None, dB=None, noshow=None, end=-1):
    
    """
    Plot the initial and final pulse and spectrum from simulation data.
    
    Parameters:
        data (Data): Simulation data object containing temporal and spectral evolution.
        Tlim (tuple, optional): Time axis limits for the plot (min, max).
        Wlim (tuple, optional): Frequency or wavelength axis limits for the plot (min, max).
        Plim (tuple, optional): Power axis limits for the time-domain plot (min, max).
        PSDlim (tuple, optional): Power spectral density axis limits for the spectrum plot (min, max).
        wavelength (bool, optional): If True, plot spectrum in wavelength domain. Default is False.
        zeros (bool, optional): If True, mark zero-dispersion and zero-nonlinearity points. Default is None.
        save (str, optional): File path to save the plot. Default is None.
        dB (bool, optional): If True, use logarithmic scale for power and spectrum. Default is None.
        noshow (bool, optional): If True, do not display the plot. Default is None.
        end (int, optional): Index of the final propagation step to plot. Default is -1 (last step).
    """
    
    #Include instance of tools
    tls = Tools()
    
    #Set the arrays for plotting
    AT  = data.T
    AW  = data.W
    sim = data.sim
    fib = data.fib

    #Plot the initial pulse and spectrum
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    #----Pulse----
    ax1.plot( sim.tiempo, tls.pot(AT[0]), label="Initial" )
    end = int(end)
    ax1.plot( sim.tiempo, tls.pot(AT[end]), label="Final" )
    if Tlim:
        ax1.set_xlim(Tlim)
    ax1.set_title("Time")
    ax1.set_xlabel("Tine (ps)")
    ax1.set_ylabel("Power (W)")
    ax1.legend(loc="best")
    #----Spectrum---- Cuidado:
    if wavelength:
        lambda_vec, Alam = Adapt_Vector(sim.freq, fib.omega0, AW)
        x = lambda_vec
        y_1 = tls.pot(Alam[0])
        y_2 = tls.pot(Alam[end])
    else:
        ax2.set_title("Spectrum")
        ax2.set_xlabel("Frequency (THz)")
        x = fftshift( sim.freq  )
        y_1 = tls.pot( fftshift(AW[0])  )
        y_2 = tls.pot( fftshift(AW[end]) )
    ax2.plot( x, y_1, label="Initial")
    ax2.plot( x, y_2 , label="Final")
    if Wlim:
        ax2.set_xlim(Wlim)
    if Plim:
        ax1.set_ylim(Plim)
    if PSDlim:
        ax2.set_ylim(PSDlim)
    if zeros:
        freq_zdw = (fib.omega0 - fib.w_zdw)/(2*np.pi)
        freq_znw = (fib.omega0 - fib.w_znw)/(2*np.pi)
        if wavelength:
            ax2.axvline(x = fib.zdw, linestyle=":", color="gray",  label="ZDW")
            ax2.axvline(x = fib.znw, linestyle="--", color="gray", label="ZNW")
        else:
            ax2.axvline(x = freq_zdw, linestyle="--", label="ZDW")
            ax2.axvline(x = freq_znw, linestyle="--", label="ZNW")
    ax2.legend(loc="best")
    if dB:
        ax1.set_yscale("log")
        ax2.set_yscale("log")
    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=800)        
    if noshow:
        plt.close()
    else:
        plt.show()
    
    
    
"""
Function to plot colormaps of the temporal and spectral evolution from simulation data.
"""

def plotcmap(data:Data, wavelength=False, dB=False,
             vlims=[], cmap="turbo", Tlim=None, Wlim=None,
             zeros=False, save=None, noshow=False, plot_type="both"):
    """
    Plot colormaps of the temporal and spectral evolution from simulation data.
    
    Parameters:
        data (Data): Simulation data object containing temporal and spectral evolution.
        wavelength (bool, optional): If True, plot spectrum in wavelength domain. Default is False.
        dB (bool, optional): If True, use logarithmic scale for power and spectrum. Default is False.
        vlims (list, optional): Colorbar limits [vmin_t, vmax_t, vmin_w, vmax_w]. Default is [].
        cmap (str, optional): Colormap to use for the plots. Default is "turbo".
        Tlim (tuple, optional): Time axis limits for the plot (min, max). Default is None.
        Wlim (tuple, optional): Frequency or wavelength axis limits for the plot (min, max). Default is None.
        zeros (bool, optional): If True, mark zero-dispersion and zero-nonlinearity points. Default is False.
        save (str, optional): File path to save the plot. Default is None.
        noshow (bool, optional): If True, do not display the plot. Default is False.
        plot_type (str, optional): Type of plot to generate: 'time', 'freq', or 'both'. Default is "both".
    """
    
    #Include instance of tools
    tls = Tools()
    
    #Set arrays for plotting
    zlocs = data.z
    AT    = data.T
    AW    = data.W
    sim   = data.sim
    fib   = data.fib
    
    #Label sizes
    cbar_tick_size = 7
    tick_size      = 10
    m_label_size   = 12
    M_label_size   = 15
    
    #Power arrays and "extent" lists to pass to imshow
    P_T = tls.pot(AT)
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]
    
    #Apply fftshift to spectrum, or transform to wavelength if required
    if wavelength:
        #The lambda (or AL) array is not lienear with frequency
        #We convert frequency to wavelength, correcting for the nonlinear relation
        lamvec, AL = Adapt_Vector(sim.freq, fib.omega0, AW)
        #Create a linear wavelength vector (lamvec is not linear, as it comes from the frequency array)
        lamvec_lin = np.linspace(lamvec.min(), lamvec.max(), len(lamvec))
        #Interpolate the data to our new linear "grid"
        AW = np.empty_like(AL)
        for i in range(AL.shape[0]):
            interp_func = interp1d(lamvec, AL[i, :], kind='next')
            AW[i, :] = interp_func(lamvec_lin)
        P_W = tls.pot(AW)
        wextent = [lamvec_lin[0], lamvec_lin[-1], zlocs[0], zlocs[-1]]
    else:
        AW = fftshift(AW, axes=1)
        P_W = tls.pot(AW)
        wextent = [fftshift(sim.freq)[0], fftshift(sim.freq)[-1], zlocs[0], zlocs[-1]]
    
    #dBI scale
    if dB:
        P_T = 10*np.log10(P_T) - np.max( 10*np.log10(P_T[0]) )
        P_W = 10*np.log10(P_W) - np.max( 10*np.log10(P_W[0]) )
        
    #Colorbar limits
    if vlims:
        vmin_t = vlims[0]
        vmax_t = vlims[1]
        vmin_w = vlims[2]
        vmax_w = vlims[3]
    else:
        vmin_t = vmax_t = vmin_w = vmax_w = None
    #Plots
    if plot_type == 'both':
        fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))
    elif plot_type == 'time':
        fig, ax1 = plt.subplots(1,1,figsize=(6.5,5))
    elif plot_type == 'freq':
        fig, ax2 = plt.subplots(1,1,figsize=(6.5,5))
    else:
        raise ValueError(f"Invalid plot type: {plot_type}. Valid types are: 'time', 'freq', 'both'.")

    #fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))
    
    #---Time plot---
    if plot_type in ['time', 'both']:
        #Imshow 1
        im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                         extent=textent, vmin=vmin_t, vmax=vmax_t)
        ax1.tick_params(labelsize=tick_size)
        ax1.set_ylabel("Distance (m)", size=m_label_size)
        ax1.set_xlabel("Time (ps)", size=m_label_size)
        ax1.set_title("Pulse", size=M_label_size)
        #Colorbar 1: Interactive!
        cbar1 = fig.colorbar(im1, ax=ax1, label='Normalized power (dB)' if dB else 'Power (W)', location="bottom", aspect=50 )
        cbar1.ax.tick_params(labelsize=cbar_tick_size)
        if Tlim:
            ax1.set_xlim(Tlim)        
        
    #---Spectrum plot---
    if plot_type in ['freq', 'both']:
        #Imshow 2
        im2 = ax2.imshow(P_W, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                         extent=wextent, vmin=vmin_w, vmax=vmax_w)
        ax2.tick_params(labelsize=tick_size)
        ax2.set_title("Spectrum", size=M_label_size)
        #Colorbar 2: Also interactive!
        cbar2 = fig.colorbar(im2, ax=ax2, label='PSD (a.u. dB)' if dB else "PSD (a.u.)", location="bottom", aspect=50 )
        cbar2.ax.tick_params(labelsize=cbar_tick_size)
            
        if zeros:
            freq_zdw = (fib.omega0 - fib.w_zdw)/(2*np.pi)
            freq_znw = (fib.omega0 - fib.w_znw)/(2*np.pi)
            if wavelength:
                ax2.axvline(x = fib.zdw, linestyle="--", color="white",  label="ZDW")
                ax2.axvline(x = fib.znw, linestyle="--", color="crimson", label="ZNW")
            else:
                ax2.axvline(x = freq_zdw, linestyle="--", label="ZDW")
                ax2.axvline(x = freq_znw, linestyle="--", label="ZNW")
            plt.legend(loc="best")
        if Wlim:
            ax2.set_xlim(Wlim)
        if wavelength:
            ax2.set_xlabel("Wavelength (nm)", size=m_label_size)
        else:
            ax2.set_xlabel("Frequency (THz)", size=m_label_size)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(save, dpi=800)
    if noshow:
        plt.close()
    else:
        plt.show()
    
    
    
"""
Function to plot a spectrogram of the temporal evolution of the simulation data.
"""
def plotspecgram(data:Data, Tlim=None, Wlim=None, end=-1, save=None, dB=None, zeros=None, cmap="turbo", noshow=None, dpi=800):
    """
    Plot a spectrogram of the temporal evolution of the simulation data.
    
    Parameters:
        data (Data): Simulation data object containing temporal evolution.
        Tlim (tuple, optional): Time axis limits for the plot (min, max). Default is None.
        Wlim (tuple, optional): Frequency axis limits for the plot (min, max). Default is None.
        end (int, optional): Index of the propagation step to plot. Default is -1 (last step).
        save (str, optional): File path to save the plot. Default is None.
        dB (bool, optional): If True, use logarithmic scale for the spectrogram. Default is None.
        zeros (bool, optional): If True, mark zero-dispersion and zero-nonlinearity points. Default is None.
        cmap (str, optional): Colormap to use for the spectrogram. Default is "turbo".
        noshow (bool, optional): If True, do not display the plot. Default is None.
        dpi (int, optional): Resolution of the saved plot. Default is 800.
    """
    
    #Set arrays for plotting
    AT  = data.T
    sim = data.sim
    fib = data.fib
    
    #Select the propagation step in which to plot the spectrogram
    AT = AT[end,:]
    
    plt.figure()    
    
    #x-axis limits
    xextent = [sim.tiempo[0], sim.tiempo[-1]]
    
    #Time span
    t_span = sim.tiempo[-1] - sim.tiempo[0]
    
    #Sampling rate
    t_sampling = len(sim.tiempo) / t_span    
    
    #Colormap
    cmap = cmap
    
    #Scale for the spectrogram
    if dB:
        scale="dB"
    else:
        scale="linear"

    #Plot the spectrogram
    plt.specgram(AT,NFFT=700,noverlap=650,Fs=t_sampling,scale=scale,xextent=xextent,cmap=cmap)

    #Mark zero-dispersion and zero-nonlinearity points if specified
    if zeros:
        if fib.w_zdw:
            freq_zdw = (fib.omega0 - fib.w_zdw)/(2*np.pi)
            plt.plot(xextent, [freq_zdw,freq_zdw], "--", linewidth=2, label="ZDW = "+str(round(fib.zdw))+" nm" )
        if fib.w_znw:
            freq_znw = (fib.omega0 - fib.w_znw)/(2*np.pi)
            plt.plot(xextent, [freq_znw,freq_znw], "--", linewidth=2, label="ZNW = "+str(round(fib.znw))+" nm" )
        plt.legend(loc="best")
        
    #Set axis labels
    plt.xlabel("Time (ps)")
    plt.ylabel("Frequency (THz)")
    
    plt.tight_layout()
    
    #Apply axis limits if specified
    if Wlim:
        plt.xlim(Tlim)
    if Tlim:
        plt.ylim(Wlim)
    
    #Save the plot if path is provided
    if save:
        plt.savefig(save, dpi=dpi)
        
    #Show or close the plot
    if noshow:
        plt.close()
    else:
        plt.show()
        
        
def plotspecgram2(data:Data, Tlim=None, Wlim=None, end=-1, save=None, dB=None, zeros=None, cmap="turbo", noshow=None, dpi=800):
    
    """
    Plot a spectrogram of the temporal evolution of the simulation data.
    
    Parameters:
        data (Data): Simulation data object containing temporal evolution.
        Tlim (tuple, optional): Time axis limits for the plot (min, max). Default is None.
        Wlim (tuple, optional): Frequency axis limits for the plot (min, max). Default is None.
        end (int, optional): Index of the propagation step to plot. Default is -1 (last step).
        save (str, optional): File path to save the plot. Default is None.
        dB (bool, optional): If True, use logarithmic scale for the spectrogram. Default is None.
        zeros (bool, optional): If True, mark zero-dispersion and zero-nonlinearity points. Default is None.
        cmap (str, optional): Colormap to use for the spectrogram. Default is "turbo".
        noshow (bool, optional): If True, do not display the plot. Default is None.
        dpi (int, optional): Resolution of the saved plot. Default is 800.
    """
 
    #Set arrays for plotting
    AT  = data.T
    sim = data.sim
    fib = data.fib
    
    #Select the propagation step in which to plot the spectrogram
    AT = AT[end,:]
    
    fig, ax = plt.subplots()
    
    # x-axis limits
    xextent = [sim.tiempo[0], sim.tiempo[-1]]
    
    # Time span
    t_span = sim.tiempo[-1] - sim.tiempo[0]
    
    # Sampling rate
    t_sampling = len(sim.tiempo) / t_span    
    
    # Colormap
    cmap = cmap
    
    # Scale for the spectrogram
    if dB:
        scale = "dB"
    else:
        scale = "linear"


    # Compute the spectrogram
    Pxx, freqs, bins, im = ax.specgram(AT, NFFT=700, noverlap=650, Fs=t_sampling, scale=scale, xextent=xextent, cmap=cmap)

    # Add interactive colorbar
    cbar = fig.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)' if dB else 'Power/Frequency', location="bottom", aspect=50)
    cbar.ax.tick_params(labelsize=7)

    #Mark zero-dispersion and zero-nonlinearity points if specified
    if zeros:
        if fib.w_zdw:
            freq_zdw = (fib.omega0 - fib.w_zdw) / (2 * np.pi)
            ax.plot(xextent, [freq_zdw, freq_zdw], "--",color="blue", linewidth=2, label="ZDW = " + str(round(fib.zdw)) + " nm")
        if fib.w_znw:
            freq_znw = (fib.omega0 - fib.w_znw) / (2 * np.pi)
            ax.plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2, label="ZNW = " + str(round(fib.znw)) + " nm")
        ax.legend(loc="best")

    #Set axis labels
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Frequency (THz)")
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    #Set axis limits if specified
    if Wlim:
        ax.set_xlim(Tlim)
    if Tlim:
        ax.set_ylim(Wlim)
    
    #Save the plot if a file path is provided
    if save:
        plt.savefig(save, dpi=dpi)
        
    #Show or close the plot
    if noshow:
        plt.close()
    else:
        plt.show()

#Untranslated plotter functions:
    
'''plotenergia: Grafica la energía del pulso y espectro en función de la posición'''

def plotenergia(sim:Sim, zlocs, AT, AW, save=None):
    energia_T = np.zeros( len(AT) )
    energia_F = np.zeros( len(AW) )
    for i in range(len(energia_F)):
        energia_T[i] = np.sum( np.abs(AT[i])**2 ) * sim.paso_t
        energia_F[i] = np.sum( np.abs(AW[i])**2 ) * sim.paso_t/len(AW[i])
    plt.figure()
    plt.plot(zlocs, energia_T)
    plt.plot(zlocs, energia_F)
    plt.title("Energía")
    plt.ylabel("E")
    plt.xlabel("z")
    if save:
        plt.savefig(save, dpi=800)
    plt.show()
    
def plotfotones(sim:Sim, zlocs, AW, lambda0 = 1550, save=None):
    
    tls = Tools()
    
    c = 299792458 # Velocidad de la luz en m/s
    omega0 = 2 * np.pi * c / (lambda0*1e-9) #Definimos la frecuencia central, pasando lamba0 a m    

    fotones = np.zeros( len(zlocs) )
    for i in range(len(fotones)):
        fotones[i] = tls.photons(sim.freq, AW[i], w0=omega0)
    plt.plot(zlocs, fotones)
    plt.title("Fotones")
    plt.ylabel("N")
    plt.xlabel("z")
    plt.show()
    
def plotchirp(t, pulso): #Pendiente
    return None





