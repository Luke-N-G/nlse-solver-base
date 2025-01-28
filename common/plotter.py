# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:51:17 2023

@author: d/dt Lucas
"""

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import numpy as np
from ..common.commonfunc import Pot, fftshift, Sim, Fibra, num_fotones, Adapt_Vector
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline


''' PLOTINST
plotinst: Función que grafica el pulso y espectro inicial y final
sim:        Parámetros de la simulación
AT, AW:     Los resultados de SolveNLS
Tlim, Wlim: Región donde graficar en tiempo y frecuencia (opcional). Unidades: [T] = ps, [W] = THz
save:       Guardar imagen, el dpi se cambia manualmente (opcional)
'''

def plotinst(sim:Sim, fib:Fibra, AT, AW, Tlim=None, Wlim=None, Ylim=None, wavelength=False, zeros=None, save=None, dB=None, noshow=None, end=-1):
    #Ploteamos el pulso y espectro inicial y final:
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
    #----pulso----
    ax1.plot( sim.tiempo, Pot(AT[0]), label="Inicial" )
    end = int(end)
    ax1.plot( sim.tiempo, Pot(AT[end]), label="Final" )
    if Tlim:
        ax1.set_xlim(Tlim)
    ax1.set_title("Tiempo")
    ax1.set_xlabel("Tiempo (ps)")
    ax1.set_ylabel("Potencia (W)")
    ax1.legend(loc="best")
    #----espectro---- Cuidado:
    if wavelength:
        lambda_vec, Alam = Adapt_Vector(sim.freq, fib.omega0, AW)
        x = lambda_vec
        y_1 = Pot(Alam[0])
        y_2 = Pot(Alam[end])
    else:
        ax2.set_title("Espectro")
        ax2.set_xlabel("Frecuencia (THz)")
        x = fftshift( sim.freq  )
        y_1 = Pot( fftshift(AW[0])  )
        y_2 = Pot( fftshift(AW[end]) )
    ax2.plot( x, y_1, label="Inicial")
    ax2.plot( x, y_2 , label="Final")
    if Wlim:
        ax2.set_xlim(Wlim)
    if Ylim:
        ax1.set_ylim(Ylim)
        ax2.set_ylim(Ylim)
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
    
    
''' PLOTCMAP
plotcmap: Grafica mapas de colores con la evolución del pulso y del espectro
sim:            Parámetros de la simulación
zlocs, AT, AW:  Resultados de SolveGNLSE/Solvepcnlse/SolvepcGNLSE
wavelength:     Boolean, si "True" grafica el espectro en longitud de onda                  (opcional)
dB:             Boolean, si "True" grafica la potencia en dB                                (opcional)
Tlim, Wlim:     Región donde graficar en tiempo y frecuencia. Unidades: [T] = ps, [W] = THz (opcional)
vlims:          Vector de 4 elementos [vmin_t,vmax_t,vmin_w,vmax_w], limites del cmap       (opcional)
cmap:           Colormap de matplotlib, default "turbo"                                     (opcional)
zeros:          Boolean, si "True" muestra los ceros en el espectro
noshow:         Boolean, si "True" no muestra el gráfico                                    (opcional)
save:           Guardar imagen, el dpi se cambia manualmente                                (opcional)'''
    
#--Función requerida por "plotcmap", para cambiar los ticks a formato 10^x en escala dB--      
def format_func(value, tick_number):
    return f'$10^{{{int(value/10)}}}$' 
#----------------------------------------------------------------------------------------

def plotcmap(sim:Sim, fib:Fibra, zlocs, AT, AW, wavelength=False, dB=False,
             vlims=[], cmap="turbo", Tlim=None, Wlim=None,
             zeros=False, save=None, noshow=False, plot_type="both"):

    #Labels y tamaños
    cbar_tick_size = 7
    tick_size      = 10
    m_label_size   = 12
    M_label_size   = 15
    
    #Vectores de potencia, y listas "extent" para pasarle a imshow
    P_T = Pot(AT)
    textent = [sim.tiempo[0], sim.tiempo[-1], zlocs[0], zlocs[-1]]
    
    #Aplicamos el fftshift al espectro, o transformamos a longitud de onda de requerirlo
    if wavelength:
        #Se tiene el problema de que el vector lambda/AL no son lineales con la freq
        #Conversión freq -> lambda, corrigiendo para que sea lineal        
        lamvec, AL = Adapt_Vector(sim.freq, fib.omega0, AW)
        # Armamos un vector de lambdas lineal (lamvec es no lineal, ya que viene de la freq.)
        lamvec_lin = np.linspace(lamvec.min(), lamvec.max(), len(lamvec))
        # Interpolamos los datos a nuestra nueva "grilla" lineal
        AW = np.empty_like(AL)
        for i in range(AL.shape[0]):
            interp_func = interp1d(lamvec, AL[i, :], kind='next')
            AW[i, :] = interp_func(lamvec_lin)
        P_W = Pot(AW)
        wextent = [lamvec_lin[0], lamvec_lin[-1], zlocs[0], zlocs[-1]]
    else:
        AW = fftshift(AW, axes=1)
        P_W = Pot(AW)
        wextent = [fftshift(sim.freq)[0], fftshift(sim.freq)[-1], zlocs[0], zlocs[-1]]
    
    #Escala dBI
    if dB:
        P_T = 10*np.log10(P_T) - np.max( 10*np.log10(P_T[0]) )
        P_W = 10*np.log10(P_W) - np.max( 10*np.log10(P_W[0]) )
        
    #Limites del colorbar
    if vlims:
        vmin_t = vlims[0]
        vmax_t = vlims[1]
        vmin_w = vlims[2]
        vmax_w = vlims[3]
    else:
        vmin_t = vmax_t = vmin_w = vmax_w = None
    #Ploteamos
    if plot_type == 'both':
        fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))
    elif plot_type == 'time':
        fig, ax1 = plt.subplots(1,1,figsize=(6.5,5))
    elif plot_type == 'freq':
        fig, ax2 = plt.subplots(1,1,figsize=(6.5,5))
    else:
        raise ValueError(f"Invalid plot type: {plot_type}. Valid types are: 'time', 'freq', 'both'.")

    #fig, (ax1,ax2) = plt.subplots(1,2,sharey=True,figsize=(8.76,5))
    
    #---Plot en tiempo---
    if plot_type in ['time', 'both']:
        #Imshow 1
        im1 = ax1.imshow(P_T, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                         extent=textent, vmin=vmin_t, vmax=vmax_t)
        ax1.tick_params(labelsize=tick_size)
        ax1.set_ylabel("Distance (m)", size=m_label_size)
        ax1.set_xlabel("Time (ps)", size=m_label_size)
        ax1.set_title("Pulse", size=M_label_size)
        #Colorbar 1: Es interactivo
        cbar1 = fig.colorbar(im1, ax=ax1, label='Normalized power (dB)' if dB else 'Power (W)', location="bottom", aspect=50 )
        cbar1.ax.tick_params(labelsize=cbar_tick_size)
        #if dB: #Si dB == True, ajustamos los ticks al formato 10^x (interactuar con esto lo rompe)
        #    cbar1.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        if Tlim:
            ax1.set_xlim(Tlim)        
        
    #---Plot en espectro---
    if plot_type in ['freq', 'both']:
        #Imshow 2
        im2 = ax2.imshow(P_W, cmap=cmap, aspect="auto", interpolation='bilinear', origin="lower",
                         extent=wextent, vmin=vmin_w, vmax=vmax_w)
        ax2.tick_params(labelsize=tick_size)
        ax2.set_title("Spectrum", size=M_label_size)
        #Colorbar 2
        cbar2 = fig.colorbar(im2, ax=ax2, label='PSD (a.u. dB)' if dB else "PSD (a.u.)", location="bottom", aspect=50 )
        cbar2.ax.tick_params(labelsize=cbar_tick_size)
        #if dB:
        #    cbar2.ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
            
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
    
    
    
'''  PLOTSPECGRAM
plotspecgram: Grafica el espectrograma de un vector
sim, fib:      Parámetros de la simulación y la fibra, necesarios para calcular bien el sampling rate
AT:            Pulso, vector 1D (*No* es el resultado de SolveNLS, sino AT[i])
Tlim, Wlim:    Región donde graficar en tiempo y frecuencia        (opcional). Unidades: [T] = ps, [W] = THz
save:          Guardar imagen, el dpi se cambia manualmente        (opcional)
dB:            Boolean, si "True" grafica en escala dB             (opcional)
zeros:         Boolean, si "True" grafica la posición de ZDW y ZNW (opcional)
cmap:          String, se elije el colormap. Por default es Turbo  (opcional)
nowshow:       Boolean, si "True" no muestra el gráfico            (opcional)
dpi:           Int, ajusta la resolución de la imagen final        (opcional)
'''
def plotspecgram(sim:Sim, fib:Fibra, AT, Tlim=None, Wlim=None, end=-1, save=None, dB=None, zeros=None, cmap="turbo", noshow=None, dpi=800):
    
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
    
    #Escala del gráfico
    if dB:
        scale="dB"
    else:
        scale="linear"

    plt.specgram(AT,NFFT=700,noverlap=650,Fs=t_sampling,scale=scale,xextent=xextent,cmap=cmap)

    if zeros:
        if fib.w_zdw:
            freq_zdw = (fib.omega0 - fib.w_zdw)/(2*np.pi)
            plt.plot(xextent, [freq_zdw,freq_zdw], "--", linewidth=2, label="ZDW = "+str(round(fib.zdw))+" nm" )
        if fib.w_znw:
            freq_znw = (fib.omega0 - fib.w_znw)/(2*np.pi)
            plt.plot(xextent, [freq_znw,freq_znw], "--", linewidth=2, label="ZNW = "+str(round(fib.znw))+" nm" )
        plt.legend(loc="best")
    plt.xlabel("Time (ps)")
    plt.ylabel("Frequency (THz)")
    plt.tight_layout()
    if Wlim:
        plt.xlim(Tlim)
    if Tlim:
        plt.ylim(Wlim)
    if save:
        plt.savefig(save, dpi=dpi)
    if noshow:
        plt.close()
    else:
        plt.show()
        
        
def plotspecgram2(sim:Sim, fib:Fibra, AT, Tlim=None, Wlim=None, end=-1, save=None, dB=None, zeros=None, cmap="turbo", noshow=None, dpi=800):
    
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
    
    # Escala del gráfico
    if dB:
        scale = "dB"
    else:
        scale = "linear"


    # Compute the spectrogram
    Pxx, freqs, bins, im = ax.specgram(AT, NFFT=700, noverlap=650, Fs=t_sampling, scale=scale, xextent=xextent, cmap=cmap)


    # Add interactive colorbar
    cbar = fig.colorbar(im, ax=ax, label='Power/Frequency (dB/Hz)' if dB else 'Power/Frequency', location="bottom", aspect=50)
    cbar.ax.tick_params(labelsize=7)

    
    if zeros:
        if fib.w_zdw:
            freq_zdw = (fib.omega0 - fib.w_zdw) / (2 * np.pi)
            ax.plot(xextent, [freq_zdw, freq_zdw], "--",color="blue", linewidth=2, label="ZDW = " + str(round(fib.zdw)) + " nm")
        if fib.w_znw:
            freq_znw = (fib.omega0 - fib.w_znw) / (2 * np.pi)
            ax.plot(xextent, [freq_znw, freq_znw], "--", color="red", linewidth=2, label="ZNW = " + str(round(fib.znw)) + " nm")
        ax.legend(loc="best")

    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Frequency (THz)")
    ax.tick_params(labelsize=10)
    plt.tight_layout()

    print(np.max(Pxx))

    if Wlim:
        ax.set_xlim(Tlim)
    if Tlim:
        ax.set_ylim(Wlim)
    if save:
        plt.savefig(save, dpi=dpi)
    if noshow:
        plt.close()
    else:
        plt.show()
    
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
    
    c = 299792458 # Velocidad de la luz en m/s
    omega0 = 2 * np.pi * c / (lambda0*1e-9) #Definimos la frecuencia central, pasando lamba0 a m    

    fotones = np.zeros( len(zlocs) )
    for i in range(len(fotones)):
        fotones[i] = num_fotones(sim.freq, AW[i], w0=omega0)
    plt.plot(zlocs, fotones)
    plt.title("Fotones")
    plt.ylabel("N")
    plt.xlabel("z")
    plt.show()
    
def plotchirp(t, pulso): #Pendiente
    return None





