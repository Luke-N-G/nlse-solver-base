# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 10:37:39 2023

@author: d/dt Lucas
"""

import numpy as np
import re
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
import pickle
import datetime


# FFT siguiendo la convención -iw (Agrawal)
def FT(pulso):
    if pulso.ndim == 1:
        return ifft(pulso) * len(pulso)
    elif pulso.ndim == 2:
        return ifft(pulso, axis=1) * pulso.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

# IFFT siguiendo la convención -iw
def IFT(espectro):
    if espectro.ndim == 1:
        return fft(espectro) / len(espectro)
    elif espectro.ndim == 2:
        return fft(espectro, axis=1) / espectro.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

#Pasamos de array tiempo a array frecuencia
def t_a_freq(t_o_freq):
    return fftfreq( len(t_o_freq) , d = t_o_freq[1] - t_o_freq[0])


def Adapt_Vector(freq, omega0, Aw):
    lambda_vec = 299792458 * (1e9)/(1e12) / (freq + omega0/(2*np.pi))
    sort_indices = lambda_vec.argsort()
    lambda_vec_ordered = lambda_vec[sort_indices]
    Alam = Aw[:, sort_indices]
    return lambda_vec_ordered, Alam




#%%-------------CLASES---------------------

class Pulses:
    def __init__(self):
        pass
    #Super gaussiana
    def Sgaussian(self, t, amplitude, width, offset, chirp, order):
        return np.sqrt(amplitude)*np.exp(- (1+1j*chirp)/2*((t-offset)/(width))**(2*np.floor(order)))*(1+0j)
    #Solitones
    def soliton(self, t, width, beta2, gamma, order):
        return order * np.sqrt( np.abs(beta2)/(gamma * width**2) ) * (1/np.cosh(t/width))
    #Para colisión de pulsos
    def twopulse(self, t, amp1, amp2, width1, width2, offset1, offset2, dfreq):
        np.seterr(over='ignore') #Silenciamos avisos de overflow (1/inf == 0 para estos casos)
        pump   = np.sqrt(amp1)*(1/np.cosh(t/width1))
        signal = np.sqrt(amp2)*(1/np.cosh((t + offset2)/width2))*np.exp(-2j*np.pi*dfreq*t)
        np.seterr(over='warn')   #Reactivamos avisos de overflow
        return pump + signal
    #Orden N de un pulso
    def order(beta2, gamma, width, power):
        return np.sqrt( gamma * power * width**2 / np.abs(beta2)  )

class Tools:
    def __init__(self):
        pass
    
    def pot(self, signal):
        return np.abs(signal)**2
    
    def energy(self, t_or_freq, signal):
        return np.trapz( self.pot(signal), t_or_freq )
    
    def photons(self, freq, spectrum, omega0):
        return np.sum( np.abs(spectrum)**2 / (freq + omega0) )
    
    def find_chirp(t, signal):
        phase = np.unwrap( np.angle(signal) ) #Angle busca la fase, Unwrap la extiende de [0,2pi) a todos los reales.
        #fase = fase - fase[ int(len(fase)/2)  ] #Centramos el array
        df = np.diff( phase, prepend = phase[0] - (phase[1]  - phase[0]  ), axis=0)
        dt = np.diff( t, prepend = t[0]- (t[1] - t[0] ), axis=0 )
        chirp = -df/dt
        return chirp
    
    def find_shift(self, zlocs, freq, A_wr):
        peaks = np.zeros( len(zlocs), dtype=int )
        dw    = np.copy(peaks)
        for index_z in range( len(zlocs) ):
            peaks[index_z] = np.argmax( self.pot( fftshift(A_wr[index_z]) ) )
            #dw[index_z] = fftshift(2*np.pi*sim_r.freq)[peaks[index_z]]
        dw = fftshift(2*np.pi*freq)[peaks]
        return dw

    def find_k(self, Aw, dz, rel_thr=1e-6):
        Aw = Aw.T  
        ks    = np.zeros_like(Aw, dtype='float64')
        phis  = np.zeros_like(ks,  dtype='float64')
        mask = abs(Aw)>rel_thr*np.max(np.abs(Aw))
        phis[mask] = np.unwrap( np.angle(Aw[mask]), axis=0 )
        ks = np.diff(phis)/dz
        ks[np.diff(mask) != 0] = 0
        return ks, phis


class Units:
    def __init__(self):
        
        self.constants = { "c": 299792458, "h": 6.62607015e-34, "hb": 6.62607015e-34/(2*np.pi) }
        
        self.metric_prefixes = {
            'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3,
            'h': 1e2, 'da': 1e1, '': 1, 'o':1, 'd': 1e-1, 'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9,
            'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
        }

    def parse_unit(self, unit):
        # Regular expression to match the unit and its optional exponent
        pattern = re.compile(r'([a-zA-Zµ]+)(?:\^(-?\d+))?')
        matches = pattern.findall(unit)
        parsed = {}
        for prefix, exp in matches:
            if prefix in parsed:
                parsed[prefix] += int(exp) if exp else 1
            else:
                parsed[prefix] = int(exp) if exp else 1
        return parsed

    def convert_unit(self, original_unit, target_unit):
        original_parsed = self.parse_unit(original_unit)
        target_parsed = self.parse_unit(target_unit)

        conversion_factor = 1.0

        for prefix, exp in original_parsed.items():
            if prefix in target_parsed:
                target_exp = target_parsed[prefix]
                conversion_factor *= (self.metric_prefixes[prefix] ** (exp - target_exp))
            else:
                conversion_factor *= (self.metric_prefixes[prefix] ** exp)

        for prefix, exp in target_parsed.items():
            if prefix not in original_parsed:
                conversion_factor /= (self.metric_prefixes[prefix] ** exp)

        return conversion_factor


#Sim: Guarda los parámetros de simulación
class Sim:
    def __init__(self, puntos, Tmax):
        self.puntos = puntos #Num. de puntos de tiempo
        self.Tmax   = Tmax   #Span de tiempo [-Tmax,Tmax]
        self.paso_t = 2.0*Tmax/puntos #Step temporal
        self.tiempo = np.arange(-puntos/2,puntos/2)*self.paso_t #Vector de tiempos
        self.dW     = np.pi/Tmax #Step en frecuencia
        self.freq   = fftshift( np.pi * np.arange(-puntos/2,puntos/2) / Tmax )/(2*np.pi) #Vector frecuencia (1/s)
        
    @property #Frecuencia shifteada
    def sfreq(self):
        return fftshift(self.freq)
    @property #Frecuencia angular
    def omega(self):
        return 2*np.pi*self.freq
    @property #Frecuencia angular shifteada
    def somega(self):
        return fftshift( 2*np.pi*self.freq )
    @property #Vector longitud de onda
    def lam(self, omega0):
        lam_vec, _ = Adapt_Vector(self.freq, omega0, np.zeros([1, self.freq.size]) )
        return lam_vec
        
#Fibra: Guarda los parámetros de la fibra, más algunos métodos útiles.
class Fibra:
    def __init__(self, L, gamma, gamma1, alpha, lambda0, TR=3e-3, fR=0.18, beta1=0, beta2=0, beta3=0, betas=None):
        self.L = L  # Longitud de la fibra
        self.gamma = gamma  # gamma de la fibra, para calcular SPM
        self.gamma1 = gamma1  # gamma1 de la fibra, para self-steepening
        self.alpha = alpha  # alpha de la fibra, atenuación
        self.TR = TR  # TR de la fibra, para calcular Raman
        self.fR = fR  # fR de la fibra, para calcular Raman (y self-steepening)
        self.lambda0 = lambda0  # Longitud de onda central
        self.omega0 = 2 * np.pi * 299792458 * (1e9) / (1e12) / lambda0  # Frecuencia (angular) central
        self.beta1 = beta1 #Parámetro de velocidad de grupo, opcional

        if betas is not None:
            self.betas = betas
            self.beta2 = betas[0]
            self.beta3 = betas[1] if len(betas) > 1 else 0
        else:
            self.betas = [beta2, beta3] if beta2 is not None and beta3 is not None else [0, 0]
            self.beta2 = beta2
            self.beta3 = beta3

        if self.beta3 != 0:
            self.w_zdw = -self.beta2 / self.beta3 + self.omega0
            self.zdw = 2 * np.pi * 299792458 * (1e9) / (1e12) / self.w_zdw
        else:
            self.w_zdw = None
            self.zdw = None

        if self.gamma1 != 0:
            self.w_znw = -self.gamma / self.gamma1 + self.omega0
            self.znw = 2 * np.pi * 299792458 * (1e9) / (1e12) / self.w_znw
        else:
            self.w_znw = None
            self.znw = None

    @property
    def beta2(self):
        return self.betas[0]

    @beta2.setter
    def beta2(self, value):
        if not self.betas:
            self.betas = [0] * 2
        self.betas[0] = value

    @property
    def beta3(self):
        return self.betas[1] if len(self.betas) > 1 else 0

    @beta3.setter
    def beta3(self, value):
        if len(self.betas) > 1:
            self.betas[1] = value
        else:
            self.betas.append(value)

    # Método para pasar de omega a lambda
    def omega_to_lambda(self, w):
        return 2 * np.pi * 299792458 * (1e9) / (1e12) / (self.omega0 + w)

    # Método para pasar de lambda a Omega
    def lambda_to_omega(self, lam):
        return 2 * np.pi * 299792458 * (1e9) / (1e12) * (1 / lam - 1 / self.lambda0)

    # Método para calcular gamma en función de omega
    def gamma_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        return self.gamma + self.gamma1 * w
    
    # Método para calcular beta1 en función de omega
    def beta1_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        if self.betas != 0:
            beta1 = 0
            for i, beta in enumerate(self.betas):
                beta1 += beta * w ** (i+1) / np.math.factorial(i+1)
        else:
            beta1 = self.beta2 * w + self.beta3 * w**2
        return beta1

    # Método para calcular beta2 en función de omega
    def beta2_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        if self.betas != 0:
            beta2 = 0
            for i, beta in enumerate(self.betas):
                beta2 += beta * w ** i / np.math.factorial(i)
        else:
            beta2 = self.beta2 + self.beta3 * w
        return beta2

    def calculate_zdw(self):
        # Solve the polynomial equation for omega
        coefficients = self.betas[::-1]
        omega_solutions = np.roots(coefficients)

        # Convert to absolute frequency
        w_zdw = omega_solutions + self.omega0

        # Calculate ZDW
        ZDW = 2 * np.pi * 299792458 * (1e9) / (1e12) / w_zdw
        return ZDW

#Data: Guarda los resultados de la simulación
class Data:
    def __init__(self, solve_function, sim, fibra, pulse,  **kwargs):
        self.z, self.W = solve_function(sim, fibra, pulse, **kwargs)
        self.fib = fibra
        self.sim = sim
    
    #Definimos la propiedad A.T = Evolución en tiempo
    @property
    def T(self):
        return IFT(self.W)
    
    #Definimos la propiedad A.Ws = Evolución espectral shifteada
    @property
    def Ws(self):
        return fftshift( self.W )
    
    #Definimos función de guardado
    def save(self, filename, other_par=None):
        fib = getattr(self, "fib", None)
        sim = getattr(self, "sim", None)
        saver(self, filename, other_par)

#Additional
def disp_op(sim:Sim, fib:Fibra):
           
    #Si solo damos valores de beta2 y/o beta3
    D_W = 1j * fib.beta2/2 * sim.omega**2 + 1j * fib.beta3/6 * sim.omega**3 - fib.alpha/2
    
    #Si pasamos un vector de betas, reescribimos el operador a la versión más general
    if fib.betas:
        D_w = 0
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/np.math.factorial(i+2) * sim.omega**(i+2)
            
    if fib.beta1:
        D_w = D_w + 1j*fib.beta1*sim.omega
        
    return D_w

#%% Funciones para guardar y cargar datos

# Guardando         
def saver(data:Data, filename, other_par = None):
    # Guardando los parametros de simulación y de la fibra en un diccionario.
    metadata = {'Sim': data.sim.__dict__, 'Fibra': data.fib.__dict__} #sim.__dict__ = {'puntos'=N, 'Tmax'=70, ...}

    # Guardando los datos en filename-data.txt con pickle para cargar después.
    with open(f"{filename}-data.txt", 'wb') as f:
        pickle.dump(data, f)
        
    # Guardar parametros filename-param.txt para leer directamente.
    with open(f"{filename}-param.txt", 'w') as f:
        f.write('-------------Parameters-------------\n')
        f.write(f'{datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")}\n\n')
        for class_name, class_attrs in metadata.items():
            f.write(f'\n-{class_name}:\n\n')
            for attr, value in class_attrs.items():
                f.write(f'{attr} = {value}\n')
        if other_par:
            f.write("\n\n-Other Parameters:\n\n")
            if isinstance(other_par, str):
                f.write(f'{other_par}\n')
            else:
                for i in other_par:
                    f.write(f'{str(i)}\n')

# Cargando datos    
def loader(filename):
    with open(f"{filename}-data.txt", 'rb') as f:
        data = pickle.load(f)
    return data


