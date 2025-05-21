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


"""
Utility functions for FFT, IFFT, frequency conversion, and vector adaptation.
"""

# FFT with "-iw" convention (From Agrawal: Nonlinear Fiber Optics)
def FT(pulse):
    """
    Perform the Fourier Transform (FT) with the "-iw" convention.
    
    Parameters:
        pulse (ndarray): Input array representing the time-domain signal.
                        Can be 1D or 2D.
    
    Returns:
        ndarray: Fourier-transformed signal in the frequency domain.
    
    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if pulse.ndim == 1:
        return ifft(pulse) * len(pulse)
    elif pulse.ndim == 2:
        return ifft(pulse, axis=1) * pulse.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

# IFFT with "-iw" convention
def IFT(spectrum):
    """
    Perform the Inverse Fourier Transform (IFT) with the "-iw" convention.
    
    Parameters:
        spectrum (ndarray): Input array representing the frequency-domain signal.
                            Can be 1D or 2D.
    
    Returns:
        ndarray: Inverse Fourier-transformed signal in the time domain.
    
    Raises:
        ValueError: If the input array is not 1D or 2D.
    """
    if spectrum.ndim == 1:
        return fft(spectrum) / len(spectrum)
    elif spectrum.ndim == 2:
        return fft(spectrum, axis=1) / spectrum.shape[1]
    else:
        raise ValueError("Input must be a 1D or 2D array")

#Time to frequency array
def t_a_freq(t_o_freq):
    """
    Convert a time-domain array to a frequency-domain array.
    
    Parameters:
        t_o_freq (ndarray): Time-domain array (1D).
    
    Returns:
        ndarray: Frequency-domain array (1D), with units of 1/ps = THz.
    """
    return fftfreq( len(t_o_freq) , d = t_o_freq[1] - t_o_freq[0])

#Adapt frequency array to wavelength array
def Adapt_Vector(freq, omega0, Aw):
    """
    Adapt a frequency-domain vector to a wavelength-domain vector.
    
    Parameters:
        freq (ndarray): Frequency array (1D), in THz.
        omega0 (float): Central angular frequency, in rad/ps.
        Aw (ndarray): Amplitude array (2D), with rows corresponding to frequencies.
    
    Returns:
        tuple:
            - lambda_vec_ordered (ndarray): Wavelength array (1D), in nm.
            - Alam (ndarray): Amplitude array (2D), reordered by wavelength.
    
    Notes:
        - Speed of light (c) is used in the conversion: c = 299792458 m/s.
        - Wavelengths are sorted in ascending order.
    """
    lambda_vec = 299792458 * (1e9)/(1e12) / (freq + omega0/(2*np.pi))
    sort_indices = lambda_vec.argsort()
    lambda_vec_ordered = lambda_vec[sort_indices]
    Alam = Aw[:, sort_indices]
    return lambda_vec_ordered, Alam




#%%-------------CLASSES---------------------

#Pulse initialization class
class Pulses:
    """
    A class for generating various types of optical pulses.
    """
    def __init__(self):
        pass
    #Super gaussian pulse
    def Sgaussian(self, t, amplitude, width, offset, chirp, order):
        """
        Generate a super-Gaussian pulse.
        
        Parameters:
            t (ndarray): Time array.
            amplitude (float): Pulse amplitude.
            width (float): Pulse width.
            offset (float): Temporal offset of the pulse.
            chirp (float): Chirp parameter.
            order (int): Order of the super-Gaussian pulse.
        
        Returns:
            ndarray: Super-Gaussian pulse in the time domain.
        """
        return np.sqrt(amplitude)*np.exp(- (1+1j*chirp)/2*((t-offset)/(width))**(2*np.floor(order)))*(1+0j)
    
    #Soliton pulse
    def soliton(self, t, width, beta2, gamma, order):
        """
        Generate a soliton pulse.
        
        Parameters:
            t (ndarray): Time array.
            width (float): Pulse width.
            beta2 (float): Group velocity dispersion parameter.
            gamma (float): Nonlinear coefficient.
            order (int): Soliton order.
        
        Returns:
            ndarray: Soliton pulse in the time domain.
        """
        return order * np.sqrt( np.abs(beta2)/(gamma * width**2) ) * (1/np.cosh(t/width))
    
    #Sech-type pulse collision
    def twopulse(self, t, amp1, amp2, width1, width2, offset1, offset2, dfreq):
        """
        Generate a collision of two sech-type pulses.
        
        Parameters:
            t (ndarray): Time array.
            amp1 (float): Amplitude of the first pulse.
            amp2 (float): Amplitude of the second pulse.
            width1 (float): Width of the first pulse.
            width2 (float): Width of the second pulse.
            offset1 (float): Temporal offset of the first pulse.
            offset2 (float): Temporal offset of the second pulse.
            dfreq (float): Frequency difference between the two pulses.
        
        Returns:
            ndarray: Combined pulse in the time domain.
        """
        np.seterr(over='ignore') #Silence overflow wanings (1/inf == 0 for this case)
        pump   = np.sqrt(amp1)*(1/np.cosh(t/width1))
        signal = np.sqrt(amp2)*(1/np.cosh((t + offset2)/width2))*np.exp(-2j*np.pi*dfreq*t)
        np.seterr(over='warn')   #Activate overflow warnings
        return pump + signal
    
    #Order N of pulse (used for sech-type)
    def order(beta2, gamma, width, power):
        """
        Calculate the soliton order.
        
        Parameters:
            beta2 (float): Group velocity dispersion parameter @ pulse frequency.
            gamma (float): Nonlinear coefficient @ pulse frequency.
            width (float): Pulse width.
            power (float): Peak power of the pulse.
        
        Returns:
            float: Soliton order.
        """
        return np.sqrt( gamma * power * width**2 / np.abs(beta2)  )

#General tools class
class Tools:
    """
    A class for analyzing optical pulses and their properties.
    """
    def __init__(self):
        pass
    
    #Power, |A|^2
    def pot(self, signal):
        """
        Calculate the power of a signal.
        
        Parameters:
            signal (ndarray): Input signal.
        
        Returns:
            ndarray: Power of the signal (|A|^2).
        """
        return np.abs(signal)**2
    
    #Signal energy
    def energy(self, t_or_freq, signal):
        """
        Calculate the energy of a signal.
        
        Parameters:
            t_or_freq (ndarray): Time or frequency array.
            signal (ndarray): Input signal.
        
        Returns:
            float: Energy of the signal.
        """
        return np.trapz( self.pot(signal), t_or_freq )
    
    #Number of photons
    def photons(self, freq, spectrum, omega0):
        """
        Calculate the number of photons in a spectrum.
        
        Parameters:
            freq (ndarray): Frequency array.
            spectrum (ndarray): Spectrum of the signal.
            omega0 (float): Central angular frequency.
        
        Returns:
            float: Number of photons.
        """
        return np.sum( np.abs(spectrum)**2 / (freq + omega0) )
    
    #Pulse order N
    def order(self, beta2, gamma, Po, To):
        """
        Calculate the soliton order.
    
        Parameters:
            beta2 (float): Group velocity dispersion parameter.
            gamma (float): Nonlinear coefficient.
            Po (float): Peak power of the pulse.
            To (float): Pulse width.
    
        Returns:
            float: Soliton order.
        """
        return np.sqrt(gamma * Po * To ** 2 / np.abs(beta2))

   
    #Peak power for a fundamental soliton
    def sol_p0(self, beta2, gamma, To):
        """
        Calculate the peak power for a fundamental soliton.
        
        Parameters:
            beta2 (float): Group velocity dispersion parameter.
            gamma (float): Nonlinear coefficient.
            To (float): Pulse width.
        
        Returns:
            float: Peak power for a fundamental soliton.
        """
        return np.abs(beta2)/(gamma * To**2)
    
    #Chirp of a pulse
    def find_chirp(t, signal):
        """
        Calculate the chirp of a pulse.
        
        Parameters:
            t (ndarray): Time array.
            signal (ndarray): Input signal.
        
        Returns:
            ndarray: Chirp of the pulse.
        """
        phase = np.unwrap( np.angle(signal) ) #Angle finds the phase, Unwrap extends it from [0,2pi) to the real numbers.
        #fase = fase - fase[ int(len(fase)/2)  ] #Center the array
        df = np.diff( phase, prepend = phase[0] - (phase[1]  - phase[0]  ), axis=0)
        dt = np.diff( t, prepend = t[0]- (t[1] - t[0] ), axis=0 )
        chirp = -df/dt
        return chirp
    
    #Find frequency shift of pulse
    def find_shift(self, zlocs, freq, A_wr):
        """
        Find the peak power shift in the frequency domain.
        
        Parameters:
            zlocs (ndarray): Array of spatial locations.
            freq (ndarray): Frequency array.
            A_wr (ndarray): Frequency-domain signal evolution.
        
        Returns:
            ndarray: Frequency shifts at each spatial location.
        """
        peaks = np.zeros( len(zlocs), dtype=int )
        dw    = np.copy(peaks)
        for index_z in range( len(zlocs) ):
            peaks[index_z] = np.argmax( self.pot( fftshift(A_wr[index_z]) ) )
            #dw[index_z] = fftshift(2*np.pi*sim_r.freq)[peaks[index_z]]
        dw = fftshift(2*np.pi*freq)[peaks]
        return dw

    #Find wavenumber k
    def find_k(self, Aw, dz, rel_thr=1e-6):
        """
        Calculate the wavenumber k and phase evolution.
        
        Parameters:
            Aw (ndarray): Frequency-domain signal.
            dz (float): Spatial step size.
            rel_thr (float): Relative threshold for signal amplitude.
        
        Returns:
            tuple:
                - ks (ndarray): Wavenumber array.
                - phis (ndarray): Phase array.
        """
        Aw = Aw.T  
        ks    = np.zeros_like(Aw, dtype='float64')
        phis  = np.zeros_like(ks,  dtype='float64')
        mask = abs(Aw)>rel_thr*np.max(np.abs(Aw))
        phis[mask] = np.unwrap( np.angle(Aw[mask]), axis=0 )
        ks = np.diff(phis)/dz
        ks[np.diff(mask) != 0] = 0
        return ks, phis

#Unit handling class
class Units:
    """
    A class for handling unit conversions.
    """
    def __init__(self):
        
        #Useful constants (in m/s/J)
        self.constants = { "c": 299792458, "h": 6.62607015e-34, "hb": 6.62607015e-34/(2*np.pi) }
        
        #Metric prefixes and their corresponding factors (o is the unit prefix)
        self.metric_prefixes = {
            'Y': 1e24, 'Z': 1e21, 'E': 1e18, 'P': 1e15, 'T': 1e12, 'G': 1e9, 'M': 1e6, 'k': 1e3,
            'h': 1e2, 'da': 1e1, '': 1, 'o':1, 'd': 1e-1, 'c': 1e-2, 'm': 1e-3, 'u': 1e-6, 'n': 1e-9,
            'p': 1e-12, 'f': 1e-15, 'a': 1e-18, 'z': 1e-21, 'y': 1e-24
        }

    def parse_unit(self, unit):
        """
        Parse a unit string into its components and exponents.
        
        Parameters:
            unit (str): The unit string to parse (e.g., "km^2").
        
        Returns:
            dict: A dictionary where keys are unit prefixes and values are their exponents.
        """
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
        """
        Convert a unit from one form to another using metric prefixes.
        
        Parameters:
            original_unit (str): The original unit string (e.g., "km").
            target_unit (str): The target unit string (e.g., "m").
        
        Returns:
            float: The conversion factor between the original and target units.
            
        Example:
            Convert beta4 parameter from fs^4/um to ps^4/m
            >>> b4  = -1e-5   * units.convert_unit("f^4 u^-1", "p^4 o^-1")
        """
        original_parsed = self.parse_unit(original_unit)
        target_parsed = self.parse_unit(target_unit)

        conversion_factor = 1.0

        #Calculate the conversion factor for prefixes in the original unit
        for prefix, exp in original_parsed.items():
            if prefix in target_parsed:
                target_exp = target_parsed[prefix]
                conversion_factor *= (self.metric_prefixes[prefix] ** (exp - target_exp))
            else:
                conversion_factor *= (self.metric_prefixes[prefix] ** exp)
        
        #Handle prefixes in the target unit that are not in the original unit
        for prefix, exp in target_parsed.items():
            if prefix not in original_parsed:
                conversion_factor /= (self.metric_prefixes[prefix] ** exp)

        return conversion_factor


#Simulation parameters class
class Sim:
    """
    A class for defining simulation parameters, including time and frequency arrays.
    """
    def __init__(self, N, Tmax):
        self.N = N               #Number of temporal points
        self.Tmax   = Tmax       #Time window [-Tmax,Tmax]
        self.t_step = 2.0*Tmax/N #Temporal step
        self.tiempo = np.arange(-N/2,N/2)*self.t_step #Time array
        self.dW     = np.pi/Tmax #Frequency step
        self.freq   = fftshift( np.pi * np.arange(-N/2,N/2) / Tmax )/(2*np.pi) #Frequency array (1/ps)
        
    @property #fftshifted frequency
    def sfreq(self):
        return fftshift(self.freq)
    @property #Angular frequency
    def omega(self):
        return 2*np.pi*self.freq
    @property #fftshifted angular frequency
    def somega(self):
        return fftshift( 2*np.pi*self.freq )
    @property #Wavelength array
    def lam(self, omega0):
        lam_vec, _ = Adapt_Vector(self.freq, omega0, np.zeros([1, self.freq.size]) )
        return lam_vec
        
#Fiber parameter class (with useful methods)
class Fibra:
    """
    A class for defining fiber parameters and calculating related properties.
    """

    def __init__(self, L, gamma, gamma1, alpha, lambda0, TR=3e-3, fR=0.18, beta1=0, beta2=0, beta3=0, betas=None):
        self.L = L              # Fiber length
        self.gamma = gamma      # Fiber gamma0 (SPM)
        self.gamma1 = gamma1    # Fiber gamma1 (self-steepening)
        self.alpha = alpha      # Fiber alpha (attenuation)
        self.TR = TR            # Fiber TR value (for Raman)
        self.fR = fR            # Fiber fR value (for Raman)
        self.lambda0 = lambda0  # Central wavelength
        self.omega0 = 2 * np.pi * 299792458 * (1e9) / (1e12) / lambda0  #Centra angular frequency
        self.beta1 = beta1      # Beta1 parameter (optional), used to change the moving frame

        #Set beta2 and beta3 parameters when betas is not None
        if betas is not None:
            self.betas = betas
            self.beta2 = betas[0]
            self.beta3 = betas[1] if len(betas) > 1 else 0
        #Set betas array when beta2 and beta3 are given
        else:
            self.betas = [beta2, beta3] if beta2 is not None and beta3 is not None else [0, 0]
            self.beta2 = beta2
            self.beta3 = beta3

        #Find zero dispersion wavelength when beta2 and beta3 are given (doesn't work with general betas array)
        if self.beta3 != 0:
            self.w_zdw = -self.beta2 / self.beta3 + self.omega0
            self.zdw = 2 * np.pi * 299792458 * (1e9) / (1e12) / self.w_zdw
        else:
            self.w_zdw = None
            self.zdw = None

        #Find zero nonlinearity wavelength when gamma0 and gamma1 are given
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

    #Angular frequency to wavelength method
    def omega_to_lambda(self, w):
        return 2 * np.pi * 299792458 * (1e9) / (1e12) / (self.omega0 + w)

    #Wavelength to angular frequency method
    def lambda_to_omega(self, lam):
        return 2 * np.pi * 299792458 * (1e9) / (1e12) * (1 / lam - 1 / self.lambda0)

    #Gamma as a function of frequency
    def gamma_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        return self.gamma + self.gamma1 * w
    
    #Beta1 as a function of frequency
    def beta1_w(self, w, wavelength=False):
        if wavelength:
            w = self.lambda_to_omega(w)
        if self.betas != 0:
            beta1 = 0
            for i, beta in enumerate(self.betas):
                beta1 += beta * w ** (i+1) / np.math.factorial(i+1)
        else:
            beta1 = self.beta2 * w + self.beta3 * w**2
        if self.beta1 != 0:
            beta1 = beta1 + self.beta1 
        return beta1

    #Beta2 as a function of frequency
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

    #General ZDW finder
    def calculate_zdw(self):
        # Solve the polynomial equation for omega
        coefficients = self.betas[::-1]
        omega_solutions = np.roots(coefficients)

        # Convert to absolute frequency
        w_zdw = omega_solutions + self.omega0

        # Calculate ZDW
        ZDW = 2 * np.pi * 299792458 * (1e9) / (1e12) / w_zdw
        return ZDW

#Data: Runs and stores simulation results
class Data:
    """
    A class to store the results of a simulation, including temporal and spectral evolution.
    
    Attributes:
        z (ndarray): Propagation distance array.
        W (ndarray): Spectral evolution data.
        fib (Fibra): Fiber parameters used in the simulation.
        sim (Sim): Simulation parameters used in the simulation.
    """
    def __init__(self, solve_function, sim, fibra, pulse,  **kwargs):
        """
        Initialize the Data object by solving the simulation.
        
        Parameters:
            solve_function (function): Function to solve the simulation.
            sim (Sim): Simulation parameters.
            fibra (Fibra): Fiber parameters.
            pulse (ndarray): Initial pulse data.
            **kwargs: Additional arguments for the solve function.
        """
        self.z, self.W = solve_function(sim, fibra, pulse, **kwargs) #Set solve function and solve simulation
        self.fib = fibra #Store fiber parameters
        self.sim = sim   #Store simulation parameters
    
    #Temporal evolution of the simulation
    @property
    def T(self):
        return IFT(self.W)
    
    #Get shifted spectral evolution
    @property
    def Ws(self):
        return fftshift( self.W )
    
    #Definimos función de guardado
    def save(self, filename, other_par=None):
        """
        Save the simulation results and parameters to files.
        
        Parameters:
            filename (str): Base filename for saving the data.
            other_par (dict, optional): Additional parameters to save.
        """
        fib = getattr(self, "fib", None)
        sim = getattr(self, "sim", None)
        saver(self, filename, other_par)

#Dispersion operator, used by all solvers for the linear part of equation
def disp_op(sim:Sim, fib:Fibra):
    """
    Calculate the dispersion operator for the linear part of the equation.
    
    Parameters:
        sim (Sim): Simulation parameters, including angular frequency array.
        fib (Fibra): Fiber parameters, including dispersion coefficients and attenuation.
    
    Returns:
        ndarray: Dispersion operator in the frequency domain.
    """
           
    #If only beta2 and beta3 parameters are passed by the Fiber objct.
    D_W = 1j * fib.beta2/2 * sim.omega**2 + 1j * fib.beta3/6 * sim.omega**3 - fib.alpha/2
    
    #If Fiber includes betas array, use that instead
    if fib.betas:
        D_w = 0
        for i in range(len(fib.betas)):
            D_w = D_w + 1j*fib.betas[i]/np.math.factorial(i+2) * sim.omega**(i+2)
    
    #If beta1 != 0, add the corresponding term to the linear operator
    if fib.beta1:
        D_w = D_w + 1j*fib.beta1*sim.omega
        
    return D_w

#%% Saving and loading functions

# Save data        
def saver(data:Data, filename, other_par = None):
    """
    Save simulation results and parameters to files.
    
    Parameters:
        data (Data): The simulation data object to save.
        filename (str): Base filename for saving the data.
        other_par (str or list, optional): Additional parameters to save.
    """
    
    # Save simulation and fiber parameters in a dictionary
    metadata = {'Sim': data.sim.__dict__, 'Fibra': data.fib.__dict__} #sim.__dict__ = {'puntos'=N, 'Tmax'=70, ...}

    # Save data in filename-data.txt with pickle
    with open(f"{filename}-data.txt", 'wb') as f:
        pickle.dump(data, f)
        
    # Save parameters to filename-param.txt in human-readable text (not for loading!)
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

# Load data  
def loader(filename):
    """
    Load simulation data from a file.
    
    Parameters:
        filename (str): Base filename to load the data from.
    
    Returns:
        Data: The loaded simulation data object.
    """
    with open(f"{filename}-data.txt", 'rb') as f:
        data = pickle.load(f)
    return data


