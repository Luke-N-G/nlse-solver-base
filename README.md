# NLSE Solver Base
Base code for NLSE (Nonlinear Schrödinger Equation) Simulations.

This code uses the [_Interaction Picture_](https://hal.science/hal-00850518v4/document) method [1], along with a [RK-45 method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), to solve different versions of the nonlinear schrödinger equation. This equation reads, in the frequency domain, as [2]

$$\frac{\partial A_\omega}{\partial z} = i \beta(\omega) A_\omega +  \hat N(\omega)$$

where $A_\omega$ is the fourier transform of the complex envelope of the electrical field, $\beta(\omega)$ the dispersion profile, and $\hat N (\omega)$ the nonlinear operator. The current solvers allow for the simulation of the following nonlinear operators:

## GNLSE (Generalized Nonlinear Schrödinger Equation) [2]
Solves for the GNLSE nonlinear operator. Can be solved with an approximation for the Raman term.

$$\hat N (T) = i \left( \gamma(\omega_0) + i \gamma_1 \frac{\partial}{\partial T} \right) \left( A(T) \int_0^{\infty} R(t') |A(T-t')|^2 dt' \right)$$

Additionally, the solver _solvebarrierNLSE.py_ allows for the inclusion of a temporal refractive index change. 

## pcNLSE (photon-conserving Nonlinear Schrödinger Equation) [3]
Solves for the pcNLSE nonlinear operator:

$$ \hat N(\omega) = i \bar \gamma \mathcal{F}\left( C^{\*} G^2 \right) + i \bar\gamma^{\*} (\omega) \mathcal{C^2G^{\*}}  $$

with $\bar \gamma(\omega) = \frac{1}{2} \left(\gamma(\omega)\times(\omega+\omega_0)^3\right)^{1/4}$, $G_{\omega} = \left( \gamma(\omega)/(\omega + \omega_0) \right)^{1/4} A_{\omega}$ , $C_{\omega} = [\left( \gamma(\omega)/(\omega + \omega_0) \right)^{1/4}]^* A_{\omega}$.

## pcGNLSE (photon-conserving Generalized Nonlinear Schrödinger Equation) [4]
Solves for the pcGNLSE nonlinear operator:

$$\hat N(\omega) = i \bar \gamma(\omega) \mathcal{F}\left(C^{\*} G^2\right) + i \bar \gamma^{\*}(\omega) \mathcal{F}\left(C^2 G^{\*} \right) + i f_R \bar \gamma^{\*}(\omega) \mathcal{F}\left(B \int_0^\infty h_R(\tau) |B(t - \tau)|^2 d\tau - B |B|^2\right)$$


## Example
The _run.py_ script includes the following example:

Propagation of a fundamental soliton in a fiber with Raman scattering. The soliton is shown to suffer Raman-induced self frequency shift (RIFS), as seen in reference [2] 

<img src="https://github.com/user-attachments/assets/becd58da-be87-4231-a3f4-ffb8480705c2" alt="test_output"/>

## References
[1] S. Balac, A. Fernandez, F. Mahé, F. Méhats, and R. Texier-Picard, "The Interaction Picture method for solving the generalized nonlinear Schrödinger equation in optics," ESAIM Mathematical Modelling and Numerical Analysis 50(4), 945–964 (2015).

[2] G. P. Agrawal, Nonlinear Fiber Optics (Academic, 2007).

[3] J. Bonetti, N. Linale, A. D. Sánchez, S. M. Hernandez, P. I. Fierens, and D. F. Grosz, "Modified nonlinear Schrödinger equation for frequency-dependent nonlinear profiles of arbitrary sign," J. Opt. Soc. Am. B 36, 3139-3144 (2019)

[4] J. Bonetti, N. Linale, A. D. Sánchez, S. M. Hernandez, P. I. Fierens, and D. F. Grosz, "Photon-conserving generalized nonlinear Schrödinger equation for frequency-dependent nonlinearities," J. Opt. Soc. Am. B 37, 445-450 (2020)
