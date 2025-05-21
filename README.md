# NLSE Solver Base
Base code for NLSE (Nonlinear Schrödinger Equation) Simulations.

This code uses the [_Interaction Picture_](https://hal.science/hal-00850518v4/document) method [1], along with a [RK-45 method](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html), to solve different versions of the nonlinear schrödinger equation. The current solvers allow for the simulation of the following nonlinear operators:

## GNLSE (Generalized Nonlinear Schrödinger Equation) [2]
Solves for the GNLSE nonlinear operator. Can be solved with an approximation for the Raman term.
<img src="https://github.com/user-attachments/assets/58ea1ef1-7e27-4ecf-8143-621d5f9f02aa" height="60"/>

Additionally, the solver _solvebarrierNLSE.py_ allows for the inclusion of a temporal refractive index change. 

## pcNLSE (photon-conserving Nonlinear Schrödinger Equation) [3]
Solves for the pcNLSE nonlinear operator:

<img src="https://github.com/user-attachments/assets/d2709e22-7d7d-4655-adae-3d57c45fea5a" height="40"/>
<br>
<img src="https://github.com/user-attachments/assets/75658768-735a-47be-8cd0-e5a2b226103f" height="20"/>

## pcGNLSE (photon-conserving Generalized Nonlinear Schrödinger Equation) [4]
Solves for the pcGNLSE nonlinear operator:

<img src="https://github.com/user-attachments/assets/6254ee1b-dd2d-4866-a07f-143243be3cca" height="50"/>

## Example
The _run.py_ script includes the following example:

Propagation of a fundamental soliton in a fiber with Raman scattering. The soliton is shown to suffer Raman-induced self frequency shift (RIFS), as seen in reference [2] 

<img src="https://github.com/user-attachments/assets/becd58da-be87-4231-a3f4-ffb8480705c2" alt="test_output"/>

## References
[1] Stéphane Balac, Arnaud Fernandez, Fabrice Mahé, Florian Méhats, Rozenn Texier-Picard. The Interaction Picture method for solving the generalized nonlinear Schrödinger equation in optics. ESAIM:
Mathematical Modelling and Numerical Analysis, 2016, 50 (4), pp.945-964. ff10.1051/m2an/2015060ff.
ffhal-00850518v4f
[2] G. P. Agrawal, Nonlinear Fiber Optics (Academic, 2007).
[3] J. Bonetti, N. Linale, A. D. Sánchez, S. M. Hernandez, P. I. Fierens, and D. F. Grosz, "Modified nonlinear Schrödinger equation for frequency-dependent nonlinear profiles of arbitrary sign," J. Opt. Soc. Am. B 36, 3139-3144 (2019)
[4] J. Bonetti, N. Linale, A. D. Sánchez, S. M. Hernandez, P. I. Fierens, and D. F. Grosz, "Photon-conserving generalized nonlinear Schrödinger equation for frequency-dependent nonlinearities," J. Opt. Soc. Am. B 37, 445-450 (2020)
