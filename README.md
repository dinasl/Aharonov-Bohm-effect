## Simulating the Aharonov-Bohm effect using the Crank-Nicolson scheme

The following implementation has been adapted from the work of \href{https://github.com/FelixDesrochers/Electron-diffraction}{Felix Deschroeters} and \href{https://github.com/bcolmey/Aharonov-Bohm-Space-Charge-Effects-in-Python}{Benjamin Colmey}. The Crank-Nicolson (CN) scheme is used to solve the minimal coupling rule in an attempt to simulate the Aharonov-Bohm effect - a quantum phenomenon demonstrating the physical significance of electromagnetic potentials, even in regions where their respective electromagnetic fields are zero. 

#### Minimal coupling

The wavefunction of a non-relativistic quantum system is governed by the Schrödinger equation (SE)

\begin{align}
    i \hbar \frac{d}{dt} \Psi = \hat{H}\Psi,
\tag{SE} \label{SE}
\end{align}