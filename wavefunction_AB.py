import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab

# Define QM constants
EPSILON_0 = 8.65e-12    # Vacuum permittivity [F/m]
HBAR = 1.05e-34         # Planck's reduced constant h/2pi [Js]
E = 1.6e-19             # Elementary charge [C]
M_E = 9.11e-31          # Electron mass [kg]
A_0 = 5.29e-11          # Bohr radius [m]
MU_0 = 4*np.pi*1e-7     # Vacuum permeability [H/m]

class WaveFunctionAB:
    
    def __init__(self, x, y, psi_0, V, dt, A_x, A_y, hbar = 1.0, m = 1.0, t_0 = 0.0, q = E) :
        
        '''
        x, y        x- and y-interval on which problem is defined   
        psi_0       initial wavefunction
        V           potential V(x,y)
        dt          timestep
        A_x, A_y    x- and y-components of magnetic vector potential
        hbar        reduced Plank's constant (default=1)
        m           particle mass (default=1)
        t_0         initial time (default=0)   
        q           particle charge (degfault=1eV)
        
        psi         wavefunction
        dx, dy      spatial resolution
        Nx, Ny      x- and y- grid size
        t           time stamp
        alpha       dicretization constant in CN scheme
        A, M        LHS, RHS sparse matrices (CSC) for CN_scheme: Ax_{n+1} = Mx_{n}
        '''
        
        self.psi = np.array(psi_0, dtype = np.complex128)
        self.V = np.array(V, dtype = np.complex128)
        self.A_x = A_x
        self.A_y = A_y
        
        # Spatial domain
        self.x, self.y = np.array(x), np.array(y)
        self.dx, self.dy = x[1] - x[0], y[1] - y[0]
        self.Nx, self.Ny = len(x), len(y)
        self.t = t_0
        self.dt = dt
        self.alpha = self.dt/(4*self.dx**2)
        
        # Define QM consants
        self.hbar = hbar
        self.m = m
        self.q = q
                
        # PML parameters
        
        # Construct LHS (A) and RHS (M) matrices discretizising the minimal coupling eq. 
        
        main_diag_A = (1.0j - 4*self.alpha - self.V*dt/2).ravel()    # Central points (i,j) A matrix
        main_diag_M = (1.0j + 4*self.alpha + self.V*dt/2).ravel()     # Central points (i,j) M matrix
        
        off_diag_below = off_diag_left = self.alpha - 0.5j * (q/hbar) * (self.A_x * self.dt/self.dx - self.A_y * self.dt/self.dy).ravel()   # (i-1,j) and (i,j-1) points
        off_diag_above = off_diag_right = self.alpha + 0.5j * (q/hbar) * (self.A_x * self.dt/self.dx + self.A_y * self.dt/self.dy).ravel()  # (i+1,j) and (i,j+1) points
        
        diags_A = np.array([main_diag_A, off_diag_below, off_diag_above, off_diag_left, off_diag_right])   # A diagonals
        diags_M = np.array([main_diag_M, -off_diag_below, -off_diag_above, -off_diag_left, -off_diag_right])   # M diagonals
        
        offsets = np.array([0, -1, 1, -self.Ny, self.Ny])   # Five-point stencil in CN-scheme (column-major ordering)
        
        self.A = sparse.diags(diags_A, offsets, shape = (self.Nx*self.Ny, self.Nx*self.Ny), format = 'csc')    # A
        self.M = sparse.diags(diags_M, offsets, shape = (self.Nx*self.Ny, self.Nx*self.Ny), format = 'csc')    # M

        # print(self.A.shape, np.shape(self.M.dot(self.psi)))
    

    def prob(self) :
        
        return (abs(self.psi))**2
    
    # Returns normalisation constant
    def total_prob(self) :      
        
        return np.trapz(np.trapz((self.prob()).reshape(self.Ny,self.Nx), self.x).real, self.y).real
    
    # def PML(self) :
    
    #     ...
    
    def CN_step(self) :
        
        # PML ?
        
        self.psi = bicgstab(self.A, self.M.dot(self.psi.ravel()), x0 = self.psi.ravel(), atol = 1e-6)[0]
        self.t += self.dt