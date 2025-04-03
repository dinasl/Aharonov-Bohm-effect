import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.sparse.linalg import bicgstab

class WaveFunction:
    
    def __init__(self, x, y, psi_0, V, dt, t_0 = 0.0) :
        
        '''
        x, y        x- and y-interval on which problem is defined   
        psi_0       initial wavefunction
        V           potential V(x,y)
        dt          timestep
        hbar        reduced Plank's constant (defualt=1)
        m           particle mass (default=1)
        t_0         initial time (default=0)   
        
        psi         wavefunction
        dl          spatial resolution
        Nx, Ny      x- and y- grid size
        t           time stamp
        alpha       dicretization constant in CN scheme
        A, M        LHS, RHS sparse matrices (CSC) for CN_scheme: Ax_{n+1} = Mx_{n}
        '''
        
        self.psi = np.array(psi_0, dtype = np.complex128)
        self.V = np.array(V, dtype = np.complex128)
        
        # Spatial domain
        self.x, self.y = np.array(x), np.array(y)
        self.dl = x[1] - x[0]               # Spatial domain (grid) resolution
        self.Nx, self.Ny = len(x), len(y)   # Number of grid points
        self.t = t_0
        self.dt = dt
        
        self.alpha = self.dt/(4*self.dl**2)

        # Construct LHS (A) and RHS (M) CSC matrices
        
        main_diag_A = (1.0j - 4*self.alpha - self.V*dt/2).ravel()    # Central points (i,j) A matrix
        main_diag_M = (1.0j + 4*self.alpha + self.V*dt/2).ravel()     # Central points (i,j) M matrix
        
        off_diag = self.alpha * np.ones(self.Nx*self.Ny, dtype = np.complex128)    # Four eighbouring points
        
        diags_A = np.array([main_diag_A, off_diag, off_diag, off_diag, off_diag])   # A diagonals
        diags_M = np.array([main_diag_M, -off_diag, -off_diag, -off_diag, -off_diag])   # M diagonals
        
        offsets = np.array([0, -1, 1, -self.Ny, self.Ny])   # Five-point stencil in CN-scheme (column-major ordering)
        
        self.A = sparse.diags(diags_A, offsets, shape = (self.Nx*self.Ny, self.Nx*self.Ny), format = 'csc')    # A
        self.M = sparse.diags(diags_M, offsets, shape = (self.Nx*self.Ny, self.Nx*self.Ny), format = 'csc')    # M
        

    # Return probability density at each (x, y)
    def prob(self) :
        
        return (abs(self.psi))**2
    
    # Returns normalisation constant
    def total_prob(self) :      
        
        return np.trapz(np.trapz((self.prob()).reshape(self.Ny,self.Nx), self.x).real, self.y).real

    # Crank-Nicholson step
    def CN_step(self) :
        
        # self.psi = spsolve(self.A, self.M.dot(self.psi)) # Solve Ax_{n+1} = Mx_{n} at time t # bicgstab much faster
        self.psi = bicgstab(self.A, self.M.dot(self.psi.ravel()), x0 = self.psi.ravel(), atol = 1e-6)[0] # Solve Ax_{n+1} = Mx_{n} at time t
        
        self.t += self.dt # Update time