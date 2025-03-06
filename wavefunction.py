import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Define QM constants
EPSILON0 = 8.65e-12     # Vacuum permittivity [F/m]
HBAR = 1.05e-34         # Planck's reduced constant h/2pi [Js]
ECHARGE = 1.6e-19       # elementary charge [C]
M_E = 9.11e-31          # electron mass [kg]

class WaveFunction:
    
    def __init__(self, x, y, psi_0, V, dt, hbar = 1.0, m = 1.0, t_0 = 0.0) :
        
        '''
        x, y        x- and y-interval on which problem is defined   
        psi_0       initial wavefunction
        V           potential V(x,y)
        dt          timestep
        hbar        reduced Plank's constant (defualt=1)
        m           particle mass (default=1)
        t_0         initial time (default=0)   
        
        psi         wavefunction
        dx, dy      spatial resolution
        Nx, Ny      x- and y- grid size
        t           time stamp
        alpha       dicretization constant in CN scheme
        A, M        LHS, RHS spatrse matrices of Ax_{n+1} = Mx_{n} system
        '''
        
        self.psi = np.array(psi_0, dtype = np.complex128)
        self.V = np.array(V, dtype = np.complex128)
        
        # Spatial domain
        self.x, self.y = np.array(x), np.array(y)
        self.dx, self.dy = x[1] - x[0], y[1] - y[0]     # Spatial domain (grid) resolution
        self.Nx, self.Ny = len(x), len(y)       # Number of grid points
        self.t = t_0
        self.dt = dt
        self.alpha = self.dt/(4*self.dx**2)
        
        # Define QM consants
        self.hbar = hbar
        self.m = m
                
        # PML parameters
        
        # Define LHS and RHS matrices discretizising the minimal coupling eq. 
        
        N = (self.Nx - 1) * (self.Ny - 1)   # Number of internal points in domain
        size = 5*N + 2*self.Nx + 2*(self.Ny - 2)    # Number of nonzero entries in matrix system
                                                    # (points and their associated nearest neighbours)
                                                    
        I, J = np.zeros(size), np.zeros(size) # Coordinate lists (COO): row, colum indices
        K_A, K_M = np.zeros(size, dtype = np.complex128), np.zeros(size, dtype = np.complex128) # Cooridinate values for A, M matrices
        
        k = 0   # K-index (flattened)
        
        # Iterate over all points domain
        for i in range(self.Ny) :
            for j in range(self.Nx) :
                
                index = i + j*self.Ny
                
                if i==0 or i==self.Nx-1 or j==0 or j==self.Ny-1 :    # Boundary points
                    
                    I[k] = index
                    J[k] = index
                    K_A[k] = 1
                    K_M[k] = 0
                
                else:                                                # Internal points
                    
                    # Central point (i,j)
                    I[k] = index
                    J[k] = index
                    K_A[k] = 1.0j - 4*self.alpha - self.V[index]*dt/2
                    K_M[k] = 1.0j + 4*self.alpha + self.V[index]*dt/2
                    
                    # Nearest neighbours
                    k += 1                      # (i-1,j)
                    I[k] = index
                    J[k] = (i-1) + j*self.Ny
                    K_A[k] = self.alpha
                    K_M[k] = -self.alpha
                    
                    k += 1                      # (i+1,j)
                    I[k] = index
                    J[k] = (i+1) + j*self.Ny
                    K_A[k] = self.alpha
                    K_M[k] = -self.alpha
                    
                    k += 1                      # (i,j-1)
                    I[k] = index
                    J[k] = i + (j-1)*self.Ny
                    K_A[k] = self.alpha
                    K_M[k] = -self.alpha
                    
                    k += 1                      # (i,j+1)
                    I[k] = index
                    J[k] = i + (j+1)*self.Ny
                    K_A[k] = self.alpha
                    K_M[k] = -self.alpha
                    
                k += 1

        self.A = sparse.coo_matrix((K_A,(I,J)), shape = (self.Nx*self.Ny, self.Nx*self.Ny)).tocsc()     # LHS
        self.M = sparse.coo_matrix((K_M,(I,J)), shape = (self.Nx*self.Ny, self.Nx*self.Ny)).tocsc()    # RHS

    # Return probability density at each (x, y)
    def prob(self) :
        
        return (abs(self.psi))**2
    
    # Returns normalisation constant
    def total_prob(self) :      
        
        return np.trapz(np.trapz((self.prob()).reshape(self.Ny,self.Nx), self.x).real, self.y).real

    # Crank-Nicholson step
    def CN_step(self) :
        
        self.psi = spsolve(self.A, self.M.dot(self.psi))    # Solve Ax_{n+1} = Mx_{n} at time t using CN
        
        self.t += self.dt       # Update time