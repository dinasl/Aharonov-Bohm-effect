import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab

class WaveFunctionAB:
    
    def __init__(self, x, y, psi_0, V, dt, A_x, A_y, hbar = 1, m = 1, t_0 = 0.0, q = 1.602e-19) :
        
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
        A, M        LHS, RHS spatrse matrices of Ax_{n+1} = Mx_{n} system
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
                
        # PML parameters
        
        # Define LHS and RHS matrices discretizising the minimal coupling eq. 
        
        N = (self.Nx - 1) * (self.Ny - 1)           # Number of internal points in domain
        size = 5*N + 2*self.Nx + 2*(self.Ny - 2)    # Number of nonzero entries in matrix system
                                                    # (points and their associated nearest neighbours)
                                                    
        I, J = np.zeros(size), np.zeros(size) # Coordinate lists (COO): row, colum indices
        K_A, K_M = np.zeros(size, dtype = np.complex128), np.zeros(size, dtype = np.complex128) # Values in A and M matrices
        
        k = 0   # K-index (flattened)
        
        # Iterate over all points domain
        for i in range(self.Ny) :
            for j in range(self.Ny) :
                
                index = i + j*self.Ny
                I[k] = index
                
                if i==0 or i==self.Nx-1 or j==0 or j==self.Ny-1 :   # Boundary points
                    
                    J[k] = index
                    K_A[k] = 1
                
                else:                                               # Internal points
                    
                    # Central point (i,j)
                    K_A[k] = 1.0j - 4*self.alpha - self.V[index]*dt/2
                    K_M[k] = 1.0j + 4*self.alpha + self.V[index]*dt/2
                    J[k] = index
                    
                    # Nearest neighbours
                    k += 1                      # (i-1,j)
                    J[k] = (i-1) + j*self.Ny
                    K_A[k] = self.alpha + 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    K_M[k] = -self.alpha - 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    
                    k += 1                      # (i+1,j)
                    J[k] = (i+1) + j*self.Ny
                    K_A[k] = self.alpha - 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    K_M[k] = -self.alpha + 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    
                    k += 1                      # (i,j-1)
                    J[k] = i + (j-1)*self.Ny
                    K_A[k] = self.alpha + 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    K_M[k] = -self.alpha - 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    
                    k += 1                      # (i,j+1)
                    J[k] = i + (j+1)*self.Ny
                    K_A[k] = self.alpha - 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    K_M[k] = -self.alpha + 0.5j * (q/hbar) * (self.A_x[index]/self.dx - self.A_y[index]/self.dy)*self.dt
                    
                k += 1

        self.A = sparse.coo_matrix((K_A,(I,J)), shape = (self.Nx*self.Ny, self.Nx*self.Ny)).tocsc()     # LHS
        self.M = sparse.coo_matrix((K_M,(I,J)), shape = (self.Nx*self.Ny, self.Nx*self.Ny)).tocsc()    # RHS

    def prob(self) :
        
        return (abs(self.psi))**2
    
    # def PML(self) :
    
    def CN_step(self) :
        
        # PML ?
        
        self.psi = bicgstab(self.A, self.M.dot(self.psi), x0 = self.psi, tol = 1e-6)[0]
        self.t += self.dt