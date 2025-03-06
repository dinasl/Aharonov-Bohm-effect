from wavefunction import *
from wavefunction_AB import *

gauss = lambda x, y, x0, y0, dx, dy : np.exp(-((x - x0)/2*dx)**2) * np.exp(-((y - y0)/2*dy)**2)

# Gaussian electron wavepacket (phi)
def gaussian_wavepacket(x, y, x0, y0, ox, oy, k_x0, k_y0) :
    
    return 1/(2*ox**2*np.pi)**(1/4) * 1/(2*oy**2*np.pi)**(1/4) * gauss(x, y, x0, y0, ox, oy) * np.exp(1.0j * (k_x0*x + k_y0*y))

# (Heavside) step potential
def step_potential(V0, x, y, x0, x1, y0, y1) :
    
    return (V0 * ((x[:, None] >= x0) & (x[:, None] <= x1) & (y[None, :] >= y0) & (y[None, :] <= y1))).ravel()

def coulomb_potential() : 
    
    return

# Solenoid vector potential A (B = curlA)
def solenoid_vector_potential(I, R, x, y, x0=0, y0=0) :
    
    mu_0 = 4*np.pi*1e-7
    
    xx, yy = np.meshgrid(x, y, indexing = 'ij')
    
    r = np.sqrt((xx - x0)**2 + (yy - y0)**2)    # Radial distance from centre (x0, y0)
    
    outside = r > R      # Mask of points outside solenoid
    
    A_r, A_x, A_y = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
    
    A_r[outside] = (mu_0*I*R**2)/(4*np.pi*r[outside]**3)
    A_x[outside] = - A_r[outside] * (yy[outside] - y0)
    A_x[outside] = A_r[outside] * (xx[outside] - x0)
    
    return A_x.ravel(), A_y.ravel()