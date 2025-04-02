import numpy as np

gauss = lambda x, y, x0, y0, dx, dy : np.exp(-((x - x0)/2*dx)**2) * np.exp(-((y - y0)/2*dy)**2)

# Gaussian electron wavepacket (phi)
def gaussian_wavepacket(x, y, x0, y0, ox, oy, k_x0, k_y0) :
    
    return 1/(2*ox**2*np.pi)**(1/4) * 1/(2*oy**2*np.pi)**(1/4) * gauss(x, y, x0, y0, ox, oy) * np.exp(1.0j * (k_x0*x + k_y0*y))

# (Heavside) step potential
def step_potential(V0, x, y, x0, x1, y0, y1) :
    
    return (V0 * ((x[:, None] >= x0) & (x[:, None] <= x1) & (y[None, :] >= y0) & (y[None, :] <= y1))).ravel()

# Solenoid vector potential A (B = curlA)
def solenoid_vector_potential(I, R, x, y, x0=0.0, y0=0.0) :
    
    xx, yy = np.meshgrid(x, y, indexing = "ij")
    
    r = np.sqrt((xx - x0)**2 + (yy - y0)**2)    # Radial distance from centre (x0, y0)
    theta = np.arctan2(yy - y0, xx - x0)  # Compute the angular direction
    
    A_r, A_x, A_y = np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
    
    A_r[r > R] = (I*R**2)/(2*r[r > R])
    
    A_x = - A_r * np.sin(theta)  # Azimuthal component
    A_y = A_r * np.cos(theta)
    
    return A_x.ravel(), A_y.ravel()

# Returns normalization constant
def normalize(psi, x, y, Nx, Ny) :
    return np.trapz(np.trapz(((abs(psi))**2).reshape(Ny,Nx), x).real, y).real

# Draws double slit in 2D plot
def plot_double_slit_2D(ax1, top, bottom, separation) :
    
    ax1.vlines(bottom[0], bottom[2], bottom[3], colors='white', zorder=2)
    ax1.vlines(bottom[1], bottom[2], bottom[3], colors='white', zorder=2)
    ax1.vlines(top[0], top[2], top[3], colors='white', zorder=2)
    ax1.vlines(top[1], top[2], top[3], colors='white', zorder=2)
    ax1.hlines(bottom[3], bottom[0], bottom[1], colors='white', zorder=2)
    ax1.hlines(top[2], bottom[0], bottom[1], colors='white', zorder=2)
    ax1.vlines(separation[0], separation[2], separation[3], colors='white', zorder=2)
    ax1.vlines(separation[1], separation[2], separation[3], colors='white', zorder=2)
    ax1.hlines(separation[2], separation[0], separation[1], colors='white', zorder=2)
    ax1.hlines(separation[3], separation[0], separation[1], colors='white', zorder=2)

# Draws double slit in 3D
def plot_double_slit_3D(ax2, top, bottom, separation) :

    z_i = 0.0
    ax2.plot([bottom[0],bottom[1],bottom[1],bottom[0],bottom[0]], [bottom[2],bottom[2],bottom[3],bottom[3],bottom[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([top[0],top[1],top[1],top[0],top[0]], [top[2],top[2],top[3],top[3],top[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([separation[0],separation[1],separation[1],separation[0],separation[0]], [separation[2],separation[2],separation[3],separation[3],separation[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)