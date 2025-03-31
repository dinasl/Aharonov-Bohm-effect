import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from wavefunction_AB import WaveFunctionAB

from simulation import gaussian_wavepacket, step_potential, solenoid_vector_potential, normalize, plot_double_slit_2D

# ANIMATION PARAMETERS 

N_frames = 300  # Number of frames the animation will consist of
N_levels = 200  # Number of energy levels in contour plot 
dt = 0.005

# Define spatial domain
dl = 0.1 # Spatial resolution
x_min, x_max = y_min, y_max = -12, 12

x, y = np.arange(x_min, x_max+dl, dl), np.arange(y_min, y_max+dl, dl)

Nx = len(x)
Ny = len(y)

# CREATE DOUBLE SLIT WITH POTENTIAL BARRIERS

V_0 = 4000   # Magntitude of potential barriers

width = 0.03 # Barrier with
d = 1.0 # Slit separation
s = 0.3 # Slit width

bottom = x0_1, x1_1, y0_1, y1_1 = -width, width, y.min(), -d    # Bottom barrier
top = x0_2, x1_2, y0_2, y1_2 = -width, width, d, y.max()     # Top barrier
separation = x0_3, x1_3, y0_3, y1_3 = -width, width, -d+s, d-s  # Slit separation barrier

V_slit = step_potential(V_0, x, y, x0_1, x1_1, y0_1, y1_1) + step_potential(V_0, x, y, x0_2, x1_2, y0_2, y1_2) + step_potential(V_0, x, y, x0_3, x1_3, y0_3, y1_3)

# CREATE SOLENOID VECTOR POTENTIAL 

I = 35.0         # Current
R = 0.2         # Radius of solenoid
I = 0.0

x0_solenoid = 1.0
A_x, A_y = solenoid_vector_potential(I, R, x, y, x0=x0_solenoid)

# INITIALIZE WAVEFUNCTION OBJECT

# Initial wavefunction parameters
xx, yy = np.meshgrid(x, y)
k_x0 = 400
k_y0 = 0
ox = oy = 2.0

x0_e1, y0_e1= -3.0, d/2 
x0_e2,  y0_e2 = x0_e1, -d/2

#Initial wavefunctions
psi_0_e1 = gaussian_wavepacket(xx, yy, x0_e1, y0_e1, ox, oy, k_x0, k_y0).transpose().reshape(Nx*Ny)
psi_0_e2 = gaussian_wavepacket(xx, yy, x0_e2, y0_e2, ox, oy, k_x0, k_y0).transpose().reshape(Nx*Ny)

e1 = WaveFunctionAB(x, y, psi_0_e1, V_slit, dt, A_x, A_y) # Electron 1 (top path)
e2 = WaveFunctionAB(x, y, psi_0_e2, V_slit, dt, A_x, A_y) # Electron 2 (bottom path)

# Normalize
e1.psi = e1.psi/np.sqrt(e1.total_prob())
e2.psi = e2.psi/np.sqrt(e2.total_prob())

# SET UP PLOTS 

fig, ax = plt.subplots(1, 2, figsize = (13,6))

# 2D probability density plot (ax[0])

psi = e1.psi + e2.psi
psi = psi/np.sqrt(normalize(psi, x, y, Nx, Ny))

z = ((abs(psi))**2).reshape(Nx, Ny).transpose()

level = np.linspace(0, z.max(), N_levels)

cmap = ax[0].contourf(xx, yy, z, levels = level, cmap = plt.cm.jet)
# fig.colorbar(cmap, ax=ax[0])    # Add legend

ax[0].set_title('Probability density (2D)')
ax[0].set_xlabel(R'$\tilde{x}$')
ax[0].set_ylabel(R'$\tilde{y}$')

# -------------------------------------------------------------------------------
# ax[0].contourf(xx, yy, z1, levels = level, cmap = plt.cm.jet)
# ax[0].contourf(xx, yy, z2, levels = level, cmap = plt.cm.jet)

# 3D probability density surface plot (ax[1])
# ax[1] = fig.add_subplot(122, projection = '3d')
# surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')
# -------------------------------------------------------------------------------

# Cross-sectional probaility density at "screen" (ax[1])

x_screen = 6.0
x_tot = 6.0 + x_max
j_wall = int(x_tot//dl)

ax[1].plot(y, z[:,j_wall])
ax[1].set_ylim(0.0, 0.04)

ax[1].set_title(f'Probability density at x = {x_screen}')
ax[1].set_xlabel(R'$\tilde{y}$')
ax[1].set_ylabel(R'$|\psi|^2$')

plot_double_slit_2D(ax[0], ax[1], top, bottom, separation)

# ANIMATE
def update(frame) :
    
    # Perform timestep
    e1.CN_step()
    e2.CN_step()
    
    psi = e1.psi + e2.psi
    # psi = psi/np.sqrt(normalize(psi, x, y, Nx, Ny))

    z = ((abs(psi))**2).reshape(Nx, Ny).transpose()
    
    # Clear axes for next frame
    ax[0].clear()
    ax[1].clear()
    
    # 2D probability density plot (ax[0])
    
    level = np.linspace(0, z.max(), N_levels)
    
    cmap = ax[0].contourf(xx, yy, z, levels=level, cmap = plt.cm.jet)
    # colorbar = fig.colorbar(cmap, ax=ax[0]) # Runs suuuper slow!!!
    
    # Draw double slit
    plot_double_slit_2D(ax[0], ax[1], top, bottom, separation)
    ax[0].vlines(x_screen, y.min(), y.max())
    
    ax[0].set_title('Probability density (2D)')
    ax[0].set_xlabel(R'$\tilde{x}$')
    ax[0].set_ylabel(R'$\tilde{y}$')
    
    ax[0].text(-x_max + 1, y_max - 1, f"Frame {frame+1}", color = "white") # Plot frame number
    
    # ------------------------------------------------------------------------------
    # ax[0].contourf(xx, yy, z1, levels = level, cmap = plt.cm.jet)
    # ax[0].contourf(xx, yy, z2, levels = level, cmap = plt.cm.jet)

    # 3D probability density surface plot (ax[1])
    # ax[1] = fig.add_subplot(122, projection = '3d')
    # surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')
    # ------------------------------------------------------------------------------
    
    # Cross-sectional probaility density at "screen" (ax[1])

    ax[1].plot(y, z[:,j_wall])
    ax[1].set_ylim(0.0, 0.04)
    
    ax[1].set_title(f'Probability density at x = {x_screen}')
    ax[1].set_xlabel(R'$\tilde{y}$')
    ax[1].set_ylabel(R'$|\psi|^2$')
    
    print(frame+1)

animate = animation.FuncAnimation(fig, update, frames = N_frames, interval = 50, blit = False)

# animate.save('double_slit_AB(4).gif', writer = 'Pillow')
animate.save('double_slit_AB_noA(4).gif', writer = 'Pillow')