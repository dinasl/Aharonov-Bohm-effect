from simulation import *

import matplotlib.pyplot as plt
from matplotlib import animation

# ANIMATION PARAMETERS 

N_frames = 300  # Number of frames
N_levels = 200  # Number of energy levels in contour plots

# Define spatial domain
dl = 0.1
x_min, x_max = -12, 12
y_min, y_max = -12, 12

x, y = np.arange(x_min, x_max+dl, dl), np.arange(y_min, y_max+dl, dl)

dt = 0.005  # Time step

# CREATE DOUBLE SLIT WITH POTENTIAL BARRIERS

V_0 = 400   # Magntitude of potential barriers
width = 0.15
d = 1.5
s = 0.5

# Barrier positions
bottom = x0_1, x1_1, y0_1, y1_1 = -width, width, y.min(), -d # Bottom barrier
top = x0_2, x1_2, y0_2, y1_2 = -width, width, d, y.max() # Top barrier
separation = x0_3, x1_3, y0_3, y1_3 = -width, width, -d+s, d-s  # Slit separation barrier

V_slit = step_potential(V_0, x, y, x0_1, x1_1, y0_1, y1_1) + step_potential(V_0, x, y, x0_2, x1_2, y0_2, y1_2) + step_potential(V_0, x, y, x0_3, x1_3, y0_3, y1_3)

# INITIALIZE WAVEFUNCTION OBJECT

# Initial wavefunction parameters
xx, yy = np.meshgrid(x, y)
x0, ox, k_x0 = -5, 0.7, 20
y0, oy, k_y0 = 0, ox, 0

# Initial wavefunction
psi_0 = gaussian_wavepacket(xx, yy, x0, y0, ox, oy, k_x0, k_y0).transpose().reshape(len(x)*len(y))

electron = WaveFunction(x, y, psi_0, V_slit, dt)
electron.psi = electron.psi/np.sqrt(electron.total_prob())   # Normalize
# electron.psi = electron.psi/electron.total_prob()   # Normalize

z = electron.prob().reshape(electron.Nx, electron.Ny).transpose() # Probability density

# SET UP PLOTS 

fig, ax = plt.subplots(1, 2, figsize = (13,6))

# 2D probability density plot (ax[0])

level = np.linspace(0, z.max(), N_levels)

cmap = ax[0].contourf(xx, yy, z, levels = level, cmap = plt.cm.jet)

ax[0].set_title('Probability Density (2D)')
ax[0].set_xlabel(R'$x/a_0$')
ax[0].set_ylabel(R'$y/ a_0$')
# fig.colorbar(cmap, ax=ax[0])    # Add legend

# 3D probability density surface plot (ax[1])

ax[1] = fig.add_subplot(122, projection = '3d')
surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')

ax[1].set_title('Probability Density (3D)')
ax[1].set_xlabel(R'$x/a_0$')
ax[1].set_ylabel(R'$y/ a_0$')
ax[1].set_zlabel(R'$|\psi|^2$')

def plot_double_slit_2D(ax1, top, bottom, separation) :
    
    # 2D slits
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
    
def plot_double_slit_3D(ax2, top, bottom, separation) :
    # 3D slits
    z_i = 0.0
    ax2.plot([bottom[0],bottom[1],bottom[1],bottom[0],bottom[0]], [bottom[2],bottom[2],bottom[3],bottom[3],bottom[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([top[0],top[1],top[1],top[0],top[0]], [top[2],top[2],top[3],top[3],top[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([separation[0],separation[1],separation[1],separation[0],separation[0]], [separation[2],separation[2],separation[3],separation[3],separation[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)

plot_double_slit_2D(ax[0], top, bottom, separation)
plot_double_slit_3D(ax[1], top, bottom, separation)

def update(frame) :
    
    electron.CN_step() # Perform timestep
    
    z = electron.prob().reshape(electron.Nx, electron.Ny).transpose() # Probability density
    
    ax[0].clear()
    ax[1].clear()
    
    # 2D plot
    
    level = np.linspace(0, z.max(), N_levels)
    
    cmap = ax[0].contourf(xx, yy, z, levels=level, cmap = plt.cm.jet)
    # colorbar = fig.colorbar(cmap, ax=ax[0])
    
    # 3D plot
    
    surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')
    
    # Draw double slits
    plot_double_slit_2D(ax[0], top, bottom, separation) # 2D
    plot_double_slit_3D(ax[1], top, bottom, separation) # 3D
    
    ax[0].set_title('Probability Density (2D)')
    ax[0].set_xlabel(R'$x/a_0$')
    ax[0].set_ylabel(R'$y/a_0$')
    
    ax[1].set_title('Probability Density (3D)')
    ax[1].set_xlabel(R'$x/a_0$')
    ax[1].set_ylabel(R'$y/a_0$')
    ax[1].set_zlabel(R'$|\psi|^2$')
    
    print(frame)
    
    return cmap, surface

# RUN ANIMATION

animate = animation.FuncAnimation(fig, update, frames = N_frames, interval = 50, blit = False)
animate.save('double_slit(3).gif', writer = 'Pillow')