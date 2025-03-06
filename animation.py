from simulation import *

import time
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Define QM constants
EPSILON0 = 8.65e-12     # Vacuum permittivity [F/m]
HBAR = 1.05e-34         # Planck's reduced constant h/2pi [Js]
ECHARGE = 1.6e-19       # Elementary charge [C]
M_E = 9.11e-31          # Electron mass [kg]

V_0 = 400   # Magntitude of potential barriers

# Aharonov-Bohm parameters

I = 1e-8        # Current in amps [A]
R = 0.2         # Radius of solenoid [m]

# Simulation/animation timer
# start_clock = time.time()
N_frames = 30  # Number of frames the animation will consist of
N_levels = 200  # Number of energy levels in contour plot 
dt = 0.05  # Time step
# t_arr = np.arange(0, N_frames*dt, dt)  # Time array

# Define spatial domain
x_min, x_max, dx = -12, 12, 0.08
y_min, y_max, dy = -12, 12, dx

x, y = np.arange(x_min, x_max+dx, dx), np.arange(y_min, y_max+dy, dy)

# Create double slit using three potential barriers
bottom = x0_1, x1_1, y0_1, y1_1 = 0, 0.3, y.min(), -2.85    # Bottom barrier
top = x0_2, x1_2, y0_2, y1_2 = x0_1, x1_1, -y1_1, y.max()     # Top barrier
separation = x0_3, x1_3, y0_3, y1_3 = x0_1, x1_1, -0.5, 0.5   # Slit separation barrier

V_slit = step_potential(V_0, x, y, x0_1, x1_1, y0_1, y1_1) + step_potential(V_0, x, y, x0_2, x1_2, y0_2, y1_2) + step_potential(V_0, x, y, x0_3, x1_3, y0_3, y1_3)

# Create initial wavefunction
xx, yy = np.meshgrid(x, y)
x0, ox, k_x0 = -5, 0.7, 20
y0, oy, k_y0 = 0, 0.7, 0

psi_0 = gaussian_wavepacket(xx, yy, x0, y0, ox, oy, k_x0, k_y0).transpose().reshape(len(x)*len(y))

# Initialize wavefunction object
electron = WaveFunction(x, y, psi_0, V_slit, dt)
electron.psi = electron.psi/electron.total_prob()   # Normalize

# electron.CN_step()

# SET UP PLOTS 

fig, ax = plt.subplots(1, 2, figsize = (18,6))

# 2D probability density plot (ax[0])

z = electron.prob().reshape(electron.Nx, electron.Ny).transpose()

level = np.linspace(0, z.max(), N_levels)

cmap = ax[0].contourf(xx, yy, z, levels = level, cmap = plt.cm.jet, zorder = 1)

ax[0].set_title('Probability Density (2D)')
ax[0].set_xlabel(R'$x/a_0$')
ax[0].set_ylabel(R'$y/ a_0$')
# fig.colorbar(cmap, ax=ax[0])    # Add legend

# 3D probability density surface plot (ax[1])

ax[1] = fig.add_subplot(122, projection = '3d')
surface = ax[1].plot_surface(xx, yy, z, cmap = 'viridis', edgecolor = 'none')

ax[1].set_title('Probability Density (3D)')
ax[1].set_xlabel(R'$x/a_0$')
ax[1].set_ylabel(R'$y/ a_0$')
ax[1].set_zlabel(R'$|\psi|^2$')

def plot_double_slit(ax1, ax2, top, bottom, separation) :
    
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
    
    # 3D slits
    z_i = 0.0
    ax2.plot([bottom[0],bottom[1],bottom[1],bottom[0],bottom[0]], [bottom[2],bottom[2],bottom[3],bottom[3],bottom[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([top[0],top[1],top[1],top[0],top[0]], [top[2],top[2],top[3],top[3],top[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)
    ax2.plot([separation[0],separation[1],separation[1],separation[0],separation[0]], [separation[2],separation[2],separation[3],separation[3],separation[2]], z_i*np.ones(5), color='k', linewidth=1, zorder=2, alpha=1.)

plot_double_slit(ax[0], ax[1], top, bottom, separation)

def update(frame) :
    
    electron.CN_step()
    
    z = electron.prob().reshape(electron.Ny, electron.Nx).transpose()
    
    ax[0].clear()
    ax[1].clear()
    
    # 2D
    
    level = np.linspace(0, z.max(), N_levels)
    
    cmap = ax[0].contourf(xx, yy, z, levels=level, cmap = plt.cm.jet)
    # colorbar = fig.colorbar(cmap, cax=cax1)
    
    # 3D
    
    surface = ax[1].plot_surface(xx, yy, z, cmap = 'viridis', edgecolor = 'none')
    
    # Draw double slit
    plot_double_slit(ax[0], ax[1], top, bottom, separation)
    
    ax[0].set_title('Probability Density (2D)')
    ax[0].set_xlabel(R'x/a_0')
    ax[0].set_ylabel(R'y/ a_0')
    
    ax[1].set_title('Probability Density (3D)')
    ax[1].set_xlabel(R'x/a_0')
    ax[1].set_ylabel(R'y/ a_0')
    ax[1].set_zlabel(R'$|\psi|^2$')
    
    print(frame)
    
    return cmap, surface

# RUN ANIMATION

animate = animation.FuncAnimation(fig, update, frames = N_frames, interval = 50, blit = False)
animate.save('double_slit.gif', writer = 'Pillow')