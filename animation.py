import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from wavefunction import WaveFunction

from simulation import gaussian_wavepacket, step_potential, plot_double_slit_2D, plot_double_slit_3D

# ANIMATION PARAMETERS 

N_frames = 200  # Number of frames
N_levels = 200  # Number of energy levels in contour plots

# Define spatial domain
dl = 0.1 # Spatial resolution
x_min, x_max = y_min, y_max = -12, 12

x, y = np.arange(x_min, x_max+dl, dl), np.arange(y_min, y_max+dl, dl)

Nx = len(x)
Ny = len(y)

dt = 0.01  # Time step

# CREATE DOUBLE SLIT WITH POTENTIAL BARRIERS

V_0 = 400   # Magntitude of potential barriers

width = 0.15 # Barrier width
d = 1.5 # Slit separation
s = 0.5 # Slit width

# Barrier positions
bottom = x0_1, x1_1, y0_1, y1_1 = -width, width, y.min(), -d # Bottom barrier
top = x0_2, x1_2, y0_2, y1_2 = -width, width, d, y.max() # Top barrier
separation = x0_3, x1_3, y0_3, y1_3 = -width, width, -d+s, d-s  # Slit separation barrier

V_slit = step_potential(V_0, x, y, x0_1, x1_1, y0_1, y1_1) + step_potential(V_0, x, y, x0_2, x1_2, y0_2, y1_2) + step_potential(V_0, x, y, x0_3, x1_3, y0_3, y1_3)

# ELECTRON

# Initial wavefunction parameters
xx, yy = np.meshgrid(x, y)
x0, ox, k_x0 = -5.0, 0.7, 20
y0, oy, k_y0 = 0, ox, 0

# Initial wavefunction
psi_0 = gaussian_wavepacket(xx, yy, x0, y0, ox, oy, k_x0, k_y0).transpose().reshape(Nx*Ny)

# Initialize wavefunction object
electron = WaveFunction(x, y, psi_0, V_slit, dt)
electron.psi = electron.psi/np.sqrt(electron.total_prob())   # Normalize

z = electron.prob().reshape(electron.Nx, electron.Ny).transpose() # Probability density

# SET UP PLOTS 

fig, ax = plt.subplots(1, 2, figsize = (13,6))

# 2D probability density plot (ax[0])

level = np.linspace(0, z.max(), N_levels)

cmap = ax[0].contourf(xx, yy, z, levels = level, cmap = plt.cm.jet)

ax[0].set_title("Probability Density (2D)")
ax[0].set_xlabel(R"$\tilde{x}$")
ax[0].set_ylabel(R"$\tilde{y}$")
# fig.colorbar(cmap, ax=ax[0])

plot_double_slit_2D(ax[0], top, bottom, separation) # Draw double slit

# Cross-sectional probaility density at "screen" (ax[1])

x_screen = 6.0
x_tot = 6.0 + x_max
j_wall = int(x_tot//dl)

ax[1].plot(y, z[:,j_wall], color = "#002C80")
ax[1].set_ylim(0.0, 0.015)

ax[1].set_title(f"Probability density at x = {x_screen} (1D)")
ax[1].set_xlabel(R"$\tilde{y}$")
ax[1].set_ylabel(R"$|\psi|^2$")

# -------------------------------------------------------------------------------
# 3D probability density surface plot (ax[1])

# ax[1] = fig.add_subplot(122, projection = '3d')
# surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')

# ax[1].set_title('Probability Density (3D)')
# ax[1].set_xlabel(R'$x/a_0$')
# ax[1].set_ylabel(R'$y/ a_0$')
# ax[1].set_zlabel(R'$|\psi|^2$')

# plot_double_slit_3D(ax[1], top, bottom, separation) # Draw double slit
# -------------------------------------------------------------------------------

def update(frame) :
    
    electron.CN_step() # Perform timestep
    
    z = electron.prob().reshape(electron.Nx, electron.Ny).transpose() # Probability density
    
    ax[0].clear()
    ax[1].clear()
    
    # 2D plot (ax[0])
    
    level = np.linspace(0, z.max(), N_levels)
    
    cmap = ax[0].contourf(xx, yy, z, levels=level, cmap = plt.cm.jet)
    # colorbar = fig.colorbar(cmap, ax=ax[0])
    ax[0].vlines(x_screen, y.min(), y.max()) # Draw screen
    
    plot_double_slit_2D(ax[0], top, bottom, separation) # 2D
    
    ax[0].set_title("Probability Density (2D)")
    ax[0].set_xlabel(R"$\tilde{x}$")
    ax[0].set_ylabel(R"$\tilde{y}$")
    
    ax[0].text(-x_max + 1, y_max - 1, fR"$\tau =$ {round(electron.t,3)}", color = "white") # Plot time stamp
    
    # 1D plot (ax[1])
    
    ax[1].plot(y, z[:,j_wall], color = "#002C80")
    ax[1].set_ylim(0.0, 0.015)

    ax[1].set_title(f"Probability density at x = {x_screen} (1D)")
    ax[1].set_xlabel(R"$\tilde{y}$")
    ax[1].set_ylabel(R"$|\psi|^2$")
    
    print(frame+1)
    
    return cmap

# RUN ANIMATION

animate = animation.FuncAnimation(fig, update, frames = N_frames, interval = 50, blit = False)
animate.save("double_slit(4).gif", writer = "Pillow")