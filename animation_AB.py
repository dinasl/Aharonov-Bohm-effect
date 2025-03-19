from simulation import *

# import time
import matplotlib.pyplot as plt
from matplotlib import animation
# from mpl_toolkits.axes_grid1 import make_axes_locatable

# Animation parameters
# start_clock = time.time()
N_frames = 100  # Number of frames the animation will consist of
N_levels = 200  # Number of energy levels in contour plot 
# dt = 0.005  # Time step

# N_frames = 20
dt = 0.005

# Define spatial domain
x_min, x_max, dx = y_min, y_max, dy = -13, 13, 0.1

x, y = np.arange(x_min, x_max+dx, dx), np.arange(y_min, y_max+dy, dy)

Nx = len(x)
Ny = len(y)

# Aharanov-Bohm parameters
I = 5e-7        # Current in amps [A]
R = 0.2         # Radius of solenoid [m]

# Create double slit using three potential barriers
V_0 = 4000   # Magntitude of potential barriers
width = 0.025
d = 1.4
s = 0.3

bottom = x0_1, x1_1, y0_1, y1_1 = -width, width, y.min(), -d    # Bottom barrier
top = x0_2, x1_2, y0_2, y1_2 = -width, width, d, y.max()     # Top barrier
separation = x0_3, x1_3, y0_3, y1_3 = -width, width, -d+s, d-s  # Slit separation barrier

V_slit = step_potential(V_0, x, y, x0_1, x1_1, y0_1, y1_1) + step_potential(V_0, x, y, x0_2, x1_2, y0_2, y1_2) + step_potential(V_0, x, y, x0_3, x1_3, y0_3, y1_3)

# Create solenoid vector potentials
x0_solenoid = 1.0
# I = 0.0
A_x, A_y = solenoid_vector_potential(I, R, x, y, x0=x0_solenoid)

# Create initial wavefunctions
xx, yy = np.meshgrid(x, y)

# x0, ox, k_x0 = -3, 0.7, 400
# y0, oy, k_y0 = 0, ox, 0

# psi_0 = gaussian_wavepacket(xx, yy, x0, y0, ox, oy, k_x0, k_y0).transpose().reshape(len(x)*len(y))

k_x0 = 400
k_y0 = 0
ox = oy = 2.0

x0_e1, y0_e1= -3.0, -d  # Electron 1 (left path)
x0_e2,  y0_e2 = x0_e1, d # Electron 2 (right path)

psi_0_e1 = gaussian_wavepacket(xx, yy, x0_e1, y0_e1, ox, oy, k_x0, k_y0).transpose().reshape(Nx*Ny)
psi_0_e2 = gaussian_wavepacket(xx, yy, x0_e2, y0_e2, ox, oy, k_x0, k_y0).transpose().reshape(Nx*Ny)

# Initialize wavefunction objects
# electron = WaveFunctionAB(x, y, psi_0, V_slit, dt, A_x, A_y, hbar = HBAR)
# electron.psi = electron.psi/electron.total_prob()   # Normalize

e1 = WaveFunctionAB(x, y, psi_0_e1, V_slit, dt, A_x, A_y, hbar = HBAR)
e2 = WaveFunctionAB(x, y, psi_0_e2, V_slit, dt, A_x, A_y, hbar = HBAR)

# Normalize (before or afer addition?)

e1.psi = e1.psi/np.sqrt(e1.total_prob())
e2.psi = e2.psi/np.sqrt(e2.total_prob())

# SET UP PLOTS 

fig, ax = plt.subplots(1, 2, figsize = (13,6))

# 2D probability density plot (ax[0])

# z = electron.prob().reshape(electron.Nx, electron.Ny).transpose()
# z1 = e1.prob().reshape(e1.Nx, e1.Ny).transpose()
# z2 = e2.prob().reshape(e2.Nx, e2.Ny).transpose()

# z = z1 + z2 + 2 * np.sqrt(z1 * z2) * np.cos(e1.t - e2.t)  # Interference term

psi = e1.psi + e2.psi
psi = psi/np.sqrt(normalize(psi, x, y, Nx, Ny))

z = ((abs(psi))**2).reshape(Nx, Ny).transpose()

# z1 = e1.prob().reshape(e1.Nx, e1.Ny).transpose()
# z2 = e1.prob().reshape(e2.Nx, e2.Ny).transpose()

level = np.linspace(0, z.max(), N_levels)
# level = np.linspace(0, max([z1.max(), z2.max()]), N_levels)

cmap = ax[0].contourf(xx, yy, z, levels = level, cmap = plt.cm.jet)
# fig.colorbar(cmap, ax=ax[0])    # Add legend

ax[0].set_title('Probability Density (2D)')
ax[0].set_xlabel(R'$x/a_0$')
ax[0].set_ylabel(R'$y/ a_0$')

# ax[0].contourf(xx, yy, z1, levels = level, cmap = plt.cm.jet)
# ax[0].contourf(xx, yy, z2, levels = level, cmap = plt.cm.jet)

# 3D probability density surface plot (ax[1])

# ax[1] = fig.add_subplot(122, projection = '3d')
# surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')

# Cross-sectional probaility density at "wall"

x_wall = 6.0 + 13.0
j_wall = int(x_wall//dx)

ax[1].plot(y, z[:,j_wall])
ax[1].set_ylim(0.0, 0.02)
# ax[1].plot(y, z[:,j_wall])

ax[1].set_title('Probability Density (3D)')
ax[1].set_xlabel(R'$x/a_0$')
ax[1].set_ylabel(R'$y/ a_0$')
# ax[1].set_zlabel(R'$|\psi|^2$')

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

# plot_double_slit(ax[0], ax[1], top, bottom, separation)

def update(frame) :
    
    # electron.CN_step()
    
    # z = electron.prob().reshape(electron.Nx, electron.Ny).transpose()
    
    e1.CN_step()
    e2.CN_step()
    
    psi = e1.psi + e2.psi
    psi = psi/np.sqrt(normalize(psi, x, y, Nx, Ny))

    z = ((abs(psi))**2).reshape(Nx, Ny).transpose()
    
    # z1 = e1.prob().reshape(e1.Nx, e1.Ny).transpose()
    # z2 = e1.prob().reshape(e2.Nx, e2.Ny).transpose()
    
    ax[0].clear()
    ax[1].clear()
    
    # 2D
    
    level = np.linspace(0, z.max(), N_levels)
    # level = np.linspace(0, max([z1.max(), z2.max()]), N_levels)
    
    cmap = ax[0].contourf(xx, yy, z, levels=level, cmap = plt.cm.jet)
    # colorbar = fig.colorbar(cmap, cax=cax1)
    
    # ax[0].contourf(xx, yy, z1, levels = level, cmap = plt.cm.jet)
    # ax[0].contourf(xx, yy, z2, levels = level, cmap = plt.cm.jet)
    
    # 3D
    
    # surface = ax[1].plot_surface(xx, yy, z, cmap = plt.cm.jet, edgecolor = 'none')
    
    # Cross-sectional probaility density at "wall"

    ax[1].plot(y, z[:,j_wall])
    ax[1].set_ylim(0.0, 0.02)
    # ax[1].plot(y, z[:,j_wall])
    ax[0].vlines(x_wall, y.min(), y.max())
    
    # Draw double slit
    # plot_double_slit(ax[0], ax[1], top, bottom, separation)
    
    ax[0].set_title('Probability Density (2D)')
    ax[0].set_xlabel(R'$x/a_0$')
    ax[0].set_ylabel(R'$y/a_0$')
    
    ax[1].set_title('Probability Density (3D)')
    ax[1].set_xlabel(R'$x/a_0$')
    ax[1].set_ylabel(R'$y/a_0$')
    # ax[1].set_zlabel(R'$|\psi|^2$')
    
    print(frame)
    # print(normalize(psi))
    
    # return cmap, surface
    # return [cmap]

# RUN ANIMATION

for i in range(250) :
    e1.CN_step()
    e2.CN_step()

animate = animation.FuncAnimation(fig, update, frames = N_frames, interval = 50, blit = False)
animate.save('double_slit_AB(3).gif', writer = 'Pillow')