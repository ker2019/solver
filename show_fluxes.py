import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from params import *

model_num = int(sys.argv[1])
mpl.rc('mathtext', fontset='cm')
with np.load(model_names[model_num] + '-fluxes.npz', allow_pickle=True) as data:
	theta = data['theta']
	F = data['F']
	spot_area = data['spot_area']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

if model_num == 0:
	f = 4*np.pi*R
	full_area = 4*np.pi*R**2
elif model_num == 1 or model_num == 2:
	f = 8*np.pi*C/np.log((A + C)/(A - C))
	full_area = np.pi * C * B * (A**2/C**2 * (np.pi - 2*np.arccos(C/A)) + 2*B/C)

a = np.sqrt(spot_area/np.pi)
ax.plot(a, -F, 'bo ', markersize=3, label='Numerical', zorder=2)
ax.plot(a, 4*a, 'r--', label='Analytical for small spot', zorder=1, linewidth=1)
ax.scatter([np.sqrt(full_area/np.pi)], [f], marker='+', c='magenta', label='Analytical for full sphere', zorder=3)
ax.set_xlabel(r'$\sqrt{S/\pi}$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
ax.set_title(model_names[model_num])
plt.show()
