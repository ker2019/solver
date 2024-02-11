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
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

if model_num == 0:
	f = 4*np.pi*R
	r = theta*R
elif model_num == 1:
	C = np.sqrt(A**2 - B**2)
	f = 8*np.pi*C/np.log((A + C)/(A - C))
	r = theta*A

a = theta/np.pi
ax.plot(a, -F, 'bo ', markersize=3, label='Numerical', zorder=2)
ax.plot(a, 4*r, 'r--', label='Analytical for small spot', zorder=1, linewidth=1)
ax.scatter([1], [f], marker='+', c='magenta', label='Analytical for full sphere', zorder=3)
ax.set_xlabel(r'$\theta/\pi$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
plt.show()
