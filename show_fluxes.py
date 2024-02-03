import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from params import *

mpl.rc('mathtext', fontset='cm')
with np.load('fluxes.npz', allow_pickle=True) as data:
	theta = data['theta']
	F = data['F']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

r = theta*R
a = theta/np.pi
ax.plot(a, -F, 'bo ', markersize=3, label='Numerical', zorder=2)
ax.plot(a, 4*r, 'r--', label='Analytical for small spot', zorder=1, linewidth=1)
ax.scatter([1], [4*np.pi], marker='+', c='magenta', label='Analytical for full sphere', zorder=3)
ax.set_xlabel(r'$r_{\mathrm{spot}}/\pi R$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
plt.show()
