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
ax.plot(a, F*(1 - R/L), 'bo ', markersize=3, label='Numerical')
ax.plot(a, 4*r, 'r', label='Analytical for small or big spot')
ax.set_xlabel(r'$r_{\mathrm{spot}}/\pi R$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
plt.show()
