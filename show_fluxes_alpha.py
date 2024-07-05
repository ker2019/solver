import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from params import *

mpl.rc('mathtext', fontset='cm')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

model_num = int(sys.argv[1])
with np.load(model_names[model_num] + '-fluxes.npz', allow_pickle=True) as data:
	F = data['F']
	alpha = data['alpha']

ax.plot(alpha, -F, 'bo ', markersize=3, label='Numerical', zorder=2)
ax.plot(alpha, 4*np.pi*R*(1 + alpha - alpha**2 / 3), 'r--', label='Analytical', zorder=1, linewidth=1)

ax.set_xlabel(r'$v\cdot R/2D$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
plt.show()
