import matplotlib.pyplot as plt
import numpy as np
import sys

from params import *

with np.load('fluxes.npz', allow_pickle=True) as data:
	x = data['x']
	F = data['F']
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

r = R*np.arccos(x.clip(-R, R)/R)
ax.plot(r, F*(1 - R/L), 'bo ', markersize=3)
ax.plot(r, x, 'ro ', markersize=3)
ax.plot(r, 4*r)
plt.show()
