import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys

from params import *

mpl.rc('mathtext', fontset='cm')
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

model_nums = list(map(int, sys.argv[1].split(',')))
F = list()
spot_area = list()
for i in range(len(model_nums)):
	with np.load(model_names[model_nums[i]] + '-fluxes.npz', allow_pickle=True) as data:
		F.append(data['F'])
		spot_area.append(data['spot_area'])

if len(model_nums) == 1:
	model_num = model_nums[0]
	if model_num == 0:
		full_flux = 4*np.pi*R
		full_area = 4*np.pi*R**2
	elif model_num == 1 or model_num == 2:
		full_flux = 8*np.pi*C/np.log((A + C)/(A - C))
		full_area = np.pi * C * B * (A**2/C**2 * (np.pi - 2*np.arccos(C/A)) + 2*B/C)

	a = np.sqrt(spot_area[0]/np.pi)
	ax.plot(a, -F[0], 'bo ', markersize=3, label='Numerical', zorder=2)
	ax.plot(a, 4*a, 'r--', label='Analytical for small spot', zorder=1, linewidth=1)
	ax.scatter([np.sqrt(full_area/np.pi)], [full_flux], marker='+', c='magenta', label='Analytical for full sphere', zorder=3)
	ax.set_title(model_names[model_num])
else:
	for i in range(len(model_nums)):
		a = np.sqrt(spot_area[i]/np.pi)
		ax.plot(a, -F[i], 'o ', markersize=3, label=model_names[model_nums[i]])


ax.set_xlabel(r'$\sqrt{S/\pi}$', fontsize=15, labelpad=0)
ax.set_ylabel('Flux', rotation=0)
ax.legend()
plt.show()
