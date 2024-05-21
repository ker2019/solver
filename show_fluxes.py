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
	if model_num == 0 or model_num == 3:
		full_flux = 4*np.pi*R
		full_area = 4*np.pi*R**2
	elif model_num in [1, 2, 4, 5]:
		full_flux = 8*np.pi*C/np.log((A + C)/(A - C))
		full_area = np.pi * C * B * (A**2/C**2 * (np.pi - 2*np.arccos(C/A)) + 2*B/C)

	a = np.sqrt(spot_area[0]/np.pi)
	ax.plot(a, -F[0], 'bo ', markersize=3, label='Numerical', zorder=2)
	if model_num in [0, 1, 2]:
		ax.plot(a, 4*a, 'r--', label='Analytical for small cluster', zorder=1, linewidth=1)
	elif model_num == 3:
		N = a**2/s**2
		ax.plot(a, full_flux*N*s/(N*s + np.pi*R), 'r--', label='Approximate for small spots', zorder=1, linewidth=1)
	ax.scatter([np.sqrt(full_area/np.pi)], [full_flux], marker='+', c='magenta', label='Analytical for full surface', zorder=3)
	ax.set_title(model_names[model_num].replace('_', ' '))
	ax.set_xlabel(r'$\sqrt{S/\pi}$', fontsize=15, labelpad=0)
else:
	for i in range(len(model_nums)):
		a = np.sqrt(spot_area[i]/np.pi)
		S = spot_area[i]
		S = S/S.max()
		S = S[S < 0.25]
		if model_nums[i] in [1, 4]:
			ax.plot(S, -F[i][0:len(S)], 'o ', markersize=3, label='Polar cluster')
		if model_nums[i] in [2, 5]:
			ax.plot(S, -F[i][0:len(S)], 'o ', markersize=3, label='Lateral cluster')
		ax.set_xlabel(r'$S/S_{full}$', fontsize=15, labelpad=0)


ax.set_ylabel('$\\Phi$', fontsize=15, rotation=0)
ax.legend()
plt.show()
