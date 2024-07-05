import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sys
import gmsh

from params import *

model_num = int(sys.argv[1])
if model_num != 0 and model_num != 4:
	print('ERROR: Implemented only for sphere')
	sys.exit()

gmsh.initialize(sys.argv)

gmsh.open(model_names[model_num] + '-solution.msh')
solution = max(gmsh.view.getTags())

X = np.arange(R, L, 0.01)

mpl.rc('mathtext', fontset='cm')
mpl.rc('axes', labelpad=10)
fig = plt.figure()

if model_num == 4:
	step = int(sys.argv[2])
	with np.load(model_names[model_num] + '-fluxes.npz', allow_pickle=True) as data:
		alpha = data['alpha']
	alpha = alpha[step]
elif model_num == 0:
	with np.load(model_names[model_num] + '-fluxes.npz', allow_pickle=True) as data:
		 step = data['F'].size - 1
	alpha = 0

u = np.array([gmsh.view.probe(solution, -x, 0, 0, step=step, distanceMax=-1, gradient=False)[0] for x in X])
ax1 = fig.add_subplot(2, 1, 1)
ax1.plot(X, u, label='Numerical solution')
ax1.plot(X, (1 + alpha*R**2/X)*R/X * np.exp(alpha*R - 2*alpha*X), label='Analytical solution')
ax1.legend()

dudx = np.array([gmsh.view.probe(solution, -x, 0, 0, step=step, distanceMax=-1, gradient=True)[0][0] for x in X])
ax2 = fig.add_subplot(2, 1, 2)
ax2.plot(X, -dudx, label='Numerical solution')
ax2.plot(X, -(R/X**2 + 2*alpha*R**3/X**3 + 2*alpha*R/X + 2*alpha**2*R**3/X**2)*np.exp(alpha*R - 2*alpha*X), label='Analytical solution')
ax2.legend()

ax2.set_xlabel('$x$', fontsize=15, labelpad=0)
ax2.set_ylabel(r'$\frac{\partial u}{\partial x}$', rotation=0, fontsize=15)
#ax1.set_xlabel('$x$', fontsize=15)
ax1.set_ylabel('$u$', rotation=0, fontsize=15)

plt.show()
gmsh.finalize()
