import sys
import gmsh
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as sprlg
from tqdm import tqdm

from params import *
from mesh_utils import Mesh
from model_utils import *

model_num = int(sys.argv[1])
Nx = np.newaxis

print('Extracting necessary data from mesh...', flush=True)
msh = Mesh('data/' + model_names[model_num], sys.argv)

def eval_int_product_of_FEgradients(n1tag, n2tag):
	res = 0
	m = msh.min_elem3D_tag
	if n1tag != n2tag:
		for etag in msh.elems3D_adj_to_node[n1tag].intersection(msh.elems3D_adj_to_node[n2tag]):
			nodes = msh.nodes_of_elem3D[etag - m].tolist()
			n1 = nodes.index(n1tag)
			n2 = nodes.index(n2tag)
			J = msh.inv_jacobian_of_elem3D[etag - m]
			D = msh.determinant_of_elem3D[etag - m]
			res += D * (J[:, Nx, :] * J[Nx, :, :] * msh.gradients_integration_matrix3D[n1, n2, :, :, Nx]).sum()
	else:
		for etag in msh.elems3D_adj_to_node[n1tag]:
			n1 = msh.nodes_of_elem3D[etag - m].tolist().index(n1tag)
			J = msh.inv_jacobian_of_elem3D[etag - m]
			D = msh.determinant_of_elem3D[etag - m]
			res += D * (J[:, Nx, :] * J[Nx, :, :] * msh.gradients_integration_matrix3D[n1, n1, :, :, Nx]).sum()
	return res

print('Evaluating linear system coefficients...', flush=True)
M = spr.lil_matrix((msh.num_of_nodes, msh.num_of_nodes))
for ni in tqdm(range(msh.nodes_interior_min, msh.nodes_interior_max + 1)):
	for nj in msh.neigh_nodes_to_node[ni]:
		M[ni - 1, nj - 1] = eval_int_product_of_FEgradients(ni, nj)
	M[ni - 1, ni - 1] = eval_int_product_of_FEgradients(ni, ni)
for ni in tqdm(range(msh.nodes_interior_max + 1, msh.num_of_nodes + 1)):
	for nj in msh.neigh_nodes_to_node[ni]:
		r = np.linalg.norm(msh.node_coords[nj])
		M[ni - 1, nj - 1] = r*eval_int_product_of_FEgradients(ni, nj)
	r = np.linalg.norm(msh.node_coords[ni])
	M[ni - 1, ni - 1] = r*eval_int_product_of_FEgradients(ni, ni)

print('Solving linear system...', flush=True)
def solve(**kwargs):
	N = M.copy()
	b = np.zeros(msh.num_of_nodes)
	for ni in range(1, msh.nodes_interior_min):
		if not is_absorbing(msh.node_coords[ni], model_num, **kwargs):
			for nj in msh.neigh_nodes_to_node[ni]:
				N[ni - 1, nj - 1] = eval_int_product_of_FEgradients(ni, nj)
			N[ni - 1, ni - 1] = eval_int_product_of_FEgradients(ni, ni)
		else:
			N[ni - 1, ni - 1] = 1
			b[ni - 1] = 1
	N = spr.csr_matrix(N)
	sol, info = sprlg.bicg(N, b, x0=np.ones(msh.num_of_nodes))
	assert info == 0
	return np.array(sol)

def evaluate_flux(vtag, step, **kwargs):
	p = msh.integration_points_on_inner_surf
	w = msh.integration_weights_on_inner_surf
	J = msh.jacobians_on_inner_surf
	D =  msh.determinants_on_inner_surf
	flux = 0
	for e in range(p.shape[0]):
		if is_in_cluster(msh.inner_surf_elems_centers[e], model_num, **kwargs):
			for i in range(p.shape[1]):
				g, _ = gmsh.view.probe(vtag, p[e, i, 0], p[e, i, 1], p[e, i, 2],\
					distanceMax=-1, gradient=True, dim=3, step=step)
				flux += w[i] * np.dot(g, J[e, i] @ [0, 0, 1]) * D[e, i]
	return flux

def evaluate_area(**kwargs):
	S = 0
	centers = msh.inner_surf_elems_centers
	D = msh.determinants_on_inner_surf
	for e in range(centers.shape[0]):
		if is_in_cluster(centers[e, :], model_num, **kwargs):
			S += D[e, 0]/2 # Only for triangle elements
	return S

v1 = gmsh.view.add('solution')

def get_area():
	if model_num == 0:
		return 2*np.pi*R**2 * (1 - np.cos(theta))
	elif model_num in [1, 4]:
		ct = np.cos(theta)
		return np.pi * C * B * (A**2/C**2 * (np.arccos(ct*C/A) - np.arccos(C/A)) + B/C - ct * np.sqrt(A**2/C**2 - ct**2))
	elif model_num == 3:
		return num_of_spots*np.pi*s**2
	elif model_num in [2, 5]:
		return np.array([evaluate_area(theta=th) for th in theta])

if model_num in [0, 1, 2]:
	F = np.empty(steps_num)
	theta = np.linspace(0, np.sqrt(np.pi), steps_num)**2
	cluster_area = get_area()
	for i in tqdm(range(theta.size)):
		sol = solve(theta=theta[i])
		gmsh.view.addHomogeneousModelData(v1, i, model_names[model_num], 'NodeData', range(1, msh.num_of_nodes + 1), sol, time=cluster_area[i])
		F[i] = evaluate_flux(v1, i, theta=theta[i])
elif model_num == 3:
	num_of_spots = np.arange(0, 200, 10)
	F = np.empty(len(num_of_spots))
	cluster_area = get_area()
	for i in tqdm(range(num_of_spots.size)):
		spots = gen_spots(num_of_spots[i])
		sol = solve(spots=spots)
		gmsh.view.addHomogeneousModelData(v1, i, model_names[model_num], 'NodeData', range(1, msh.num_of_nodes + 1), sol, time=cluster_area[i])
		F[i] = evaluate_flux(v1, i, spots=spots)
if model_num in [4, 5]:
	F = np.empty(steps_num)
	theta = np.linspace(0, np.sqrt(np.pi), steps_num)**2
	sol = solve()
	cluster_area = get_area()
	gmsh.view.addHomogeneousModelData(v1, 0, model_names[model_num], 'NodeData', range(1, msh.num_of_nodes + 1), sol, time=cluster_area[0])
	for i in tqdm(range(theta.size)):
		F[i] = evaluate_flux(v1, 0, theta=theta[i])

np.savez('data/' + model_names[model_num] + '-fluxes.npz', spot_area=cluster_area, F=F)
gmsh.view.write(v1, 'data/' + model_names[model_num] + '-solution.msh')
