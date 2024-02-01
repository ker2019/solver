import sys
import gmsh
import numpy as np
import scipy.sparse as spr
import scipy.sparse.linalg as sprlg

from params import *

mesh = gmsh.model.mesh
Nx = np.newaxis

gmsh.initialize(sys.argv)
gmsh.open(model_name + '.msh')
element_type = mesh.getElementType(element_name, element_order)
element2D_type = mesh.getElementType(element2D_name, element_order)
_, _, _, nodes_per_elem, local_coords_of_nodes, _ = mesh.getElementProperties(element_type)
_, _, _, nodes_per_elem2D, _, _ = mesh.getElementProperties(element2D_type)

local_coords, weights = mesh.getIntegrationPoints(element_type, 'Gauss' + str(2*element_order - 2))
_, grads, _ = mesh.getBasisFunctions(element_type, local_coords, 'GradLagrange' + str(element_order))
grads = np.reshape(grads, (len(weights), nodes_per_elem, 3))
grads_by_grads = (weights[:, Nx, Nx, Nx, Nx] * (grads[:, :, Nx, :, Nx] * grads[:, Nx, :, Nx, :])).sum(axis=0)

print('Extracting necessary data from mesh...')
_, domain = gmsh.model.getEntitiesForPhysicalName('domain')[0]
_, inner_surf = gmsh.model.getEntitiesForPhysicalName('inner surface')[0]
_, outer_surf = gmsh.model.getEntitiesForPhysicalName('outer surface')[0]


# Nodes on entities
nodes_on_inner_surf = mesh.getNodes(2, inner_surf, True, False)[0].astype(np.int32)
nodes_on_outer_surf = mesh.getNodes(2, outer_surf, True, False)[0].astype(np.int32)
nodes_interior = mesh.getNodes(3, domain, False, False)[0].astype(np.int32)
num_of_nodes = mesh.getMaxNodeTag()


coords_of_node = np.zeros((3, num_of_nodes + 1))
for ntag in range(1, num_of_nodes + 1):
	coords_of_node[:, ntag] = gmsh.model.mesh.getNode(ntag)[0]

# Relation between nodes and 3D elements
elems_adj_to_node = np.array([set() for i in range(num_of_nodes + 1)])
elems3D, node_tags = mesh.getElementsByType(element_type)
jacobians, determinants, _ = mesh.getJacobians(element_type, [0, 0, 0], -1)
jacobians = np.reshape(jacobians, (len(determinants), 3, 3))
elems3D = elems3D.astype(np.int32)
max_elem_tag = elems3D.max()
min_elem_tag = elems3D.min()
node_tags = np.reshape(node_tags, (len(elems3D), nodes_per_elem)).astype(np.int32)
determinant_of_elem = np.empty(max_elem_tag - min_elem_tag + 1)
inv_jacobian_of_elem = np.empty((max_elem_tag - min_elem_tag + 1, 3, 3))
for i in range(len(elems3D)):
	for j in range(nodes_per_elem):
		elems_adj_to_node[node_tags[i, j]].add(elems3D[i])
	determinant_of_elem[elems3D[i] - min_elem_tag] = determinants[i]
	inv_jacobian_of_elem[elems3D[i] - min_elem_tag, :, :] = np.linalg.inv(jacobians[i, :, :].T)
nodes_of_elem = node_tags[np.argsort(elems3D), :]

# Neighbour nodes
neigh_nodes_to_node = np.full(num_of_nodes + 1, set())
for ntag in range(1, num_of_nodes + 1):
	for etag in elems_adj_to_node[ntag]:
		neigh_nodes_to_node[ntag] = neigh_nodes_to_node[ntag].union(set(nodes_of_elem[etag - min_elem_tag, :]))
	neigh_nodes_to_node[ntag].discard(ntag)

def prod(n1tag, n2tag):
	res = 0
	if n1tag != n2tag:
		for etag in elems_adj_to_node[n1tag].intersection(elems_adj_to_node[n2tag]):
			n1 = list(nodes_of_elem[etag - min_elem_tag, :]).index(n1tag)
			n2 = list(nodes_of_elem[etag - min_elem_tag, :]).index(n2tag)
			J = inv_jacobian_of_elem[etag - min_elem_tag, :, :]
			D = determinant_of_elem[etag - min_elem_tag]
			res += D * (J[:, Nx, :] * J[Nx, :, :] * grads_by_grads[n1, n2, :, :, Nx]).sum()
	else:
		for etag in elems_adj_to_node[n1tag]:
			n1 = list(nodes_of_elem[etag - min_elem_tag, :]).index(n1tag)
			J = inv_jacobian_of_elem[etag - min_elem_tag, :, :]
			D = determinant_of_elem[etag - min_elem_tag]
			res += D * (J[:, Nx, :] * J[Nx, :, :] * grads_by_grads[n1, n1, :, :, Nx]).sum()
	return res

print('Evaluating linear system coefficients...')
M = spr.lil_matrix((num_of_nodes, num_of_nodes))
for ni in nodes_interior:
	for nj in neigh_nodes_to_node[ni]:
		M[ni - 1, nj - 1] = prod(ni, nj)
	M[ni - 1, ni - 1] = prod(ni, ni)
for ntag in nodes_on_outer_surf:
	M[ntag - 1, ntag - 1] = 1

b = np.zeros(num_of_nodes)
for i in range(num_of_nodes):
	if i + 1 in nodes_on_outer_surf:
		b[i] = 1

print('Solving linear system...')
def solve_for_spot(theta):
	x = R*np.cos(theta)
	N = M.copy()
	for ni in nodes_on_inner_surf:
		if coords_of_node[0, ni] < x:
			for nj in neigh_nodes_to_node[ni]:
				N[ni - 1, nj - 1] = prod(ni, nj)
			N[ni - 1, ni - 1] = prod(ni, ni)
		else:
			N[ni - 1, ni - 1] = 1
	N_ = spr.csr_matrix(N)
	sol, info = sprlg.bicg(N_, b, x0=np.ones(num_of_nodes))
	if info != 0:
		print('Solving error: ', info)
	return np.array(sol)

local_coords2D, weights2D = mesh.getIntegrationPoints(element2D_type, 'Gauss' + str(element_order - 1))
J, D, p = mesh.getJacobians(element2D_type, local_coords2D, inner_surf)
coords_num = len(local_coords2D)//3
elems_num = len(D)//coords_num
J = np.reshape(J, (elems_num, coords_num, 3, 3)).swapaxes(2, 3)
D = np.reshape(D, (elems_num, coords_num))
p = np.reshape(p, (elems_num, coords_num, 3))
def evaluate_flux(vtag, step):
	flux = 0
	for e in range(elems_num):
		for i in range(coords_num):
			g, _ = gmsh.view.probe(vtag, p[e, i, 0], p[e, i, 1], p[e, i, 2],\
				distanceMax=-1, gradient=True, dim=3, step=step)
			flux += weights2D[i] * np.dot(g, J[e, i] @ [0, 0, 1]) * D[e, i]
	return flux


v1 = gmsh.view.add('solution')
n = 30
theta = np.linspace(0, np.pi, n)
F = np.empty(n)
for i in range(n):
	print('%i/%i' % (i, n))
	sol = solve_for_spot(theta[i])
	gmsh.view.addHomogeneousModelData(v1, i, model_name, 'NodeData', range(1, num_of_nodes + 1), sol)
	F[i] = evaluate_flux(v1, i)
np.savez('fluxes.npz', theta=theta, F=F)

gmsh.view.write(v1, model_name + '-solution.msh')
gmsh.finalize()
