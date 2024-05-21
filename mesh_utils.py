import gmsh
import numpy as np
from params import *

mesh = gmsh.model.mesh

class Mesh:
	def reorder_nodes(self):
		nodes_on_inner_surf = mesh.getNodes(2, self._inner_surf, True, False)[0].astype(np.int32)
		nodes_on_outer_surf = mesh.getNodes(2, self._outer_surf, True, False)[0].astype(np.int32)
		nodes_interior = mesh.getNodes(3, self._domain, False, False)[0].astype(np.int32)
		self.num_of_nodes = len(nodes_on_inner_surf) + len(nodes_interior) + len(nodes_on_outer_surf)
		self.nodes_interior_min = nodes_on_inner_surf.size + 1
		self.nodes_interior_max = nodes_on_inner_surf.size + nodes_interior.size
		mesh.renumberNodes(nodes_on_inner_surf.tolist() + nodes_interior.tolist() + nodes_on_outer_surf.tolist(), np.arange(1, self.num_of_nodes + 1))

	def evaluate_adjacents_and_jacobians(self):
		self.elems3D_adj_to_node = np.array([set() for i in range(self.num_of_nodes + 1)])

		elems3D, node_tags = mesh.getElementsByType(self.element3D_type)
		elems3D = elems3D.astype(np.int32)
		max_elem3D_tag = elems3D.max()
		min_elem3D_tag = elems3D.min()
		node_tags = np.reshape(node_tags, (len(elems3D), self.nodes_per_elem3D)).astype(np.int32)

		J, D, _ = mesh.getJacobians(self.element3D_type, [0, 0, 0], -1)
		J = np.reshape(J, (len(D), 3, 3))
		self.determinant_of_elem3D = np.empty(max_elem3D_tag - min_elem3D_tag + 1)
		self.inv_jacobian_of_elem3D = np.empty((max_elem3D_tag - min_elem3D_tag + 1, 3, 3))

		for i, e in enumerate(elems3D):
			for j in range(self.nodes_per_elem3D):
				self.elems3D_adj_to_node[node_tags[i, j]].add(e)
			self.determinant_of_elem3D[e - min_elem3D_tag] = D[i]
			self.inv_jacobian_of_elem3D[e - min_elem3D_tag, :, :] = np.linalg.inv(J[i, :, :].T)
		self.nodes_of_elem3D = node_tags[np.argsort(elems3D), :]
		self.min_elem3D_tag = min_elem3D_tag

	def evaluate_neighbour_nodes(self):
		self.neigh_nodes_to_node = np.array([set() for i in range(self.num_of_nodes + 1)])
		for ntag in range(1, self.num_of_nodes + 1):
			for etag in self.elems3D_adj_to_node[ntag]:
				self.neigh_nodes_to_node[ntag] = self.neigh_nodes_to_node[ntag].union(set(self.nodes_of_elem3D[etag - self.min_elem3D_tag, :]))
		self.neigh_nodes_to_node[ntag].discard(ntag)

	def evaluate_integration_on_inner_surf(self):
		local_coords2D, weights2D = mesh.getIntegrationPoints(self.element2D_type, 'Gauss' + str(2*element_order - 1))
		J, D, points = mesh.getJacobians(self.element2D_type, local_coords2D, self._inner_surf)
		coords_num = len(local_coords2D)//3
		self.elems_num_on_inner_surf = len(D)//coords_num

		etags, _ = mesh.getElementsByType(self.element2D_type, self._inner_surf)
		self.jacobians_on_inner_surf = np.reshape(J, (-1, coords_num, 3, 3)).swapaxes(2, 3)
		self.determinants_on_inner_surf = np.reshape(D, (-1, coords_num))
		self.integration_points_on_inner_surf = np.reshape(points, (-1, coords_num, 3))
		self.integration_weights_on_inner_surf = weights2D
		self.inner_surf_elems_centers = mesh.getJacobians(self.element2D_type, [0, 0, 0, 1, 0, 0, 0, 1, 0], self._inner_surf)[2].reshape((-1, 3, 3)).mean(axis=1)

	def __init__(self, model_name, argv):
		gmsh.initialize(argv)
		gmsh.open(model_name + '.msh')
		_, self._domain = gmsh.model.getEntitiesForPhysicalName('domain')[0]
		_, self._inner_surf = gmsh.model.getEntitiesForPhysicalName('inner surface')[0]
		_, self._outer_surf = gmsh.model.getEntitiesForPhysicalName('outer surface')[0]

		self.element3D_type = mesh.getElementType(element_name, element_order)
		self.element2D_type = mesh.getElementType(element2D_name, element_order)
		self.nodes_per_elem3D = mesh.getElementProperties(self.element3D_type)[3]
		self.nodes_per_elem2D = mesh.getElementProperties(self.element2D_type)[3]
		self.local_coords3D = mesh.getElementProperties(self.element3D_type)[4]

		Nx = np.newaxis
		local_coords, weights = mesh.getIntegrationPoints(self.element3D_type, 'Gauss' + str(2*element_order - 2))
		_, grads, _ = mesh.getBasisFunctions(self.element3D_type, local_coords, 'GradLagrange' + str(element_order))
		grads = np.reshape(grads, (-1, self.nodes_per_elem3D, 3))
		self.gradients_integration_matrix3D = (weights[:, Nx, Nx, Nx, Nx] * (grads[:, :, Nx, :, Nx] * grads[:, Nx, :, Nx, :])).sum(axis=0)

		self.reorder_nodes()
		self.node_coords = np.zeros((self.num_of_nodes + 1, 3))
		for ntag in range(1, self.num_of_nodes + 1):
			self.node_coords[ntag, :] = mesh.getNode(ntag)[0]

		self.evaluate_adjacents_and_jacobians()
		self.evaluate_neighbour_nodes()
		self.evaluate_integration_on_inner_surf()

	def __del__(self):
		gmsh.finalize()
