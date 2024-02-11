import sys
import gmsh
import numpy as np

from params import *

model_num = int(sys.argv[1])
gmsh.initialize(sys.argv)

gmsh.open(model_names[model_num] + '-solution.msh')
# Solution
solution = max(gmsh.view.getTags())

# Cut with plane z = 0
gmsh.plugin.setNumber('CutPlane', 'A', 0)
gmsh.plugin.setNumber('CutPlane', 'B', 0)
gmsh.plugin.setNumber('CutPlane', 'C', 1)
gmsh.plugin.setNumber('CutPlane', 'D', 0)
gmsh.plugin.run('CutPlane')
plane_cut = max(gmsh.view.getTags())

# Cut with sphere
if model_num == 0:
	gmsh.plugin.setNumber('CutSphere', 'R', 1.1*R)
	gmsh.plugin.setNumber('CutSphere', 'View', 0)
	gmsh.plugin.run('CutSphere')
	sphere_cut = max(gmsh.view.getTags())

"""
# Error
error = gmsh.view.add('error')
_, tags, sol, _, _ = gmsh.view.getHomogeneousModelData(solution, 0)
r = np.array([np.linalg.norm(gmsh.model.mesh.getNode(t)[0]) for t in tags])
gmsh.view.addHomogeneousModelData(error, 0, model_name + '-solution', 'NodeData', tags, sol - (1 - R/r)/(1 - R/L))

# Cut of error with plane z = 0
gmsh.plugin.setNumber('CutPlane', 'A', 0)
gmsh.plugin.setNumber('CutPlane', 'B', 0)
gmsh.plugin.setNumber('CutPlane', 'C', 1)
gmsh.plugin.setNumber('CutPlane', 'D', 0)
gmsh.plugin.run('CutPlane')
error_plane_cut = max(gmsh.view.getTags())
"""

gmsh.option.setNumber('View[%i].Visible' % gmsh.view.getIndex(solution), 0)
#gmsh.option.setNumber('View[%i].Visible' % gmsh.view.getIndex(error), 0)

gmsh.option.setNumber('View.VectorType', 4)
gmsh.option.setNumber('Mesh.Tetrahedra', 0)
gmsh.option.setNumber('Mesh.Triangles', 0)
gmsh.option.setNumber('Mesh.Lines', 0)

gmsh.fltk.run()
gmsh.finalize()
