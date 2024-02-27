import numpy as np
import sys
import gmsh
from params import *

model_num = int(sys.argv[1])
gmsh.initialize(sys.argv)
gmsh.model.add(model_names[model_num])
cad = gmsh.model.occ

if model_num == 0:
	cad.addSphere(0, 0, 0, R, 1)
	cad.rotate(cad.getEntities(), 0, 0, 0, 0, 1, 0, np.pi/2)
else:
	cad.addSphere(0, 0, 0, 1, 1)
	cad.rotate(cad.getEntities(), 0, 0, 0, 0, 1, 0, np.pi/2)
	cad.dilate(cad.getEntities(), 0, 0, 0, A, B, B)

cad.addSphere(0, 0, 0, L, 2)
cad.cut([(3, 2)], [(3, 1)], 3, removeObject=False, removeTool=False)
cad.remove([(3, 2), (3, 1)])
cad.synchronize()

gmsh.model.addPhysicalGroup(3, [3], name='domain')
gmsh.model.addPhysicalGroup(2, [1], name='inner surface')
gmsh.model.addPhysicalGroup(2, [2], name='outer surface')


gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)

gmsh.model.mesh.setSize([(0, 1)], mesh_micro_size)
gmsh.model.mesh.setSize([(0, 2)], mesh_middle_size)
gmsh.model.mesh.setSize([(0, 3), (0, 4)], mesh_macro_size)

gmsh.model.mesh.generate(3)
gmsh.model.mesh.removeDuplicateNodes()
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()
gmsh.model.mesh.setOrder(element_order)
gmsh.write(model_names[model_num] + '.msh')
gmsh.fltk.run()
gmsh.finalize()
