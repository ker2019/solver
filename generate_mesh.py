import numpy as np
import sys
import gmsh
from params import *

gmsh.initialize(sys.argv)
gmsh.model.add(model_name)
cad = gmsh.model.occ

cad.addSphere(0, 0, 0, R, 1)
cad.addSphere(0, 0, 0, L, 2)
cad.cut([(3, 2)], [(3, 1)], 3, removeObject=False, removeTool=False)
cad.remove([(3, 2), (3, 1)])
cad.synchronize()

gmsh.model.addPhysicalGroup(3, [3], name='domain')
gmsh.model.addPhysicalGroup(2, [1], name='inner surface')
gmsh.model.addPhysicalGroup(2, [2], name='outer surface')


gmsh.option.setNumber('Mesh.SecondOrderLinear', 1)
gmsh.model.mesh.setSizeCallback(lambda dim, tag, x, y, z, lc: np.clip(0.1*(x**2 + y**2 + z**2), 0, 1))
gmsh.model.mesh.generate(3)
gmsh.model.mesh.removeDuplicateNodes()
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()
gmsh.model.mesh.setOrder(element_order)
gmsh.write(model_name + '.msh')
gmsh.fltk.run()
gmsh.finalize()
