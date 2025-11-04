model_names = [
    'sphere_with_spot',
    'ellipsoid_with_cluster_on_pole',
    'ellipsoid_with_lateral_cluster',
    'sphere_with_uniform_spots',
    'full_ellipsoid_polar',
    'full_ellipsoid_lateral',
]

R = 1  # ephere radius
A = 1  # ellipsoid major semiaxis
B = 0.5  # ellipsoid minor semiaxis
C = (A**2 - B**2)**0.5  # ellipsoid "radius"
L = 4  # radius of the outer spheric envelope
s = 0.1
mesh_micro_size = 0.02
mesh_middle_size = 0.1
mesh_macro_size = 2
element_order = 2
element_name = 'Tetrahedron'
element2D_name = 'Triangle'
steps_num = 50  # number of spot sizes to be solved
