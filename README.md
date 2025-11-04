# Simple 3D Laplace equation solver using gmsh

## Requirements

* python3
* **python packages:** gmsh, numpy, scipy, tqdm, matplotlib, pathlib
* gmsh library https://gmsh.info/

## Usage

First, you need to generate mesh of the model you want to solve. Models are listed in `params.py`:

0. Sphere with Dirichlet boundary conditions (DBC) on round spot at the pole and Niemann boundary conditions (NBC) on the rest of the surface.
1. Ellipsoid with DBC on round spot at the pole and NBC on the rest of the surface.
2. Ellipsoid with DBC on round spot at the side and NBC on the rest of the surface.
3. Sphere with many uniformly distributed round spots with DBC and with NBC at the rest of the surface.
4. Ellipsoid with DBC overall the surface, but flux is calculated only for a round spot at the pole.
5. Ellipsoid with DBC overall the surface, but flux is calculated only for a round spot at the side.

To generate mesh, run

`python generate_mesh.py <model_number>`

The program will generate the mesh, save it to `data/`subdirectory and show it in the window.

Then, to solve the model run

`python solver.py <model_number>`

It will solve the model on the generated mesh for a range of the spot sizes, save the solution to `data/`and calculate flux through the spot. Size of the range is governed by `step_num` variable in `params.py`. The flux vs spot size is saved into .npz file.

To view the obtained solution, run

`python visualize.py <model_number>`

You can switch between different spot sizes using arrows.

To view calculated fluxes vs spot size, you can run

`python show_fluxes.py <model_number>`