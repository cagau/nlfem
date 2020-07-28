#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import MeshIO, MeshfromDict
from examples.DomainDecomposition.SubdivideMesh import mesh_data, submesh_data

try:
    from assemble import assemble, solve_cg
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

if __name__ == "__main__":
    # Mesh construction ------------------------------------------------------------------------------------------------
    #mesh = MeshIO(DATA_PATH + mesh_name, boundaryConditionType=boundaryConditionType, ansatz=ansatz)
    elements, vertices, lines, elementLabels, subdomainLabels, K_Omega, diam, Gamma_hat, mesh_dict = mesh_data(geofile, element_size, delta)
    mesh = MeshfromDict(mesh_dict)
    submesh, submesh_dicts = submesh_data(elements, vertices, lines, subdomainLabels, diam)
    mesh_child = [MeshfromDict(mdict) for mdict in submesh_dicts]
    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    mesh = mesh_child[0]
    mesh.write(OUTPUT_PATH + mesh_name + ".vtk")

    # Assemble and solve -----------------------------------------------------------------------------------------------
    A, f = assemble(mesh, py_Px, py_Py, dx, dy, delta,
                      path_spAd=None,
                      path_fd=None,
                      compute="systemforcing", # "forcing", "system"
                      model_kernel="constant",
                      model_f = "constant",
                      integration_method="retriangulate",
                      is_PlacePointOnCap=1)
    A_O = A[:, :mesh.K_Omega]
    A_I = A[:, mesh.K_Omega:]

    g = np.apply_along_axis(eval_g, 1, mesh.vertices[mesh.K_Omega:])
    f -= A_I@g

    # Solve ------------------------------------------------------------------------------------------------------------
    print("Solve...")
    solution = solve_cg(A_O, f, f)
    print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])

    # Write output to Paraview -----------------------------------------------------------------------------------------
    u = np.zeros(mesh.K)
    u[:mesh.K_Omega] = solution["x"]
    mesh.point_data["ud"] = u
    mesh.write(OUTPUT_PATH + mesh_name + ".vtk")
