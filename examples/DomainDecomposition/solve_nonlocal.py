#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

import numpy as np
import meshio
from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import MeshIO

try:
    from assemble import assemble, solve_cg
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    # Mesh construction --------------------
    mesh = MeshIO(DATA_PATH + mesh_name, boundaryConditionType=boundaryConditionType, ansatz=ansatz)

    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

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

    # Solve ---------------------------------------------------------------------------
    print("Solve...")
    #mesh.write_ud(np.linalg.solve(A_O, f), conf.u_exact)
    solution = solve_cg(A_O, f, f)
    print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
