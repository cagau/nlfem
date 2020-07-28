#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

import numpy as np
import meshio
from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import MeshIO

try:
    from assemble import assemble
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

# Necessary definitions for intersection -------------------------------------------------------------------------------
if __name__ == "__main__":
    confDict = {"domainPhysicalName": 1, "boundaryPhysicalName": 9, "interactiondomainPhysicalName": 2,
                "boundaryConditionType": boundaryConditionType, "ansatz": ansatz}
    # Mesh construction --------------------
    mesh = MeshIO(DATA_PATH + mesh_name, **confDict)

    print("Delta: ", delta, "\t Mesh: ", mesh_name)
    print("Number of basis functions: ", mesh.K)

    Ad, fd = assemble(mesh, py_Px, py_Py, dx, dy, delta,
                      path_spAd=None,
                      path_fd=None,
                      compute="systemforcing", # "forcing", "system"
                      model_kernel="constant",
                      model_f = "constant",
                      integration_method="retriangulate",
                      is_PlacePointOnCap=1)


