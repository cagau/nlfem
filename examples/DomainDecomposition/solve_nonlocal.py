#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import setCeta, MeshfromDict
from examples.DomainDecomposition.SubdivideMesh import mesh_data, submesh_data
from scipy.sparse import coo_matrix
try:
    from assemble import assemble, solve_cg
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

if __name__ == "__main__":
    # Mesh construction ------------------------------------------------------------------------------------------------
    elements, vertices, lines, elementLabels, subdomainLabels, K_Omega, diam, Gamma_hat, mesh_dict = \
        mesh_data(geofile, element_size, delta)
    submesh, submesh_dicts = submesh_data(elements, vertices, lines, subdomainLabels, diam)
    n_submesh = len(submesh)

    M = [MeshfromDict(mesh_dict)]
    M += [MeshfromDict(subm) for subm in submesh_dicts]
    M = setCeta(M)

    print("Delta: ", delta, "\t Mesh: ", mesh_name)

    A_list = []
    f_list = []
    u_list = []

    for k, mesh in enumerate(M):
        print("Number of basis functions: ", mesh.K)
        # Assemble and solve -----------------------------------------------------------------------------------------------
        A, f = assemble(mesh, py_Px, py_Py, dx, dy, delta,
                          path_spAd=None,
                          path_fd=None,
                          compute="systemforcing", # "forcing", "system"
                          model_kernel="constant",
                          model_f = "constant",
                          integration_method="retriangulate",
                          is_PlacePointOnCap=1)
        A_list.append(coo_matrix(A))
        f_list.append(f.copy())

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
        u_list.append(u.copy())
        mesh.point_data["ud"] = u
        mesh.write(OUTPUT_PATH + mesh_name + str(k) + ".vtk")

    # Compare ...
    main_mesh = M[0]
    B = np.zeros((main_mesh.K, main_mesh.K))

    for k in range(2):
        subm = M[k+1]
        A = A_list[k+1].todense()
        for i in range(subm.K_Omega):
            for j in range(subm.K):
                pari = subm.embedding_vertices[i]
                parj = subm.embedding_vertices[j]
                B[pari, parj] += A[i, j]


    B_O = B[:main_mesh.K_Omega, :main_mesh.K_Omega]
    B_I = B[:main_mesh.K_Omega, main_mesh.K_Omega:]
    fb = f_list[0] ### CETA not yet implemented on right side!!!
    g = np.apply_along_axis(eval_g, 1, main_mesh.vertices[main_mesh.K_Omega:])
    #f -= B_I@g ### CETA not yet implemented on right side!!!
    print("Solve...")
    solution = solve_cg(B_O, fb, fb)
    print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])
    # Write output to Paraview -----------------------------------------------------------------------------------------
    u = np.zeros(main_mesh.K)
    u[:main_mesh.K_Omega] = solution["x"]
    u_list.append(u.copy())
    main_mesh.point_data["ud"] = u
    main_mesh.write(OUTPUT_PATH + mesh_name + "compare" + ".vtk")

    print("Difference", np.linalg.norm(u_list[0] - u_list[-1]))