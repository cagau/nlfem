#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import setCeta, MeshfromDict
from examples.DomainDecomposition.SubdivideMesh import mesh_data, submesh_data
from scipy.linalg import pinvh, solve, pinv

try:
    from assemble import assemble, solve_cg
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

def quick_compare():
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

def solve_KKT(M, A_list, u_list, f_list):
    import matplotlib.pyplot as plt
    main = M[0]
    sub1 = M[1]
    sub2 = M[2]

    # Setup KKT-System

    A1 = A_list[1][:sub1.K_Omega, :sub1.K_Omega]
    n1 = sub1.K_Omega
    A2 = A_list[2][:sub2.K_Omega, :sub2.K_Omega]
    n2 = sub2.K_Omega

    embed1 = np.zeros((main.nV, n1))
    embed1[sub1.embedding_vertices[:n1], np.arange(n1)] = 1.
    embed1 = embed1[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    embed2 = np.zeros((main.nV, n2))
    embed2[sub2.embedding_vertices[:n2], np.arange(n2)] = 1.
    embed2 = embed2[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    row_filter = np.array((np.sum(embed1, axis=1) > 0)*(np.sum(embed2, axis=1) > 0), dtype=bool)
    P1 = embed1[row_filter, :]
    P2 = embed2[row_filter, :]
    n3 = np.sum(row_filter)

    nkkt = n1 + n2 + n3
    KKT = np.zeros((nkkt, nkkt))
    KKT[:n1, :n1] = A1
    KKT[n1:n2+n1, n1:n2+n1] = A2
    KKT[:n1, -n3:] = P1.T
    KKT[n1:n2+n1, -n3:] = -P2.T
    KKT[-n3:, :n1] = P1
    KKT[-n3:, n1:n2+n1] = -P2

    rhs = np.concatenate((f_list[1], f_list[2], np.zeros(n3)))
    x0 = np.concatenate((u_list[1], u_list[2], np.zeros(n3)))
    x = np.linalg.solve(KKT, rhs)
    #x = sol["x"]

    u1 = x[:n1]
    u2 = x[n1:n1+n2]

    u = np.zeros(sub1.K)
    u[:sub1.K_Omega] = x[:n1]
    sub1.point_data["ud_kkt"] = u

    u = np.zeros(sub2.K)
    u[:sub2.K_Omega] = x[n1:n2+n1]
    sub2.point_data["ud_kkt"] = u
    sub1.write(OUTPUT_PATH + mesh_name + str(1) + ".vtk")
    sub2.write(OUTPUT_PATH + mesh_name + str(2) + ".vtk")

def FETI_invertible(M, A_list, u_list, f_list):
    main = M[0]
    sub1 = M[1]
    sub2 = M[2]

    f1 = f_list[1]
    f2 = f_list[2]

    A1 = A_list[1][:sub1.K_Omega, :sub1.K_Omega]
    n1 = sub1.K_Omega
    A2 = A_list[2][:sub2.K_Omega, :sub2.K_Omega]
    n2 = sub2.K_Omega

    embed1 = np.zeros((main.nV, n1))
    embed1[sub1.embedding_vertices[:n1], np.arange(n1)] = 1.
    embed1 = embed1[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    embed2 = np.zeros((main.nV, n2))
    embed2[sub2.embedding_vertices[:n2], np.arange(n2)] = 1.
    embed2 = embed2[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    row_filter = np.array((np.sum(embed1, axis=1) > 0)*(np.sum(embed2, axis=1) > 0), dtype=bool)

    # Sub-blocks of M = [P1, -P2]
    P1 = embed1[row_filter, :]
    P2 = embed2[row_filter, :]

    pinvA1 = pinvh(A1)
    pinvA2 = pinvh(A2)

    K1 = P1 @ pinvA1 @ P1.T
    K2 = P2 @ pinvA2 @ P2.T # Actually -P2, but that makes no difference here.
    K = K1 + K2  # Coupling!

    # 1. Compute e, g = 0 -----------------------------
    e1 = (P1 @ pinvA1) @ f1
    e2 = -(P2 @ pinvA2) @ f2
    e = e1 + e2  # Coupling!

    # Due to the invertibility of the blocks in A we can find lambda in one step
    lambda0 = np.linalg.solve(K, e)

    # Deduce u1 and u2
    u1 = pinvA1 @ (f1 - P1.T @ lambda0)
    u2 = pinvA2 @ (f2 + P2.T @ lambda0)

    u = np.zeros(sub1.K)
    u[:sub1.K_Omega] = u1
    sub1.point_data["ud_feti"] = u
    u = np.zeros(sub2.K)
    u[:sub2.K_Omega] = u2
    sub2.point_data["ud_feti"] = u

    sub1.write(OUTPUT_PATH + mesh_name + str(1) + ".vtk")
    sub2.write(OUTPUT_PATH + mesh_name + str(2) + ".vtk")

def FETI_floating(M, A_list, u_list, f_list):
        main = M[0]
    sub1 = M[1]
    sub2 = M[2]

    f1 = f_list[1]
    f2 = f_list[2]

    A1 = A_list[1][:sub1.K_Omega, :sub1.K_Omega]
    n1 = sub1.K_Omega
    A2 = A_list[2][:sub2.K_Omega, :sub2.K_Omega]
    n2 = sub2.K_Omega

    embed1 = np.zeros((main.nV, n1))
    embed1[sub1.embedding_vertices[:n1], np.arange(n1)] = 1.
    embed1 = embed1[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    embed2 = np.zeros((main.nV, n2))
    embed2[sub2.embedding_vertices[:n2], np.arange(n2)] = 1.
    embed2 = embed2[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    row_filter = np.array((np.sum(embed1, axis=1) > 0)*(np.sum(embed2, axis=1) > 0), dtype=bool)

    # Sub-blocks of M = [P1, -P2]
    P1 = embed1[row_filter, :]
    P2 = embed2[row_filter, :]

    pinvA1 = pinvh(A1)
    pinvA2 = pinvh(A2)

    K1 = P1 @ pinvA1 @ P1.T
    K2 = P2 @ pinvA2 @ P2.T # Actually -P2, but that makes no difference here.
    K = K1 + K2  # Coupling!

    # 1. Compute e, g = 0 -----------------------------
    e1 = (P1 @ pinvA1) @ f1
    e2 = -(P2 @ pinvA2) @ f2
    e = e1 + e2  # Coupling!

    # Due to the invertibility of the blocks in A we can find lambda in one step
    lambda0 = np.linalg.solve(K, e)

    # Deduce u1 and u2
    u1 = pinvA1 @ (f1 - P1.T @ lambda0)
    u2 = pinvA2 @ (f2 + P2.T @ lambda0)

    u = np.zeros(sub1.K)
    u[:sub1.K_Omega] = u1
    sub1.point_data["ud_feti"] = u
    u = np.zeros(sub2.K)
    u[:sub2.K_Omega] = u2
    sub2.point_data["ud_feti"] = u

    sub1.write(OUTPUT_PATH + mesh_name + str(1) + ".vtk")
    sub2.write(OUTPUT_PATH + mesh_name + str(2) + ".vtk")


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
        A_list.append(A.todense())
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
        u_list.append(solution["x"].copy())
        mesh.point_data["ud"] = u
        #mesh.write(OUTPUT_PATH + mesh_name + str(k) + ".vtk")
    #solve_KKT(M, A_list, u_list, f_list)
    FETI_invertible(M, A_list, u_list, f_list)
