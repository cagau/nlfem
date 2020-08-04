#!/home/klar/.venv/bin/python3
#-*- coding:utf-8 -*-

# In order to find the scripts in the python/ directory
# we add the current project path == working directory to sys-path

from scipy.linalg import pinvh, solve, pinv
import matplotlib.pyplot as plt
import time
from examples.DomainDecomposition.conf import *
from examples.DomainDecomposition.nlocal import setZeta, MeshfromDict, timestamp
from examples.DomainDecomposition.SubdivideMesh import mesh_data, submesh_data, submesh_data_2
from matplotlib.backends.backend_pdf import PdfPages

try:
    from assemble import assemble, solve_cg
except ImportError:
    print("\nCan't import assemble.\nTry: export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/libCassemble.\n")
    raise ImportError

def matrix_compare(M, A_list, u_list, f_list):
    """Compare the stiffnes matrices of subdivided and single domain problem by reassembling them.
    The function expects a list of meshes and data [main, sub1, sub2, ...].

    :param M: List of meshes
    :param A_list: List of dense Matrices (K_Omega, K)
    :param u_list: List of dense solutions (K_Omega,)
    :param f_list: List of RHS (K_Omega, )
    :return: -
    """

    main = M[0]
    B = np.zeros((main.K, main.K))
    fb = np.zeros(main.K)
    for k in range(len(M)-1):
        subm = M[k+1]
        C = A_list[k+1]
        for i in range(subm.K_Omega):
            pari = subm.embedding_vertices[i]
            fb[pari] += f_list[k+1][i]
            for j in range(subm.K):
                parj = subm.embedding_vertices[j]
                B[pari, parj] += C[i, j]

    B_O = B[:main.K_Omega, :main.K_Omega]
    fb = fb[:main.K_Omega]
    A_O = A_list[0][:main.K_Omega, :main.K_Omega]

    print("Solve...")
    solution = solve_cg(B_O, fb, fb)
    print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])

    # Write output to Paraview -----------------------------------------------------------------------------------------
    u = np.zeros(main.K)
    u[:main.K_Omega] = solution["x"]
    u_list.append(solution["x"].copy())
    main.point_data["ud_compare"] = u

    relDiff = np.linalg.norm(A_O - B_O)/np.linalg.norm(A_O)
    Diff = np.linalg.norm(A_O - B_O)
    print("Relative difference of matrices |A-B|/|A|\n", relDiff)
    print("difference of matrices |A-B|\n", Diff)
    return relDiff, Diff


def solve_KKT(M, A_list, u_list, f_list):
    if len(M) > 2:
        raise NotImplementedError("KKT Not implemented for 4 Domain System")

    import matplotlib.pyplot as plt
    main = M[0]
    sub = M[1:]
    # Setup KKT-System
    K_Omega = [mesh.K_Omega for mesh in M]
    A_O = []
    for k, k_om in enumerate(K_Omega):
        A_O.append(A_list[k][:k_om, :k_om])

    psi = []  # Embedding of vertices
    row_filter = np.ones()
    for k, sub in enumerate(sub):
        n = K_Omega[k]
        psi_ = np.zeros((main.nV, n))
        psi_[sub.embedding_vertices[:n], np.arange(n)] = 1
        psi_ = psi_[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex
        psi.append(psi_)

    row_filter = np.array((np.sum(psi1, axis=1) > 0)*(np.sum(psi2, axis=1) > 0), dtype=bool)
    P1 = psi1[row_filter, :]
    P2 = psi2[row_filter, :]
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
    if len(M) > 2:
        raise NotImplementedError("FETI Not implemented for 4 Domain System")

    main = M[0]
    sub1 = M[1]
    sub2 = M[2]

    f1 = f_list[1]
    f2 = f_list[2]

    A1 = A_list[1][:sub1.K_Omega, :sub1.K_Omega]
    n1 = sub1.K_Omega
    A2 = A_list[2][:sub2.K_Omega, :sub2.K_Omega]
    n2 = sub2.K_Omega

    psi1 = np.zeros((main.nV, n1))
    psi1[sub1.embedding_vertices[:n1], np.arange(n1)] = 1.
    psi1 = psi1[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    psi2 = np.zeros((main.nV, n2))
    psi2[sub2.embedding_vertices[:n2], np.arange(n2)] = 1.
    psi2 = psi2[:main.nV_Omega] # Due to points which change their status from Dirchlet to Neumann Vertex

    row_filter = np.array((np.sum(psi1, axis=1) > 0)*(np.sum(psi2, axis=1) > 0), dtype=bool)

    # Sub-blocks of M = [P1, -P2]
    P1 = psi1[row_filter, :]
    P2 = psi2[row_filter, :]

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
    elSizes = [0.07, 0.05, 0.04, 0.03, 0.02, 0.01]
    h = []
    assemblyTime = []
    relDiffMat = []
    diffMat = []

    for element_size in elSizes:
        elements, vertices, lines, elementLabels, subdomainLabels, K_Omega, diam, Gamma_hat, mesh_dict = \
            mesh_data(geofile, element_size, delta)
        if geofile[-1] == "4":
            submesh, submesh_dicts = submesh_data_2(elements, vertices, lines, subdomainLabels, diam, element_size, delta)
        else:
            submesh, submesh_dicts = submesh_data(elements, vertices, lines, subdomainLabels, diam, element_size, delta)
        h.append(diam)

        n_submesh = len(submesh)

        M = [MeshfromDict(mesh_dict)]
        M += [MeshfromDict(subm) for subm in submesh_dicts]
        M = setZeta(M)
        print("Delta: ", delta, "\t Mesh: ", mesh_name)

        A_list = []
        f_list = []
        u_list = []

        subAssemblyTimes = []
        for k, mesh in enumerate(M):
            print("Number of basis functions: ", mesh.K)
            # Assemble and solve -------------------------------------------------------------------------------------------
            start = time.time()
            A, f = assemble(mesh, py_Px, py_Py, dx, dy, delta,
                              path_spAd=None,
                              path_fd=None,
                              compute="systemforcing", # "forcing", "system"
                              model_kernel="constant",
                              model_f = "constant",
                              integration_method="retriangulate",
                              is_PlacePointOnCap=1)
            subAssemblyTimes.append(time.time() - start)
            A_list.append(A.todense())
            f_list.append(f.copy())

            A_O = A[:, :mesh.K_Omega]
            A_I = A[:, mesh.K_Omega:]

            # Solve --------------------------------------------------------------------------------------------------------
            print("Solve...")
            solution = solve_cg(A_O, f, f)
            print("CG Solve:\nIterations: ", solution["its"], "\tError: ", solution["res"])

            # Write output to Paraview -------------------------------------------------------------------------------------
            mesh.add_u(solution["x"], "separateSolve")
            u_list.append(solution["x"].copy())
        assemblyTime.append(subAssemblyTimes.copy())

        #solve_KKT(M, A_list, u_list, f_list)
        #FETI_floating(M, A_list, u_list, f_list)
        #FETI_invertible(M, A_list, u_list, f_list)
        relDiff, Diff = matrix_compare(M, A_list, u_list, f_list)
        relDiffMat.append(relDiff)
        diffMat.append(Diff)

        for k, mesh in enumerate(M):
            mesh.write(OUTPUT_PATH + mesh_name + str(k) + "_" + tmpstp + str(diam) + ".vtk")

    diffData = np.array([h, relDiffMat, diffMat])
    np.save(OUTPUT_PATH+tmpstp+"diffData.npy", diffData)
    timeData = np.array(assemblyTime)
    np.save(OUTPUT_PATH+tmpstp+"timeData.npy", timeData)

    pp = PdfPages(OUTPUT_PATH + "plot.pdf")

    plt.yscale("log")
    plt.xlabel("diameter h")
    plt.ylabel("difference")
    plt.plot(h, diffData[1, :])
    plt.scatter(h, diffData[1, :])
    plt.title("Relative difference of stiffnes matrix\n |A-B|/|A|")
    plt.savefig(pp, format='pdf')
    plt.close()

    plt.yscale("log")
    plt.ylabel("difference")
    plt.xlabel("diameter h")
    plt.plot(h, diffData[2, :])
    plt.scatter(h, diffData[2, :])
    plt.title("Difference of stiffnes matrix\n |A-B|")
    plt.savefig(pp, format='pdf')
    plt.close()

    plt.xlabel("diameter h")
    plt.ylabel("time [s]")
    assemblyTime = np.array(assemblyTime).T
    for k, times in enumerate(assemblyTime):
        if k == 0:
            plt.plot(h, times, label="single domain")
            plt.scatter(h, times)
        else:
            plt.plot(h, times)
            plt.scatter(h, times)
    plt.title("Assembly Time")
    plt.legend()
    plt.savefig(pp, format='pdf')
    plt.close()

    pp.close()




