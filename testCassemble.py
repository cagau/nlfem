
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from plot import plot_mesh_DG, plot_mesh_CG


def test_retriangulate():
    from assemble import py_retriangulate
    from conf import py_P

    def plot_retriangulate(x_center, delta, TE, RD, Rdx, pp):
        ax = plt.gca()
        plt.gca().set_aspect('equal')
        plt.fill(TE[:, 0], TE[:, 1], edgecolor='r', fill=False, alpha=.5)
        plt.scatter(x_center[0], x_center[1], s=1, c="red", alpha=.5)
        circ = plt.Circle(x_center, delta, fill=False, color="b", lw=.1, alpha=.7)
        ax.add_artist(circ)

        if RD.size > 0:
            plt.scatter(RD[:, 0], RD[:, 1], s=5, color="b", alpha=1)
        for k in range(Rdx):
            plt.fill(RD[(3*k):(3*k+3), 0], RD[(3*k):(3*k+3), 1], edgecolor='b', fill=False, alpha=.3)
        plt.savefig(pp, format='pdf')
        plt.close()

    deltaList = [.1, .1, .85, .3, .3, .3, .9, .6, .4, .3]
    x_centerList = [np.array([-.2,.6]),
               np.array([.2,.6]),
               np.array([.3,.6]),
               np.array([.6,.4]),
               np.array([.1,.1]),
               np.array([.6,.8]),
               np.array([.8,.3]),
               np.array([.3,.7]),
               np.array([.3,.7]),
               np.array([.25,.6])]

    TE = np.array([[0,0],[1,1],[0,1]], dtype=float)
    c_TE = TE.flatten("C")
    print("Triangle: ", c_TE)

    pp = PdfPages("testCassemble_Retriangulate" + ".pdf")
    for i in range(len(deltaList)):
        print("Page ", i + 1)
        delta  = deltaList[i]
        x_center = x_centerList[i]
        # retriangulate(double * x_center, double * TE, double sqdelta, double * out_reTriangle_list, int is_placePointOnCap)
        Rdx, RD = py_retriangulate(x_center, c_TE, delta, 1, pp)
        print("Rdx ", Rdx,"\n", RD)
        RD = RD.reshape((-1,2))
        plot_retriangulate(x_center, delta, TE, np.array(RD[:3*Rdx]), Rdx, pp)
    pp.close()

def test_interfacedependendKernel():
    from conf import py_Px, py_Py, dx, dy, delta, ansatz, mesh_name
    from nlocal import Mesh
    from assemble import assemble
    import pickle as pkl
    import scipy.sparse as ss
    import sys
    # insert at 1, 0 is the script path (or '' in REPL)
    sys.path.insert(1, '../nonlocal-assembly-chris')

    mesh = Mesh(mesh_name+ ".msh", ansatz)
    #T = np.load("triangles.npy")
    #V =  np.load("verts.npy")

    #mesh.triangles = np.load("../compare_data/Triangles_Chris.npy")
    #mesh.vertices = np.load("../compare_data/Verts_Chris.npy")

    mesh_chris = pkl.load(open( "../compare_data/mesh.pkl", "rb" ))
    mesh.triangles = mesh_chris.triangles
    mesh.vertices =  mesh_chris.verts

    Ad, fd  = assemble(mesh, py_Px, py_Py, dx, dy, delta)

    Ad_O = np.array(Ad[:, :mesh.K_Omega])
    ud = np.linalg.solve(Ad_O, fd)

    fd_Ext = np.zeros(mesh.K)
    fd_Ext[:mesh.K_Omega] = fd
    ud_Ext = np.zeros(mesh.K)
    ud_Ext[:mesh.K_Omega] = ud

    np.save("Ad_O", Ad_O)
    if mesh.ansatz == "CG":
        plot_mesh_CG(mesh, delta, "testCassemble_interfaceKernel", ud=ud_Ext, fd=fd_Ext, Ad_O=Ad_O)
    if mesh.ansatz == "DG":
        plot_mesh_DG(mesh, delta, "testCassemble_interfaceKernel", ud=ud_Ext, fd=fd_Ext, Ad_O = Ad_O, maxTriangles=None)

    if True:
        my_Ad_O = Ad_O
        loader = np.load("../compare_data/A_L2_h0.025_delta0.05_ballexact_L2_.npz")
        A = ss.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])
        chris_Ad_O = np.array(A.todense()[:mesh.K_Omega, :mesh.K_Omega])

        #chris_Ad_O = np.load("../compare_data/A_Chris.npy")
        diff_norm = np.linalg.norm(chris_Ad_O - my_Ad_O)
        print("L2 Norm Difference:\t", diff_norm)
        Ad_diff = my_Ad_O - chris_Ad_O
        minim = np.min(Ad_diff)
        maxim = np.max(Ad_diff)
    print("Stop")

if __name__ == "__main__":
    test_interfacedependendKernel()
    #test_retriangulate()