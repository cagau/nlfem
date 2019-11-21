#-*- coding:utf-8 -*-
import time
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from python.aux import filename
import pickle as pkl

def uLocaSol(x,y):
    n = x.shape[0]
    u =  1/4*(1 - x**2 - y**2)
    u = np.maximum(u, np.zeros(n))
    return u

## Lädt die Mesh-Daten aus einem gespeicherten File und plottet.
def plot_mesh_CG(mesh, delta, fnm, **kwargs):
    """
    Plot function. Plots the solution and the right side and saves them to a pdf.

    :param mesh_name: Name of mesh as string e.g. "medium"
    :param delta: Interaction radius (for Filename)
    :param Tstmp: Timestamp (for Filename)
    :return: None
        """
    pp = PdfPages(fnm + ".pdf")

    fd = kwargs.get("fd", None)
    if fd is not None:
        plt.gca().set_aspect('equal')
        grid = np.arange(-1-delta, 1+delta, delta)
        plt.yticks(grid)
        plt.xticks(grid)
        plt.grid(True, color='white', lw=0.1, alpha=.6)

        plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles[:,1:], lw=0.1, color='white', alpha=.3)
        plt.tricontourf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles[:,1:], fd)
        plt.colorbar(orientation='horizontal', shrink=.7)

        #plt.scatter(mesh.vertices[:mesh.nV_Omega, 0], mesh.vertices[:mesh.nV_Omega, 1], c="b")
        plt.title("Right side compute_f")
        plt.savefig(pp, format='pdf')
        plt.close()

    ud = kwargs.get("ud", None)
    if ud is not None:
        plt.gca().set_aspect('equal')
        grid = np.arange(-1-delta, 1+delta, delta)
        plt.yticks(grid)
        plt.xticks(grid)
        plt.grid(True, color='white', lw=0.1, alpha=.6)

        plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles[:,1:], lw=0.1, color='white', alpha=.3)
        plt.tricontourf(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles[:,1:], ud)
        plt.colorbar(orientation='horizontal', shrink=.7)
        plt.title("Solution u")
        plt.savefig(pp, format='pdf')
        plt.close()

        fig = plt.figure()
        ax = fig.gca(projection='3d', title="Solution u")
        # True Solution
        #u_true = uLocaSol(mesh.vertices[:, 0], mesh.vertices[:, 1])
        #ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], u_true, alpha=.4)
        # Computed Solution
        ax.plot_trisurf(mesh.vertices[:, 0], mesh.vertices[:, 1], ud, alpha=.8, color='red')
        plt.savefig(pp, format='pdf')
        plt.close()

    Ad_O = kwargs.get("Ad_O", None)
    if Ad_O is not None:
        plt.imshow(Ad_O)
        plt.colorbar(orientation='horizontal', shrink=.7)
        plt.title(r"$A$")
        plt.savefig(pp, format='pdf')
        plt.close()

    sol_logs = kwargs.get("sol_logs", None)
    if sol_logs is not None:
        for label, sol in sol_logs.items():
            plt.plot(np.log10(np.abs(sol["log"])), label=label)
        plt.legend()
        plt.title("CG-Iteration Residuals")
        plt.ylabel("log10 residual")
        plt.xlabel("Iterations")
        plt.savefig(pp, format='pdf')
        plt.close()

        for label, sol in sol_logs.items():
            plt.plot(sol["time"], np.log10(np.abs(sol["log"])), label=label)
        plt.legend()
        plt.title("CG-Iteration Residuals")
        plt.ylabel("log10 residual")
        plt.xlabel("Time")
        plt.savefig(pp, format='pdf')
        plt.close()
    pp.close()

def plot_mesh_DG(mesh, delta, fnm, **kwargs):

    maxTriangles = kwargs.get("maxTriangles", None)
    if maxTriangles is None:
        maxTriangles = mesh.nE_Omega

    pp = PdfPages(fnm + ".pdf")

    plt.gca().set_aspect('equal')
    grid = np.arange(-1-delta, 1+delta, delta)
    plt.yticks(grid)
    plt.xticks(grid)
    plt.grid(True, color='white', lw=0.1, alpha=.6)


    plt.triplot(mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.triangles[:, 1:], lw=0.1, color='white', alpha=.3)

    start = time.time()

    ud = kwargs.get("ud", None)
    if not ud is None:
        minval = np.min(ud)
        maxval = np.max(ud)

        for k in range(maxTriangles):
            print("Iterate ", k, " ", ud[3*k:(3*k+3)],end="\r", flush=True)
            Vdx = mesh.triangles[k,1:]
            plt.tricontourf(mesh.vertices[Vdx, 0], mesh.vertices[Vdx, 1], ud[3*k:(3*k+3)],  5, cmap=plt.cm.get_cmap('rainbow'), vmin=minval, vmax=maxval)
        ax, _ = matplotlib.colorbar.make_axes(plt.gca(), shrink=.7)
        matplotlib.colorbar.ColorbarBase(ax, cmap=plt.cm.get_cmap('rainbow'),
                                         norm=matplotlib.colors.Normalize(vmin=minval, vmax=maxval))

        plt.title("Solution u")
        plt.savefig(pp, format='pdf')
        plt.close()

    Ad_O = kwargs.get("Ad_O", None)
    if not Ad_O is None:
        plt.imshow(Ad_O)
        plt.savefig(pp, format='pdf')
        plt.close()

    print()
    print("Time Needed: ", time.time()-start)
    pp.close()

def plot(OUTPUT_PATH, mesh_name, delta, Tstmp= "", **kwargs):
    """
    Plot function. Plots the solution and the right side and saves them to a pdf.

    :param mesh_name: Name of mesh as string e.g. "medium"
    :param delta: Interaction radius (for Filename)
    :param Tstmp: Timestamp (for Filename)
    :return: None
        """

    fnm = Tstmp + filename(OUTPUT_PATH + mesh_name, delta)[1]

    fileObject2 = open(fnm, 'rb')
    datakwargs = pkl.load(fileObject2)
    fileObject2.close()
    try:
        datakwargs["Ad_O"] = np.load(OUTPUT_PATH + "Ad_O.npy")
    except IOError:
        print("In plot.plot(): No Matrix Ad_O found")
        datakwargs["Ad_O"] = None

    mesh = datakwargs.pop("mesh")
    if mesh.ansatz == "CG":
        plot_mesh_CG(mesh, delta, fnm, **datakwargs, **kwargs)
    elif mesh.ansatz == "DG":
        plot_mesh_DG(mesh, delta, fnm, **datakwargs, **kwargs)


if __name__ == "__main__":
    plot("circle_large", .1, "", maxTriangles=100)