#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from conf import mesh_name, delta
from aux import filename
import pickle as pkl

def uLocaSol(x,y):
    n = x.shape[0]
    u =  1/4*(1 - x**2 - y**2)
    u = np.maximum(u, np.zeros(n))
    return u

def plot_interpoaltion(mesh_list, u_list, delta, file_name="interpolation_plot", Tstmp= ""):
    fnm = Tstmp + filename(file_name)[1]

    pp = PdfPages(fnm + ".pdf")

    for k, mesh in enumerate(mesh_list):
        plt.gca().set_aspect('equal')
        grid = np.arange(-1 - delta, 1 + delta, delta)
        plt.yticks(grid)
        plt.xticks(grid)
        plt.grid(True, color='white', lw=0.1, alpha=.6)

        plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.1, color='white', alpha=.3)
        plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, u_list[k])
        plt.colorbar(orientation='horizontal', shrink=.7)
        plt.title("Solution u"+str(k+1))
        plt.savefig(pp, format='pdf')
        plt.close()
    pp.close()

def plot_convRate(data, delta, file_name="convergence_rate", Tstmp= ""):
    fnm = Tstmp + filename(file_name, delta)[1]
    pp = PdfPages(fnm + ".pdf")

    plt.plot(data["log2 relerror"])
    plt.scatter(np.arange(len(data["name"])-2), data["log2 relerror"])
    plt.savefig(pp, format='pdf')
    plt.close()
    pp.close()

def plot_sol(mesh, ud_Ext, delta, Tstmp= "", mesh_name="plot_solution", sol_logs=[]):
    fnm = Tstmp + filename(mesh_name, delta)[1]
    pp = PdfPages(fnm + ".pdf")

    plt.gca().set_aspect('equal')
    grid = np.arange(-1-delta, 1+delta, delta)
    plt.yticks(grid)
    plt.xticks(grid)
    plt.grid(True, color='white', lw=0.1, alpha=.6)

    plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.1, color='white', alpha=.3)
    plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, ud_Ext)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title("Solution u")
    plt.savefig(pp, format='pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d', title="Solution u")
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], ud_Ext, alpha=.8, color='red')
    plt.savefig(pp, format='pdf')
    plt.close()

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
def plot(mesh_name, delta, Tstmp= ""):
    """
    Plot function. Plots the solution and the right side and saves them to a pdf.

    :param mesh_name: Name of mesh as string e.g. "medium"
    :param delta: Interaction radius (for Filename)
    :param Tstmp: Timestamp (for Filename)
    :return: None
        """
    fnm = Tstmp + filename(mesh_name, delta)[1]

    fileObject2 = open(fnm, 'rb')
    arg = pkl.load(fileObject2)
    fileObject2.close()
    Ad_O = np.load("Ad_O.npy")
    ud_Ext = arg["ud_Ext"]
    fd_Ext = arg["fd_Ext"]
    mesh = arg["mesh"]

    pp = PdfPages(fnm + ".pdf")

    plt.gca().set_aspect('equal')
    grid = np.arange(-1-delta, 1+delta, delta)
    plt.yticks(grid)
    plt.xticks(grid)
    plt.grid(True, color='white', lw=0.1, alpha=.6)

    plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.1, color='white', alpha=.3)
    plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, fd_Ext)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title("Right side compute_f")
    plt.savefig(pp, format='pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d', title="Right side compute_f")
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], fd_Ext)
    plt.savefig(pp, format='pdf')
    plt.close()

    plt.gca().set_aspect('equal')
    grid = np.arange(-1-delta, 1+delta, delta)
    plt.yticks(grid)
    plt.xticks(grid)
    plt.grid(True, color='white', lw=0.1, alpha=.6)

    plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.1, color='white', alpha=.3)
    plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, ud_Ext)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title("Solution u")
    plt.savefig(pp, format='pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d', title="Solution u")
    # True Solution
    #u_true = uLocaSol(mesh.V[:, 0], mesh.V[:, 1])
    #ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], u_true, alpha=.4)
    # Computed Solution
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], ud_Ext, alpha=.8, color='red')
    plt.savefig(pp, format='pdf')
    plt.close()


    plt.imshow(Ad_O)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title(r"$A_{\Omega\Omega}$")
    plt.savefig(pp, format='pdf')
    plt.close()
    pp.close()

if __name__ == "__main__":
    plot("output/indep_x/large", .3, "")