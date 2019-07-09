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

    Ad_O = arg["Ad_O"]
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
    plt.title("Right side f")
    plt.savefig(pp, format='pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d', title="Right side f")
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
    u_true = uLocaSol(mesh.V[:, 0], mesh.V[:, 1])
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], u_true, alpha=.2)
    # Computed Solution
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], ud_Ext, alpha=.9)
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