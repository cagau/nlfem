#-*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from conf import mesh_name, delta
from aux import filename
import pickle as pkl

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
    plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.5, color='white', alpha=.5)
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
    plt.triplot(mesh.V[:, 0], mesh.V[:, 1], mesh.T, lw=0.5, color='white', alpha=.5)
    plt.tricontourf(mesh.V[:, 0], mesh.V[:, 1], mesh.T, ud_Ext)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title("Solution u")
    plt.savefig(pp, format='pdf')
    plt.close()

    fig = plt.figure()
    ax = fig.gca(projection='3d', title="Solution u")
    ax.plot_trisurf(mesh.V[:, 0], mesh.V[:, 1], ud_Ext)
    plt.savefig(pp, format='pdf')
    plt.close()

    plt.imshow(Ad_O)
    plt.colorbar(orientation='horizontal', shrink=.7)
    plt.title(r"$A_{\Omega\Omega}$")
    plt.savefig(pp, format='pdf')
    plt.close()

    pp.close()
