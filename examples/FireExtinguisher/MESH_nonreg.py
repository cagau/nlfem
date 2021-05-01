import scipy.sparse as ss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
import math
import scipy.optimize as sc
#from pathos.multiprocessing import ProcessingPool as Pool
import scipy.interpolate as si
import random
import scipy.sparse.linalg as ssl
from scipy.integrate import quad#quadrature as
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import os
import bib3 as bib
import meshzoo



def prepare_mesh_gmsh(n, geofile = "unit_square_2"):
    h = 12/n/10
    os.system('gmsh mesh/' + geofile + '.geo -v 0 -2 -format msh2 -clscale ' + str(h) + ' -o mesh/' + geofile + '.msh')
    verts, lines, triangles = bib.read_mesh('mesh/' + geofile + '.msh')
    def diam(T):
        length_of_edges = np.array(
            [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
        return np.max(length_of_edges)
    diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in
                range(len(triangles))]
    h_min = np.min(diameter)
    h_max = np.max(diameter)
    return verts[:,0:2], triangles[:,1:], h_min, h_max

def prepare_mesh_nonreg(n,delta=0.1):

    def transform(x):
        #a = 3
        #b = 3
        #y1 = (1. - np.exp(-a * x[0])) / (1. - np.exp(-a))
        #y2 = np.sin(np.pi * x[1] / 2) ** b
        # return np.array([y1, y2])
        return np.array([x[0], x[1] ** 2])

    #n1, n2 = int(np.ceil((1. + 2. * delta) / h1)) + 1, int(np.ceil((1. + 2. * delta) / h2)) + 1  # int(1./h)+1
    n1=n
    n2=n
    x, ret = np.linspace(-delta, 1. + delta, n1, endpoint=True, retstep=1)
    y = np.linspace(-delta, 1. + delta, n2, endpoint=True)
    # xv, yv = np.meshgrid(x, y, indexing='ij')

    verts = np.around(np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2), decimals=12)

    omega = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5)
    boundary = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) == 0.5)
    omega_i = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) > 0.5)

    verts = np.concatenate((verts[omega], verts[boundary], verts[omega_i]))

    omega = list(range(len(verts[omega])))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)
    boundary = list(range(len(verts[omega]), len(verts[omega]) + len(verts[boundary])))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) == 0.5)
    omega_i = list(range(len(verts[omega]) + len(verts[boundary]),len(verts)))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) > 0.5)
    nodes = omega+boundary

    # TRANSFORM
    index_to_transform = omega# np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5 - max(h1, h2))[0].tolist()
    if 1:
        """
        problem: index_to_transform inner-inner points --> might then overlap and interpolation does not work anymore
        """
        for i in index_to_transform:
            verts[i] = transform(verts[i])
    triangles = Delaunay(verts).simplices
    def diam(T):
        length_of_edges = np.array(
            [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
        return np.max(length_of_edges)
    diameter = [diam(np.array([verts[triangles[i,][0]], verts[triangles[i,][1]], verts[triangles[i,][2]]])) for i in
                range(len(triangles))]
    h_min = np.min(diameter)
    h_max = np.max(diameter)
    return verts, triangles, h_min, h_max

if 0:
    h1, h2, delta = 0.05, 0.05, 0.1
    n1, n2 = int(np.ceil((1. + 2. * delta) / h1)) + 1, int(np.ceil((1. + 2. * delta) / h2)) + 1  # int(1./h)+1
    x, ret = np.linspace(-delta, 1. + delta, n1, endpoint=True, retstep=1)
    y = np.linspace(-delta, 1. + delta, n2, endpoint=True)
    verts = np.around(np.array(np.meshgrid(x, y, indexing='ij')).T.reshape(-1, 2), decimals=12)
    omega = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5)
    boundary = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) == 0.5)
    omega_i = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) > 0.5)

    verts = np.concatenate((verts[omega], verts[boundary], verts[omega_i]))

    omega = list(range(len(verts[omega])))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)
    boundary = list(range(len(verts[omega]), len(verts[omega]) + len(verts[boundary])))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) == 0.5)
    omega_i = list(range(len(verts[omega]) + len(verts[boundary]),len(verts)))  # np.where(np.linalg.norm(verts - np.ones(1) * 0.5, axis=1, ord=np.inf) > 0.5)
    nodes = omega+boundary
    def transform(x):
        return np.array([x[0], x[1] ** 2])
    # TRANSFORM
    index_to_transform = omega# np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5 - max(h1, h2))[0].tolist()
    if 0:
        for i in index_to_transform:
            verts[i] = transform(verts[i])
    triangles = Delaunay(verts).simplices

    #verts, triangles, h_min, h_max = prepare_mesh_gmsh(n1, geofile="unit_square_3_2")
    verts, triangles = meshzoo.rectangle(
        xmin=-delta, xmax=1.0 + delta,
        ymin=-delta, ymax=1.0 + delta,
        nx=n1 + 1, ny=n1 + 1,
        variant="zigzag")

    bary = (verts[triangles[:, 0]] + verts[triangles[:, 1]] + verts[
        triangles[:, 2]]) / 3.
    triangles = np.concatenate((np.zeros((len(triangles), 1), dtype=np.int), triangles), axis=1)
    new_omega = list(np.where(np.linalg.norm(bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])
    new_omega_i = list(set(range(len(triangles))) - set(new_omega))
    triangles[new_omega, 0] = 1
    triangles[new_omega_i, 0] = 2
    omega = triangles[new_omega]
    def plot_tri(liste, closed, fill, color):
        for i in liste:
            plt.gca().add_patch(plt.Polygon(verts[i], closed=closed, fill=fill, color=color, alpha=1))
    color = ['black', 'sienna']
    labels_domain = np.sort(np.unique(triangles[:, 0])).tolist()
    for label in labels_domain:
        omega = triangles[np.where(triangles[:, 0] == label)[0]]
        plot_tri(omega[:, 1:], closed=True, fill=False, color=color[label - 1])
    plt.axis('equal')
    # plt.savefig("a-zigzag.pdf", dpi=500)
    # plt.show()
#
