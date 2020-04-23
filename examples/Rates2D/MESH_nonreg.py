import scipy.sparse as ss
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from copy import copy
import math
import scipy.optimize as sc
from pathos.multiprocessing import ProcessingPool as Pool
import scipy.interpolate as si
import random
import scipy.sparse.linalg as ssl
from scipy.integrate import quad#quadrature as
from scipy.optimize import minimize
from scipy.spatial import Delaunay
import os
import bib3 as bib
"""-------------------------------------------------------------------------"""
"""                 PREPARE MESH                                            """
"""-------------------------------------------------------------------------"""
class Mesh:

    # object with all processed mesh data
    def __init__(self, proc_mesh_data):

        self.triangles = proc_mesh_data[0]
        self.omega = proc_mesh_data[1]
        self.verts = proc_mesh_data[2]
        self.hash_table = proc_mesh_data[3]
        self.boundary_verts = proc_mesh_data[4]
        self.nodes = proc_mesh_data[5]
        self.nhd = proc_mesh_data[6]
        self.diam = proc_mesh_data[7]
        self.support = proc_mesh_data[8]
        self.hash_table_approx = proc_mesh_data[9]
        self.bary = proc_mesh_data[10]
        self.boundary = proc_mesh_data[11]
        self.hash_table_bary = proc_mesh_data[12]
        self.shape_interface = proc_mesh_data[13]
        self.lines = proc_mesh_data[14]
        self.nV = proc_mesh_data[15]
        self.nE = proc_mesh_data[16]
        self.nE_Omega = proc_mesh_data[17]
        self.nV_Omega = proc_mesh_data[18]
        self.vertices = proc_mesh_data[2]

class Mesh_slim:
    # object with all processed mesh data
    def __init__(self, proc_mesh_data):

        self.triangles = proc_mesh_data[0]
        self.omega = proc_mesh_data[1]
        self.verts = proc_mesh_data[2]
        self.boundary_verts = proc_mesh_data[3]
        self.nodes = proc_mesh_data[4]
        self.boundary = proc_mesh_data[5]
        self.shape_interface = proc_mesh_data[6]


def prepare_mesh_gmsh(h, geofile = "unit_square"):

    os.system('gmsh mesh/' + geofile + '.geo -v 0 -2 -clscale ' + str(h) + ' -o mesh/' + geofile + '.msh')
    verts, lines, triangles = bib.read_mesh('mesh/' + geofile + '.msh')
    verts = verts[:, 0:2]
    sort_indices = np.argsort(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf))
    verts = verts[sort_indices]
    sort_indices_inv = np.arange(len(sort_indices))[np.argsort(sort_indices)]
    def f(n):
        return sort_indices_inv[n]
    triangles[:, 1:] = f(triangles[:, 1:])
    boundary = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) == 0.5)[0]
    nodes = np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) <= 0.5)[0]
    bary = (verts[triangles[:, 1]] + verts[triangles[:, 2]] + verts[triangles[:, 3]]) / 3.
    new_omega = list(np.where(np.linalg.norm(bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])
    new_omega_i = list(set(range(len(triangles))) - set(new_omega))
    triangles[new_omega, 0] = 1
    triangles[new_omega_i, 0] = 2
    omega = triangles[new_omega]
    def diam(T):
        length_of_edges = np.array(
            [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
        return np.max(length_of_edges)

    diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in
                range(len(triangles))]
    diam = np.max(diameter)
    proc_mesh_data = [triangles, omega, verts, [], boundary, nodes, [], diameter, [], [], bary,
                      boundary, [], [], [], len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    return mesh, proc_mesh_data

def prepare_mesh_nonreg(h1,h2, delta, transform_switch, transform):
    n1, n2 = int(np.ceil((1. + 2. * delta) / h1)) + 1, int(np.ceil((1. + 2. * delta) / h2)) + 1  # int(1./h)+1

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

    index_to_transform = omega# np.where(np.linalg.norm(verts - np.ones(2) * 0.5, axis=1, ord=np.inf) < 0.5 - max(h1, h2))[0].tolist()

    if transform_switch:
        """
        problem: index_to_transform inner-inner points --> might then overlap and interpolation does not work anymore
        """
        for i in index_to_transform:
            verts[i] = transform(verts[i])

    triangles = Delaunay(verts).simplices
    bary = (verts[triangles[:, 0]] + verts[triangles[:, 1]] + verts[
        triangles[:, 2]]) / 3.

    triangles = np.concatenate((np.zeros((len(triangles), 1), dtype=np.int), triangles), axis=1)

    new_omega = list(np.where(np.linalg.norm(bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])

    new_omega_i = list(set(range(len(triangles))) - set(new_omega))

    triangles[new_omega, 0] = 1
    triangles[new_omega_i, 0] = 2

    omega = triangles[new_omega]

    def diam(T):
        length_of_edges = np.array(
            [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
        return np.max(length_of_edges)

    diameter = [diam(np.array([verts[triangles[i,][1]], verts[triangles[i,][2]], verts[triangles[i,][3]]])) for i in
                range(len(triangles))]
    diam = np.max(diameter)


    proc_mesh_data = [triangles, omega, verts, [], boundary, nodes, [], diameter, [], [], bary,
                      boundary, [], [], [], len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)

    return mesh, proc_mesh_data




def prepare_mesh_nonreg_depricated(H, eps, norm, num_cores, transform_switch, transform):
    # if not (1 / h).is_integer() or not (eps / h).is_integer():
    #     print('(1/h and delta/h have to be an integer !!!')
    #     eps_i = h
    #
    # if eps < h:
    #     eps_i = h
    # else:
    #     eps_i = eps

    eps_i = eps

    # norm = norm_dict[Norm]
    a = [0., 0.]
    b = [1., 1.]
    gridsize = np.array([H[0], H[1]])

    diam = np.sqrt(gridsize[0] ** 2 + gridsize[1] ** 2)  # = longest edge of the triangle

    dim = 2

    def E(z, L):
        p = [np.prod([L[k] for k in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]
        return int(summe)

    def iE(k, L):
        p = [np.prod([L[l] for l in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k // (p[i]) - (k // (p[i - 1])) * L[i]
        return x

    # ==============================================================================
    #                    COMPUTE ARRAYS
    # ==============================================================================
    roundof = 6  # verts will not give precisely the value but rather 0.1000009 (in order to make
    # np.where () work we have to round of, roughly in the dimension of the gridsize)
    N = [int((b[i] - a[i]) / gridsize[i]) for i in range(dim)]

    #    L = [N[i]-1   for i in range(dim)]
    def ha(x):
        return np.array([gridsize[0] * x[0], gridsize[1] * x[1]])

    a_i = [a[0] - eps_i, a[1] - eps_i]
    b_i = np.around([b[0] + eps_i, b[1] + eps_i], decimals=roundof)
    N_i = [int(np.around((b_i[i] - a_i[i]) / gridsize[i])) + 1 for i in range(dim)]

    # -----------------------------------------------------------------------------#
    """ VERTS """

    def fun(k):
        return np.around(np.array(a_i) + ha(iE(k, N_i)), decimals=roundof)

    pool = Pool(processes=num_cores)
    verts = np.array(list(pool.map(fun, range(np.prod(N_i)))))  # np.array([  for k in range(np.prod(N_i))])
    pool.close()
    pool.join()
    pool.clear()
    #    print verts
    # -----------------------------------------------------------------------------#
    """ OMEGA """  # +  ha(iE(0,N)
    k_0 = np.where(np.all(verts[:, 0:2] == np.around(np.array(a), decimals=roundof).tolist(), axis=1))[0][0]
    omega = [k_0 + k + j * N_i[1] for k in range(N[1]) for j in range(N[0])]  # pool.map(om, range(num_omega))
    ####-----------------------------------------------------------------------------#
    """ OMEGA_I """
    omega_i = list(set(range(len(verts))) - set(omega) - set(np.where(verts[:, 0] == b_i[0])[0]) - set(
        np.where(verts[:, 1] == b_i[1])[0]))
    ###-----------------------------------------------------------------------------#
    # """ NODES """
    nodes = omega
    ##-----------------------------------------------------------------------------#
    """ OMEGA """
    omega = np.zeros((2 * len(nodes), 3), dtype=int)

    for j in range(len(nodes)):
        k = nodes[j]
        omega[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        omega[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    """ OMEGA_i """
    Omega_i = np.zeros((2 * len(omega_i), 3), dtype=int)

    for j in range(len(omega_i)):
        k = omega_i[j]
        Omega_i[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        Omega_i[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    omega_i = Omega_i

    ##-----------------------------------------------------------------------------#
    """ BOUNDARY """
    boundary1 = [k_0 + kk for kk in range(N[1])]
    boundary2 = [k_0 + N[1] + N_i[1] * kkk for kkk in range(N[0])]
    boundary3 = [k_0 + j * N_i[1] for j in range(N[0])]
    boundary4 = [k_0 + N[0] * N_i[1] + j for j in range(N[1] + 1)]

    boundary = np.unique(boundary1 + boundary2 + boundary3 + boundary4)

    ##-----------------------------------------------------------------------------#
    aux = np.zeros((len(omega), 4), dtype=int)
    aux[:, 1:] = omega
    aux[:, 0] = 1 * np.ones(len(omega), dtype=int)
    omega = aux

    aux = np.zeros((len(omega_i), 4), dtype=int)
    aux[:, 1:] = omega_i
    aux[:, 0] = 2 * np.ones(len(omega_i), dtype=int)
    omega_i = aux

    ##-----------------------------------------------------------------------------#
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))

    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:, 1:4].reshape(3 * num_omega))

    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """
    nodes_inner = list(set(nodes) - set(boundary))
    nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))
    verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))

    new_order = nodes_inner + list(boundary) + nodes_rest

    def permutation(i):
        return new_order.index(i)

    triangles_old = triangles

    triangles = np.zeros(np.shape(triangles_old), dtype=int)
    triangles[:, 0] = triangles_old[:, 0]

    for i in range(np.shape(triangles_old)[0]):
        for j in range(1, 4):
            triangles[i, j] = int(permutation(triangles_old[i, j]))

    omega = triangles[np.where(triangles[:, 0] != 2)[0]]
    # plot omega
    #    for i in range(len(omega)):
    #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2))

    boundary = list(range(len(nodes_inner), len(nodes_inner) + len(boundary)))
    nodes = list(range(len(nodes_inner))) + list(boundary)

    # PLOT verts
    #    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bo')
    #    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'rx')

    """ SUPPORT (subset omega)"""
    support = []  # [list(np.where(omega[:,1:4] == aa)[0]) for aa in nodes]

    # --------------------------------------------------------------------------
    """ Neighboring nodes x_j of x_k for which j<=k (subset of nodes)"""
    NHD = []
    # for k in range(len(nodes)):
    #     nhd = list(set(np.unique(triangles[support[k]][:,1:])) & set(omega[:,1:].reshape(3*len(omega)))) # subset of verts, intersect with omega, since nodes only are in omega
    #     #convert nhd from subset of verts into subset of nodes and take only those<=k
    #     aux = np.array([np.where(np.all(verts[nodes] == verts[j],axis=1))[0][0] for j in nhd]  )
    #     NHD += [aux[aux<=k].tolist()]
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset triangles)"""
    bary = (verts[triangles[:, 1]] + verts[triangles[:, 2]] + verts[triangles[:, 3]]) / 3.
    hash_table = []  # [np.where(norm((bary-np.repeat(verts[nodes[i]][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist() for i in range(len(nodes))]

    # --------------------------------------------------------------------------
    """ HASH TABLE_nodes (subset nodes)"""
    #    hash_table_nodes = [list(set(np.where(norm((verts[nodes]-np.repeat(verts[nodes[i]][:,np.newaxis], len(nodes), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist())&set(range(i+1))) for i in range(len(nodes))]

    """ HASH TABLE_approx (= hash table bary without puffer)"""
    hash_table_approx = []  # [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps))[0].tolist() for i in range(len(bary))]
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = []  # [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+ diam))[0].tolist() for i in range(len(bary))]

    # --------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""
    bdry_nodes = [nodes.index(i) for i in boundary]

    # --------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""
    interface = []  # np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()

    # ------------------------------------------------------------------------------
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, hash_table, boundary, nodes, NHD, diam, support, hash_table_approx, bary,
                      bdry_nodes, hash_table_bary, interface, [], len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    # ------------------------------------------------------------------------------
    if transform_switch:

        for i in range(len(mesh.verts[mesh.nodes])):
            mesh.verts[i] = transform(mesh.verts[i])

        ## REMESH
        """ ACHTUNG: LABELING interaction domain und domain passen nicht mehr !!!"""


        new_triangles = Delaunay(mesh.verts).simplices
        mesh.triangles[:, 1:] = new_triangles

        mesh.bary = (mesh.verts[mesh.triangles[:, 1]] + mesh.verts[mesh.triangles[:, 2]] + mesh.verts[
            mesh.triangles[:, 3]]) / 3.
        new_omega = list(np.where(np.linalg.norm(mesh.bary - np.ones(1) * 0.5, axis=1, ord=np.inf) < 0.5)[0])
        new_omega_i = list(set(range(len(mesh.triangles))) - set(new_omega))
        mesh.triangles[new_omega, 0] = 1
        mesh.triangles[new_omega_i, 0] = 2
        mesh.omega = mesh.triangles[new_omega]

        def diam(T):
            length_of_edges = np.array(
                [np.linalg.norm(T[0] - T[1]), np.linalg.norm(T[0] - T[2]), np.linalg.norm(T[1] - T[2])])
            return np.max(length_of_edges)

        diameter = [diam(np.array(
            [mesh.verts[mesh.triangles[i,][1]], mesh.verts[mesh.triangles[i,][2]], mesh.verts[mesh.triangles[i,][3]]]))
                    for i in range(len(mesh.triangles))]
        mesh.diam = np.max(diameter)
        print('new grid size: ', mesh.diam)



    return mesh, proc_mesh_data




def prepare_mesh(verts, lines, triangles, eps, Norm):
    verts = verts[: ,0:2]
    # note that eps is determined already in the mesh, however: for
    # homogeneous dirichtlet constraints it simply has to be smaller than
    # the pre-determined epsilon
    labels_domains = np.sort(np.unique(triangles[: ,0]))

    """ sort triangles """
    # we sort by labels
    # this is important such that omega[k,] = triangle[k,]
    # As a consequence we can use omega for the list 'support'
    triangles = triangles[triangles[: ,0].argsort()]

    norm = norm_dict[Norm]
    #    def norm(x):
    #        return np.max(np.abs(x), axis= 0)
    # --------------------------------------------------------------------------
    """ OMEGA """
    omega = triangles[np.where(triangles[: ,0] != labels_domains[-1])[0]]

    # --------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""
    interface = np.unique(lines[np.where(lines[: ,0 ]==12)][: ,1:]).tolist()

    # --------------------------------------------------------------------------
    """ NODES (subset of verts)"""

    omega1 = triangles[np.where(triangles[: ,0] == labels_domains[0])]
    nodes1 = list(np.unique(omega1[: ,1:4]))

    if len(labels_domains) > 2:
        omega2 = triangles[np.where(triangles[: ,0] == labels_domains[1])]
        nodes2 = list(np.unique(omega2[: ,1:4]))
    else:
        omega2 = []
        nodes2 = []

    G_N = []
    for i in interface:
        G_N += np.where(norm((verts -np.repeat(verts[i][: ,np.newaxis], len(verts), axis = 1).transpose()).transpose() )<=eps)[0].tolist()

    G_N1 = list( (set(nodes1 ) &set(G_N)) )
    G_N2 = list(  (set(nodes2 ) &set(G_N)) )

    nodes1 = list(set(nodes1) - set(G_N1) )
    nodes2 = list(set(nodes2) - set(G_N2) )
    G_N1 = list( (set(G_N1 ) -set(interface)) )
    G_N2 = list(  (set(G_N2 ) -set(interface)) )
    nodes = list(nodes1)  + list(G_N1 )+ interface + list(G_N2) + list(nodes2)
    nodes = np.array(nodes)

    #    print len(nodes1), len(G_N1), len(interface), len(G_N2), len(nodes2)

    #    nodes =  np.unique(omega[:,1:4])

    # test nodes
    #    plt.plot(verts[nodes1][:,0], verts[nodes1][:,1], 'bo')
    #    plt.plot(verts[nodes2][:,0], verts[nodes2][:,1], 'ro')
    #    plt.plot(verts[G_N1][:,0], verts[G_N1][:,1], 'gx')
    #    plt.plot(verts[G_N2][:,0], verts[G_N2][:,1], 'bx')
    #    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')

    # --------------------------------------------------------------------------
    """ BOUNDARY_verts (subset of verts); label = 9"""
    boundary = np.unique(lines[lines[: ,0] == 9][: ,1:3])
    # test nodes
    #    plt.plot(verts[boundary][:,0], verts[boundary][:,1], 'ro')


    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """
    if True:
        nodes_inner = list(set(nodes) - set(boundary))
        nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))

        verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))

        new_order = nodes_inner + list(boundary) + nodes_rest

        def permutation(i):
            return new_order.index(i)

        triangles_old = triangles

        triangles = np.zeros(np.shape(triangles_old), dtype = int)
        triangles[: ,0] = triangles_old[: ,0]

        for i in range(np.shape(triangles_old)[0]):
            for j in range(1 ,4):
                triangles[i ,j] = int(permutation(triangles_old[i ,j] ))

        lines_old = lines

        lines = np.zeros(np.shape(lines_old), dtype = int)
        lines[: ,0] = lines_old[: ,0]

        for i in range(np.shape(lines_old)[0]):
            for j in range(1 ,3):
                lines[i ,j] = int(permutation(lines_old[i ,j] ))

        interface_old= interface
        interface = np.zeros(len(interface_old), dtype = int)
        for i in range(len(interface_old)):
            interface[i] = permutation(interface_old[i])



        omega = triangles[np.where(triangles[: ,0] != labels_domains[-1])[0]]

        # plot omega
        #    for i in range(len(omega)):
        #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2))

        boundary = np.array(range(len(nodes_inner), len(nodes_inner ) +len(boundary)))

        nodes = np.array(range(len(nodes_inner ) +len(boundary))  )  # +list(boundary)

    #     PLOT verts
    #    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bx')
    #    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')
    #    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'ro')

    # --------------------------------------------------------------------------
    """ SUPPORT (subset omega)"""
    support = [  ]  # [list(np.where(omega[:,1:4] == a)[0]) for a in nodes]
    ##test support
    #    idx = 44#len(nodes)/2
    #    plt.plot(verts[nodes[idx]][0], verts[nodes[idx]][1], 'go')
    #    for i in support[idx]:
    #        plt.gca().add_patch(plt.Polygon([verts[triangles[i,1]], verts[triangles[i,2]],verts[triangles[i,3]]], closed=True, fill = True, color = 'blue'))

    # --------------------------------------------------------------------------
    """ Neighboring nodes x_j of x_k for which j<=k (subset of nodes)"""
    NHD = []
    # for k in range(len(nodes)):
    #     nhd = list(set(np.unique(triangles[support[k]][:,1:]))&set(omega[:,1:].reshape(3*len(omega)))) # subset of verts, intersect with omega, since nodes only are in omega
    #     #convert nhd from subset of verts into subset of nodes and take only those<=k
    #     aux = np.array([np.where(np.all(verts[nodes] == verts[j],axis=1))[0][0] for j in nhd]  )
    #     NHD += [aux[aux<=k].tolist()]
    ##test nhd
    #    idx = 44
    #    print verts[nodes[NHD[idx]]]
    #    for j in range(len(NHD[idx])):
    #        plt.plot(verts[nodes[NHD[idx]]][j][0], verts[nodes[NHD[idx]]][j][1], 'yo')
    #    plt.plot(verts[nodes[idx]][0], verts[nodes[idx]][1], 'ro')
    # --------------------------------------------------------------------------
    """Determine maximum diameter for hash_table """
    def diam(T):
        length_of_edges = np.array \
            ([np.linalg.norm(T[0 ] -T[1]) ,np.linalg.norm(T[0 ] -T[2]), np.linalg.norm(T[1 ] -T[2])] )
        return np.max(length_of_edges)
    diameter = [diam(np.array([verts[triangles[i,][1]] , verts[triangles[i,][2]] , verts[triangles[i,][3]] ])) for i in range(len(triangles))]
    diam = np.max(diameter)
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset triangles)"""
    bary = (verts[triangles[: ,1]] + verts[triangles[: ,2]] + verts[triangles[: ,3]]) / 3.
    hash_table = [  ]  # [np.where(norm((bary-np.repeat(verts[nodes[i]][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist() for i in range(len(nodes))]
    #    idx = 38#int(num_nodes/2 * 1.5)
    #    plt.plot(verts[nodes[idx]][0], verts[nodes[idx]][1], 'ro')
    #    for i in hash_table[idx]:
    #        T_i = triangles[i,]
    #        barycenter_i = (verts[T_i[1]] + verts[T_i[2]] + verts[T_i[3]]) / 3.
    #        plt.plot(barycenter_i[0],barycenter_i[1], 'yo')
    #
    #        plt.gca().add_patch(plt.Polygon([verts[triangles[i,1]], verts[triangles[i,2]],verts[triangles[i,3]]], closed=True, fill = True))
    #
    #    neighboring_nodes = list(np.unique(triangles[support[idx]][:,1:]))
    #    ngh_verts = verts[neighboring_nodes]
    #    for i in range(len(ngh_verts)):
    #        square = plt.Rectangle(tuple(ngh_verts[i]-(eps)*np.ones(2)), 2*(eps), 2*(eps), color='grey', fill= True)
    #        ax.add_artist(square)
    # --------------------------------------------------------------------------
    """ HASH TABLE_nodes (subset nodes)"""
    #    hash_table_nodes = [list(set(np.where(norm((verts[nodes]-np.repeat(verts[nodes[i]][:,np.newaxis], len(nodes), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist())&set(range(i+1))) for i in range(len(nodes))]

    """ HASH TABLE_approx (= hash table bary without puffer)"""
    hash_table_approx = [  ]# [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps))[0].tolist() for i in range(len(bary))]
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = [  ]  # [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+ diam))[0].tolist() for i in range(len(bary))]

    # --------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""
    bdry_nodes = [nodes.tolist().index(i) for i in boundary]

    # ------------------------------------------------------------------------------
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, hash_table, boundary, nodes, NHD, max(diameter), support, hash_table_approx, bary, bdry_nodes, hash_table_bary, interface, lines, len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    # ------------------------------------------------------------------------------
    return mesh, proc_mesh_data
def prepare_mesh_reg_slim(h, eps, Norm, num_cores):
    if not (1 / h).is_integer() or not (eps / h).is_integer():
        print('(1/h and delta/h have to be an integer !!!')

    if eps < h:
        eps_i = h
    else:
        eps_i = eps

    a = [0., 0.]
    b = [1., 1.]
    gridsize = np.array([h, h])

    dim = 2

    def E(z, L):
        p = [np.prod([L[k] for k in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]
        return int(summe)

    def iE(k, L):
        p = [np.prod([L[l] for l in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k // (int(p[i])) - (k // (int(p[i - 1]))) * L[i]
        return x

    # ==============================================================================
    #                    COMPUTE ARRAYS
    # ==============================================================================
    roundof = 6  # verts will not give precisely the value but rather 0.1000009 (in order to make
    # np.where () work we have to ceil roughly in the dimension of the gridsize)
    N = [int((b[i] - a[i]) / gridsize[i]) for i in range(dim)]

    #    L = [N[i]-1   for i in range(dim)]
    def ha(x):
        return np.array([gridsize[0] * x[0], gridsize[1] * x[1]])

    a_i = [a[0] - eps_i, a[1] - eps_i]
    b_i = np.around([b[0] + eps_i, b[1] + eps_i], decimals=roundof)
    N_i = [int(np.around((b_i[i] - a_i[i]) / gridsize[i])) + 1 for i in range(dim)]

    # -----------------------------------------------------------------------------#
    """ VERTS """

    def fun(k):
        return np.around(np.array(a_i) + ha(iE(k, N_i)), decimals=roundof)

    pool = Pool(processes=num_cores)
    verts = np.array(pool.map(fun, range(np.prod(N_i))))  # np.array([  for k in range(np.prod(N_i))])
    pool.close()
    pool.join()
    pool.clear()
    #    print verts
    # -----------------------------------------------------------------------------#
    """ OMEGA """  # +  ha(iE(0,N)
    k_0 = np.where(np.all(verts[:, 0:2] == np.around(np.array(a), decimals=roundof).tolist(), axis=1))[0][0]
    omega = [k_0 + k + j * N_i[1] for k in range(N[1]) for j in range(N[0])]  # pool.map(om, range(num_omega))
    ####-----------------------------------------------------------------------------#
    """ OMEGA_I """
    omega_i = list(set(range(len(verts))) - set(omega) - set(np.where(verts[:, 0] == b_i[0])[0]) - set(
        np.where(verts[:, 1] == b_i[1])[0]))
    ###-----------------------------------------------------------------------------#
    # """ NODES """
    nodes = omega
    ##-----------------------------------------------------------------------------#
    """ OMEGA """
    omega = np.zeros((2 * len(nodes), 3), dtype=int)

    for j in range(len(nodes)):
        k = nodes[j]
        omega[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        omega[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    """ OMEGA_i """
    Omega_i = np.zeros((2 * len(omega_i), 3), dtype=int)

    for j in range(len(omega_i)):
        k = omega_i[j]
        Omega_i[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        Omega_i[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    omega_i = Omega_i

    ##-----------------------------------------------------------------------------#
    """ BOUNDARY """
    boundary1 = [k_0 + kk for kk in range(N[0])]
    boundary2 = [k_0 + N[1] * N_i[0] + kkk for kkk in range(N[0] + 1)]
    boundary3 = [k_0 + j * N_i[0] for j in range(N[1])]
    boundary4 = [k_0 + N[0] + j * N_i[0] for j in range(N[1])]

    boundary = np.unique(boundary1 + boundary2 + boundary3 + boundary4)

    ##-----------------------------------------------------------------------------#

    aux = np.zeros((len(omega), 4), dtype=int)
    aux[:, 1:] = omega
    aux[:, 0] = 1 * np.ones(len(omega), dtype=int)
    omega = aux

    aux = np.zeros((len(omega_i), 4), dtype=int)
    aux[:, 1:] = omega_i
    aux[:, 0] = 2 * np.ones(len(omega_i), dtype=int)
    omega_i = aux

    ##-----------------------------------------------------------------------------#
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))

    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:, 1:4].reshape(3 * num_omega))

    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """
    nodes_inner = list(set(nodes) - set(boundary))
    nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))
    verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))

    new_order = nodes_inner + list(boundary) + nodes_rest

    def permutation(i):
        return new_order.index(i)

    triangles_old = triangles
    triangles = np.zeros(np.shape(triangles_old), dtype=int)
    triangles[:, 0] = triangles_old[:, 0]

    for i in range(np.shape(triangles_old)[0]):
        for j in range(1, 4):
            triangles[i, j] = int(permutation(triangles_old[i, j]))

    omega = triangles[np.where(triangles[:, 0] != 2)[0]]
    boundary = range(len(nodes_inner), len(nodes_inner) + len(boundary))
    nodes = list(range(len(nodes_inner))) + list(boundary)

    # --------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""
    bdry_nodes = [nodes.index(i) for i in boundary]

    # --------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""
    interface = []  # np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()

    # ------------------------------------------------------------------------------
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, boundary, nodes, bdry_nodes, interface]
    mesh = Mesh_slim(proc_mesh_data)
    # ------------------------------------------------------------------------------
    return mesh, proc_mesh_data
def prepare_mesh_slim(verts, lines, triangles, eps):
    verts = verts[:, 0:2]
    # note that eps is determined already in the mesh, however: for
    # homogeneous dirichtlet constraints it simply has to be smaller than
    # the pre-determined epsilon

    """ sort triangles """
    # we sort by labels
    # this is important such that omega[k,] = triangle[k,]
    # As a consequence we can use omega for the list 'support'
    triangles = triangles[triangles[:, 0].argsort()]

    def norm(x):
        return np.max(np.abs(x), axis=0)

    # --------------------------------------------------------------------------
    """ DECRYPTING LABELS: 1 = shape; 2 = omega\shape; 3 = omega_i """
    omega = triangles[np.where(triangles[:, 0] != 3)[0]]

    # --------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""
    interface = np.unique(lines[np.where(lines[:, 0] == 12)][:, 1:]).tolist()

    # --------------------------------------------------------------------------
    """ NODES (subset of verts)"""

    omega1 = triangles[np.where(triangles[:, 0] == 1)]
    omega2 = triangles[np.where(triangles[:, 0] == 2)]
    nodes1 = list(np.unique(omega1[:, 1:4]))
    nodes2 = list(np.unique(omega2[:, 1:4]))

    G_N = []
    for i in interface:
        G_N += \
        np.where(norm((verts - np.repeat(verts[i][:, np.newaxis], len(verts), axis=1).transpose()).transpose()) <= eps)[
            0].tolist()

    G_N1 = list((set(nodes1) & set(G_N)))
    G_N2 = list((set(nodes2) & set(G_N)))

    nodes1 = list(set(nodes1) - set(G_N1))
    nodes2 = list(set(nodes2) - set(G_N2))
    G_N1 = list((set(G_N1) - set(interface)))
    G_N2 = list((set(G_N2) - set(interface)))
    nodes = list(nodes1) + list(G_N1) + interface + list(G_N2) + list(nodes2)
    nodes = np.array(nodes)

    # --------------------------------------------------------------------------
    """ BOUNDARY_verts (subset of verts); label = 9"""
    boundary = np.unique(lines[lines[:, 0] == 9][:, 1:3])
    # test nodes
    #    plt.plot(verts[boundary][:,0], verts[boundary][:,1], 'ro')
    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """
    if True:
        nodes_inner = list(set(nodes) - set(boundary))
        nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))

        verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))

        new_order = nodes_inner + list(boundary) + nodes_rest

        def permutation(i):
            return new_order.index(i)

        triangles_old = triangles
        triangles = np.zeros(np.shape(triangles_old), dtype=int)
        triangles[:, 0] = triangles_old[:, 0]

        for i in range(np.shape(triangles_old)[0]):
            for j in range(1, 4):
                triangles[i, j] = int(permutation(triangles_old[i, j]))

        interface_old = interface
        interface = np.zeros(len(interface_old), dtype=int)
        for i in range(len(interface_old)):
            interface[i] = permutation(interface_old[i])

        omega = triangles[np.where(triangles[:, 0] != 3)[0]]
        # plot omega
        #    for i in range(len(omega)):
        #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2))

        boundary = np.array(range(len(nodes_inner), len(nodes_inner) + len(boundary)))
        nodes = np.array(range(len(nodes_inner) + len(boundary)))  # +list(boundary)
    #     PLOT verts
    #    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bx')
    #    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')
    #    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'ro')
    # --------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""
    bdry_nodes = [nodes.tolist().index(i) for i in boundary]

    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, boundary, nodes, bdry_nodes, interface]
    mesh = Mesh_slim(proc_mesh_data)
    # ------------------------------------------------------------------------------
    return mesh, proc_mesh_data
def prepare_mesh_reg(h, eps, norm, num_cores):
    if not ( 1 /h).is_integer() or not (eps /h).is_integer():
        print('(1/h and delta/h have to be an integer !!!')
        eps_i = h

    if eps < h:
        eps_i = h
    else:
        eps_i = eps



    a = [0. ,0.]
    b = [1. ,1.]
    gridsize = np.array([h ,h])

    diam = np.sqrt( h**2 + h** 2)  # = longest edge of the triangle

    dim = 2

    def E(z, L):
        p = [np.prod([L[k] for k in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]
        return int(summe)

    def iE(k, L):
        p = [np.prod([L[l] for l in range(j + 1, dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k // (p[i]) - (k // (p[i - 1])) * L[i]
        return x

    # ==============================================================================
    #                    COMPUTE ARRAYS
    # ==============================================================================
    roundof = 6  # verts will not give precisely the value but rather 0.1000009 (in order to make
    # np.where () work we have to round of, roughly in the dimension of the gridsize)
    N = [int((b[i] - a[i]) / gridsize[i]) for i in range(dim)]

    #    L = [N[i]-1   for i in range(dim)]
    def ha(x):
        return np.array([gridsize[0] * x[0], gridsize[1] * x[1]])

    a_i = [a[0] - eps_i, a[1] - eps_i]
    b_i = np.around([b[0] + eps_i, b[1] + eps_i], decimals=roundof)
    N_i = [int(np.around((b_i[i] - a_i[i]) / gridsize[i])) + 1 for i in range(dim)]

    # -----------------------------------------------------------------------------#
    """ VERTS """

    def fun(k):
        return np.around(np.array(a_i) + ha(iE(k, N_i)), decimals=roundof)

    pool = Pool(processes=num_cores)
    verts = np.array(list(pool.map(fun, range(np.prod(N_i)))))  # np.array([  for k in range(np.prod(N_i))])
    pool.close()
    pool.join()
    pool.clear()
    #    print verts
    # -----------------------------------------------------------------------------#
    """ OMEGA """  # +  ha(iE(0,N)
    k_0 = np.where(np.all(verts[:, 0:2] == np.around(np.array(a), decimals=roundof).tolist(), axis=1))[0][0]
    omega = [k_0 + k + j * N_i[1] for k in range(N[1]) for j in range(N[0])]  # pool.map(om, range(num_omega))
    ####-----------------------------------------------------------------------------#
    """ OMEGA_I """
    omega_i = list(set(range(len(verts))) - set(omega) - set(np.where(verts[:, 0] == b_i[0])[0]) - set(
        np.where(verts[:, 1] == b_i[1])[0]))
    ###-----------------------------------------------------------------------------#
    # """ NODES """
    nodes = omega
    ##-----------------------------------------------------------------------------#
    """ OMEGA """
    omega = np.zeros((2 * len(nodes), 3), dtype=int)

    for j in range(len(nodes)):
        k = nodes[j]
        omega[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        omega[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    """ OMEGA_i """
    Omega_i = np.zeros((2 * len(omega_i), 3), dtype=int)

    for j in range(len(omega_i)):
        k = omega_i[j]
        Omega_i[2 * j,] = np.array([k, k + N_i[1] + 1, k + 1])  # clockwise
        Omega_i[2 * j + 1,] = np.array([k, k + N_i[1], k + N_i[1] + 1])  # clockwise

    omega_i = Omega_i

    ##-----------------------------------------------------------------------------#
    """ BOUNDARY """
    boundary1 = [k_0 + kk for kk in range(N[0])]
    boundary2 = [k_0 + N[1] * N_i[0] + kkk for kkk in range(N[0] + 1)]
    boundary3 = [k_0 + j * N_i[0] for j in range(N[1])]
    boundary4 = [k_0 + N[0] + j * N_i[0] for j in range(N[1])]

    boundary = np.unique(boundary1 + boundary2 + boundary3 + boundary4)

    ##-----------------------------------------------------------------------------#
    aux = np.zeros((len(omega), 4), dtype=int)
    aux[:, 1:] = omega
    aux[:, 0] = 1 * np.ones(len(omega), dtype=int)
    omega = aux

    aux = np.zeros((len(omega_i), 4), dtype=int)
    aux[:, 1:] = omega_i
    aux[:, 0] = 2 * np.ones(len(omega_i), dtype=int)
    omega_i = aux

    ##-----------------------------------------------------------------------------#
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))

    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:, 1:4].reshape(3 * num_omega))

    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """
    nodes_inner = list(set(nodes) - set(boundary))
    nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))
    verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))

    new_order = nodes_inner + list(boundary) + nodes_rest

    def permutation(i):
        return new_order.index(i)

    triangles_old = triangles

    triangles = np.zeros(np.shape(triangles_old), dtype=int)
    triangles[:, 0] = triangles_old[:, 0]

    for i in range(np.shape(triangles_old)[0]):
        for j in range(1, 4):
            triangles[i, j] = int(permutation(triangles_old[i, j]))

    omega = triangles[np.where(triangles[:, 0] != 2)[0]]
    # plot omega
    #    for i in range(len(omega)):
    #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2))

    boundary = list(range(len(nodes_inner), len(nodes_inner) + len(boundary)))
    nodes = list(range(len(nodes_inner))) + list(boundary)

    # PLOT verts
    #    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bo')
    #    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'rx')

    """ SUPPORT (subset omega)"""
    support = []  # [list(np.where(omega[:,1:4] == aa)[0]) for aa in nodes]

    # --------------------------------------------------------------------------
    """ Neighboring nodes x_j of x_k for which j<=k (subset of nodes)"""
    NHD = []
    # for k in range(len(nodes)):
    #     nhd = list(set(np.unique(triangles[support[k]][:,1:])) & set(omega[:,1:].reshape(3*len(omega)))) # subset of verts, intersect with omega, since nodes only are in omega
    #     #convert nhd from subset of verts into subset of nodes and take only those<=k
    #     aux = np.array([np.where(np.all(verts[nodes] == verts[j],axis=1))[0][0] for j in nhd]  )
    #     NHD += [aux[aux<=k].tolist()]
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset triangles)"""
    bary = (verts[triangles[:, 1]] + verts[triangles[:, 2]] + verts[triangles[:, 3]]) / 3.
    hash_table = []  # [np.where(norm((bary-np.repeat(verts[nodes[i]][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist() for i in range(len(nodes))]

    # --------------------------------------------------------------------------
    """ HASH TABLE_nodes (subset nodes)"""
    #    hash_table_nodes = [list(set(np.where(norm((verts[nodes]-np.repeat(verts[nodes[i]][:,np.newaxis], len(nodes), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist())&set(range(i+1))) for i in range(len(nodes))]

    """ HASH TABLE_approx (= hash table bary without puffer)"""
    hash_table_approx = []  # [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps))[0].tolist() for i in range(len(bary))]
    # --------------------------------------------------------------------------
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = []  # [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+ diam))[0].tolist() for i in range(len(bary))]

    # --------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""
    bdry_nodes = [nodes.index(i) for i in boundary]

    # --------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""
    interface = []  # np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()

    # ------------------------------------------------------------------------------
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, hash_table, boundary, nodes, NHD, diam, support, hash_table_approx, bary,
                      bdry_nodes, hash_table_bary, interface, [], len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    # ------------------------------------------------------------------------------
    return mesh, proc_mesh_data