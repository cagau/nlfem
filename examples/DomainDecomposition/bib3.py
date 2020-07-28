# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:35:57 2018

@author: vollmann
"""
import numpy as np
Px = np.array([[0.33333333, 0.33333333],
              [0.45929259, 0.45929259],
              [0.45929259, 0.08141482],
              [0.08141482, 0.45929259],
              [0.17056931, 0.17056931],
              [0.17056931, 0.65886138],
              [0.65886138, 0.17056931],
              [0.05054723, 0.05054723],
              [0.05054723, 0.89890554],
              [0.89890554, 0.05054723],
              [0.26311283, 0.72849239],
              [0.72849239, 0.00839478],
              [0.00839478, 0.26311283],
              [0.72849239, 0.26311283],
              [0.26311283, 0.00839478],
              [0.00839478, 0.72849239]])

weightsxy = 0.5 * np.array([0.14431560767779
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443])

#from assemble import assemble

import scipy.sparse as ss
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

"""========================================================================="""
""" PARAMETERS """
#ball = 'exact'  # choose from ['exact', 'approx1', 'approx2', 'approx3']

quad_order_outer = 'new'#'#8#'test'#'new'#8 # choose from [8, 5, 3, 2, 'new', 'GL-5-12']
quad_order_inner = 1

# for 1d
n1_1d, n2_1d = 16,2

# for tri_adapt
tol1_Radon =  1e-10# difference between g_low and g_high
tol2_Radon = 0.1 #for diameter


test_assembly = 0
P_test = np.array([ [0.0, 0.0]]).transpose()#np.array([ [ 1./3., 1./3.]]).transpose()
"""========================================================================="""

""" CONVENTIONS """
# LABELS: 1 = shape; 2 = omega\shape; 3 = omega_i, 9 = boundary of omega


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

def prepare_mesh(verts, lines, triangles, eps, Norm):
    verts = verts[:,0:2]
    # note that eps is determined already in the mesh, however: for 
    # homogeneous dirichtlet constraints it simply has to be smaller than
    # the pre-determined epsilon
    labels_domains = np.sort(np.unique(triangles[:,0]))

    """ sort triangles """
    # we sort by labels
    # this is important such that omega[k,] = triangle[k,]
    # As a consequence we can use omega for the list 'support'
    triangles = triangles[triangles[:,0].argsort()]

    norm = norm_dict[Norm]
#    def norm(x):
#        return np.max(np.abs(x), axis= 0)
    #--------------------------------------------------------------------------
    """ OMEGA """   
    omega = triangles[np.where(triangles[:,0] != labels_domains[-1])[0]]    


    #--------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""      
    interface = np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()   
    
    #--------------------------------------------------------------------------
    """ NODES (subset of verts)"""   

    omega1 = triangles[np.where(triangles[:,0] == labels_domains[0])] 
    nodes1 = list(np.unique(omega1[:,1:4]))

    if len(labels_domains) > 2:
        omega2 = triangles[np.where(triangles[:,0] == labels_domains[1])] 
        nodes2 = list(np.unique(omega2[:,1:4]))
    else:
        omega2 = []
        nodes2 = []

    G_N = []
    for i in interface:
        G_N += np.where(norm((verts-np.repeat(verts[i][:,np.newaxis], len(verts), axis = 1).transpose()).transpose())<=eps)[0].tolist()

    G_N1 = list( (set(nodes1)&set(G_N)) )
    G_N2 = list(  (set(nodes2)&set(G_N)) )
    
    nodes1 = list(set(nodes1) - set(G_N1) )
    nodes2 = list(set(nodes2) - set(G_N2) )
    G_N1 = list( (set(G_N1)-set(interface)) )
    G_N2 = list(  (set(G_N2)-set(interface)) )
    nodes = list(nodes1)  + list(G_N1)+ interface + list(G_N2) + list(nodes2) 
    nodes = np.array(nodes)

#    print len(nodes1), len(G_N1), len(interface), len(G_N2), len(nodes2)

#    nodes =  np.unique(omega[:,1:4])

    # test nodes
#    plt.plot(verts[nodes1][:,0], verts[nodes1][:,1], 'bo')
#    plt.plot(verts[nodes2][:,0], verts[nodes2][:,1], 'ro')
#    plt.plot(verts[G_N1][:,0], verts[G_N1][:,1], 'gx')
#    plt.plot(verts[G_N2][:,0], verts[G_N2][:,1], 'bx')
#    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')
    
    #--------------------------------------------------------------------------
    """ BOUNDARY_verts (subset of verts); label = 9"""
    boundary = np.unique(lines[lines[:,0] == 9][:,1:3])
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
        triangles[:,0] = triangles_old[:,0]
        
        for i in range(np.shape(triangles_old)[0]):
            for j in range(1,4):
                triangles[i,j] = int(permutation(triangles_old[i,j] ))
        
        lines_old = lines
        
        lines = np.zeros(np.shape(lines_old), dtype = int)
        lines[:,0] = lines_old[:,0]
        
        for i in range(np.shape(lines_old)[0]):
            for j in range(1,3):
                lines[i,j] = int(permutation(lines_old[i,j] ))

        interface_old= interface
        interface = np.zeros(len(interface_old), dtype = int)
        for i in range(len(interface_old)):
            interface[i] = permutation(interface_old[i])



        omega = triangles[np.where(triangles[:,0] != labels_domains[-1])[0]]    
        
        # plot omega
    #    for i in range(len(omega)):
    #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2)) 
        
        boundary = np.array(range(len(nodes_inner), len(nodes_inner)+len(boundary)))
        
        nodes = np.array(range(len(nodes_inner)+len(boundary)))#+list(boundary)
    
#     PLOT verts
#    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bx')
#    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')
#    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'ro')
    
    
    
    
    #--------------------------------------------------------------------------
    """ SUPPORT (subset omega)"""
    support = []#[list(np.where(omega[:,1:4] == a)[0]) for a in nodes]
    ##test support
#    idx = 44#len(nodes)/2
#    plt.plot(verts[nodes[idx]][0], verts[nodes[idx]][1], 'go')
#    for i in support[idx]:
#        plt.gca().add_patch(plt.Polygon([verts[triangles[i,1]], verts[triangles[i,2]],verts[triangles[i,3]]], closed=True, fill = True, color = 'blue')) 

    #--------------------------------------------------------------------------
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
    #--------------------------------------------------------------------------
    """Determine maximum diameter for hash_table """
    def diam(T):
        length_of_edges = np.array([np.linalg.norm(T[0]-T[1]),np.linalg.norm(T[0]-T[2]), np.linalg.norm(T[1]-T[2])] )   
        return np.max(length_of_edges)
    diameter = [diam(np.array([verts[triangles[i,][1]] , verts[triangles[i,][2]] , verts[triangles[i,][3]] ])) for i in range(len(triangles))]
    diam = np.max(diameter)
    #--------------------------------------------------------------------------
    """ HASH TABLE (subset triangles)"""
    bary = (verts[triangles[:,1]] + verts[triangles[:,2]] + verts[triangles[:,3]]) / 3.       
    hash_table = []#[np.where(norm((bary-np.repeat(verts[nodes[i]][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist() for i in range(len(nodes))]
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
    #--------------------------------------------------------------------------
    """ HASH TABLE_nodes (subset nodes)"""      
#    hash_table_nodes = [list(set(np.where(norm((verts[nodes]-np.repeat(verts[nodes[i]][:,np.newaxis], len(nodes), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist())&set(range(i+1))) for i in range(len(nodes))]
    
    """ HASH TABLE_approx (= hash table bary without puffer)"""      
    hash_table_approx = []# [np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps))[0].tolist() for i in range(len(bary))]
    #--------------------------------------------------------------------------
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = []#[np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+ diam))[0].tolist() for i in range(len(bary))]
    
    #--------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""      
    bdry_nodes = [nodes.tolist().index(i) for i in boundary] 

    #------------------------------------------------------------------------------  
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, hash_table, boundary, nodes, NHD, max(diameter), support, hash_table_approx, bary, bdry_nodes, hash_table_bary, interface, lines, len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    #------------------------------------------------------------------------------       
    return mesh, proc_mesh_data


def prepare_mesh_reg(h, eps, Norm, num_cores):
    if not (1/h).is_integer() or not (eps/h).is_integer():
        print('(1/h and delta/h have to be an integer !!!')
        eps_i = h
        
    if eps < h:
        eps_i = h
    else:
        eps_i = eps


    norm = norm_dict[Norm]
    a = [0.,0.]
    b = [1.,1.]
    gridsize = np.array([h,h])
    
    diam = np.sqrt(h**2 + h**2) # = longest edge of the triangle
    
    dim = 2 
    def E(z, L):
        p = [np.prod([L[k] for k in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]  
        return int(summe)
    def iE(k, L):
        p = [np.prod([L[l] for l in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k//(p[i]) - (k//(p[i-1])) * L[i]
        return x

    #==============================================================================
    #                    COMPUTE ARRAYS
    #==============================================================================
    roundof = 6 # verts will not give precisely the value but rather 0.1000009 (in order to make 
                # np.where () work we have to round of, roughly in the dimension of the gridsize)
    N = [int((b[i]-a[i])/gridsize[i]) for i in range(dim)]
#    L = [N[i]-1   for i in range(dim)]
    def ha(x):
        return np.array([gridsize[0]*x[0], gridsize[1]*x[1]])
    
    a_i = [a[0]-eps_i, a[1]-eps_i]
    b_i = np.around([b[0]+eps_i, b[1]+eps_i], decimals = roundof)
    N_i = [int(np.around((b_i[i]-a_i[i])/gridsize[i])) + 1 for i in range(dim)]
    
    #-----------------------------------------------------------------------------#
    """ VERTS """
    def fun(k):
        return np.around(np.array(a_i) +  ha(iE(k,N_i)), decimals = roundof)
    pool = Pool(processes=num_cores) 
    verts = np.array(list(pool.map(fun,range(np.prod(N_i)) )))# np.array([  for k in range(np.prod(N_i))])
    pool.close()
    pool.join()
    pool.clear()
#    print verts
    #-----------------------------------------------------------------------------#
    """ OMEGA """#+  ha(iE(0,N)
    k_0 =  np.where(np.all(verts[:,0:2] == np.around(np.array(a) , decimals = roundof).tolist(),axis=1))[0][0]
    omega = [k_0 + k + j*N_i[1] for k in range(N[1]) for j in range(N[0])]# pool.map(om, range(num_omega))    
    ####-----------------------------------------------------------------------------#
    """ OMEGA_I """
    omega_i = list(set(range(len(verts))) - set(omega) -set(np.where(verts[:,0]== b_i[0])[0])  -set(np.where(verts[:,1]== b_i[1])[0]  ) )
    ###-----------------------------------------------------------------------------#
    #""" NODES """
    nodes = omega  
    ##-----------------------------------------------------------------------------#
    """ OMEGA """
    omega = np.zeros((2*len(nodes), 3), dtype = int)
    
    for j in range(len(nodes)):
        k = nodes[j]
        omega[2*j,    ] = np.array([k, k+ N_i[1]+1,k + 1])#  clockwise
        omega[2*j + 1,    ] = np.array([k, k + N_i[1], k+ N_i[1]+1])#  clockwise
    
    """ OMEGA_i """
    Omega_i = np.zeros((2*len(omega_i), 3), dtype = int)
    
    for j in range(len(omega_i)):
        k = omega_i[j]
        Omega_i[2*j,    ] = np.array([k, k+ N_i[1]+1,k + 1 ]) # clockwise
        Omega_i[2*j +1, ] = np.array([k,k + N_i[1], k+ N_i[1]+1])#clockwise
    
    omega_i = Omega_i
    
    ##-----------------------------------------------------------------------------#
    """ BOUNDARY """
    boundary1 = [k_0 + kk  for kk in range(N[0])]
    boundary2 = [k_0 + N[1]*N_i[0] + kkk  for kkk in range(N[0]+1)]
    boundary3 = [k_0 + j*N_i[0] for j in range(N[1])]
    boundary4 = [k_0 +N[0]+ j*N_i[0] for j in range(N[1])]
    
    boundary = np.unique( boundary1 + boundary2 + boundary3 + boundary4)

    ##-----------------------------------------------------------------------------#
    aux = np.zeros((len(omega),4), dtype=int)    
    aux[:,1:] = omega  
    aux[:,0] = 1*np.ones(len(omega), dtype = int)
    omega = aux
    
    aux = np.zeros((len(omega_i),4), dtype=int)    
    aux[:,1:] = omega_i  
    aux[:,0] = 2*np.ones(len(omega_i), dtype = int)
    omega_i = aux
    
    ##-----------------------------------------------------------------------------#
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))

    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:,1:4].reshape(3*num_omega))

    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """ 
    nodes_inner = list(set(nodes) - set(boundary))
    nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))
    verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))
    
    new_order = nodes_inner + list(boundary) + nodes_rest
    def permutation(i):
        return new_order.index(i)
    
    triangles_old = triangles
    
    triangles = np.zeros(np.shape(triangles_old), dtype = int)
    triangles[:,0] = triangles_old[:,0]
    
    for i in range(np.shape(triangles_old)[0]):
        for j in range(1,4):
            triangles[i,j] = int(permutation(triangles_old[i,j] ))

    omega = triangles[np.where(triangles[:,0] != 2)[0]]
    # plot omega
#    for i in range(len(omega)):
#        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2)) 
    
    boundary = list(range(len(nodes_inner), len(nodes_inner)+len(boundary)))
    nodes = list(range(len(nodes_inner)))+list(boundary)
    
    # PLOT verts
#    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bo')
#    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'rx')
    
    """ SUPPORT (subset omega)"""
    support = []# [list(np.where(omega[:,1:4] == aa)[0]) for aa in nodes]

    #--------------------------------------------------------------------------
    """ Neighboring nodes x_j of x_k for which j<=k (subset of nodes)"""
    NHD = []
    # for k in range(len(nodes)):
    #     nhd = list(set(np.unique(triangles[support[k]][:,1:])) & set(omega[:,1:].reshape(3*len(omega)))) # subset of verts, intersect with omega, since nodes only are in omega
    #     #convert nhd from subset of verts into subset of nodes and take only those<=k
    #     aux = np.array([np.where(np.all(verts[nodes] == verts[j],axis=1))[0][0] for j in nhd]  )
    #     NHD += [aux[aux<=k].tolist()]
    #--------------------------------------------------------------------------
    """ HASH TABLE (subset triangles)"""
    bary = (verts[triangles[:,1]] + verts[triangles[:,2]] + verts[triangles[:,3]]) / 3.       
    hash_table = []#[np.where(norm((bary-np.repeat(verts[nodes[i]][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist() for i in range(len(nodes))]

    #--------------------------------------------------------------------------
    """ HASH TABLE_nodes (subset nodes)"""      
#    hash_table_nodes = [list(set(np.where(norm((verts[nodes]-np.repeat(verts[nodes[i]][:,np.newaxis], len(nodes), axis = 1).transpose()).transpose())<=(eps+diam))[0].tolist())&set(range(i+1))) for i in range(len(nodes))]
    
    """ HASH TABLE_approx (= hash table bary without puffer)"""      
    hash_table_approx = []#[np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps))[0].tolist() for i in range(len(bary))]
    #--------------------------------------------------------------------------
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = []#[np.where(norm((bary-np.repeat(bary[i][:,np.newaxis], len(bary), axis = 1).transpose()).transpose())<=(eps+ diam))[0].tolist() for i in range(len(bary))]
    
    #--------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""      
    bdry_nodes = [nodes.index(i) for i in boundary] 
    
    #--------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""      
    interface = []#np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()     

    #------------------------------------------------------------------------------  
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, hash_table, boundary, nodes, NHD, diam, support, hash_table_approx, bary, bdry_nodes, hash_table_bary, interface,[],len(verts), len(triangles), len(omega), len(nodes)]
    mesh = Mesh(proc_mesh_data)
    #------------------------------------------------------------------------------       
    return mesh, proc_mesh_data


def prepare_mesh_nonreg(H, eps, Norm, num_cores):
    # if not (1 / h).is_integer() or not (eps / h).is_integer():
    #     print('(1/h and delta/h have to be an integer !!!')
    #     eps_i = h
    #
    # if eps < h:
    #     eps_i = h
    # else:
    #     eps_i = eps

    eps_i = eps

    norm = norm_dict[Norm]
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
    boundary4 = [k_0 + N[0]*N_i[1] + j for j in range(N[1]+1)]

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


def prepare_mesh_reg_slim(h, eps, Norm, num_cores):
    if not (1/h).is_integer() or not (eps/h).is_integer():
        print('(1/h and delta/h have to be an integer !!!')
        
    if eps < h:
        eps_i = h
    else:
        eps_i = eps

    a = [0.,0.]
    b = [1.,1.]
    gridsize = np.array([h,h])

    dim = 2 
    def E(z, L):
        p = [np.prod([L[k] for k in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]  
        return int(summe)
    def iE(k, L):
        p = [np.prod([L[l] for l in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k//(int(p[i])) - (k//(int(p[i-1]))) * L[i]
        return x

    #==============================================================================
    #                    COMPUTE ARRAYS
    #==============================================================================
    roundof = 6 # verts will not give precisely the value but rather 0.1000009 (in order to make 
                # np.where () work we have to ceil roughly in the dimension of the gridsize)
    N = [int((b[i]-a[i])/gridsize[i]) for i in range(dim)]
#    L = [N[i]-1   for i in range(dim)]
    def ha(x):
        return np.array([gridsize[0]*x[0], gridsize[1]*x[1]])
    
    a_i = [a[0]-eps_i, a[1]-eps_i]
    b_i = np.around([b[0]+eps_i, b[1]+eps_i], decimals = roundof)
    N_i = [int(np.around((b_i[i]-a_i[i])/gridsize[i])) + 1 for i in range(dim)]
    
    #-----------------------------------------------------------------------------#
    """ VERTS """
    def fun(k):
        return np.around(np.array(a_i) +  ha(iE(k,N_i)), decimals = roundof)
    pool = Pool(processes=num_cores) 
    verts = np.array(pool.map(fun,range(np.prod(N_i)) ))# np.array([  for k in range(np.prod(N_i))])
    pool.close()
    pool.join()
    pool.clear()
#    print verts
    #-----------------------------------------------------------------------------#
    """ OMEGA """#+  ha(iE(0,N)
    k_0 =  np.where(np.all(verts[:,0:2]==np.around(np.array(a) , decimals = roundof).tolist(),axis=1))[0][0] 
    omega = [k_0 + k + j*N_i[1] for k in range(N[1]) for j in range(N[0])]# pool.map(om, range(num_omega))    
    ####-----------------------------------------------------------------------------#
    """ OMEGA_I """
    omega_i = list(set(range(len(verts))) - set(omega) -set(np.where(verts[:,0]== b_i[0])[0])  -set(np.where(verts[:,1]== b_i[1])[0]  ) )
    ###-----------------------------------------------------------------------------#
    #""" NODES """
    nodes = omega  
    ##-----------------------------------------------------------------------------#
    """ OMEGA """
    omega = np.zeros((2*len(nodes), 3), dtype = int)
    
    for j in range(len(nodes)):
        k = nodes[j]
        omega[2*j,    ] = np.array([k, k+ N_i[1]+1,k + 1])#  clockwise
        omega[2*j + 1,    ] = np.array([k, k + N_i[1], k+ N_i[1]+1])#  clockwise
    
    """ OMEGA_i """
    Omega_i = np.zeros((2*len(omega_i), 3), dtype = int)
    
    for j in range(len(omega_i)):
        k = omega_i[j]
        Omega_i[2*j,    ] = np.array([k, k+ N_i[1]+1,k + 1 ]) # clockwise
        Omega_i[2*j +1, ] = np.array([k,k + N_i[1], k+ N_i[1]+1])#clockwise
    
    omega_i = Omega_i
    
    ##-----------------------------------------------------------------------------#
    """ BOUNDARY """
    boundary1 = [k_0 + kk  for kk in range(N[0])]
    boundary2 = [k_0 + N[1]*N_i[0] + kkk  for kkk in range(N[0]+1)]
    boundary3 = [k_0 + j*N_i[0] for j in range(N[1])]
    boundary4 = [k_0 +N[0]+ j*N_i[0] for j in range(N[1])]
    
    boundary = np.unique( boundary1 + boundary2 + boundary3 + boundary4)

    ##-----------------------------------------------------------------------------#
    
    aux = np.zeros((len(omega),4), dtype=int)    
    aux[:,1:] = omega  
    aux[:,0] = 1*np.ones(len(omega), dtype = int)
    omega = aux
    
    aux = np.zeros((len(omega_i),4), dtype=int)    
    aux[:,1:] = omega_i  
    aux[:,0] = 2*np.ones(len(omega_i), dtype = int)
    omega_i = aux
    
    ##-----------------------------------------------------------------------------#
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))

    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:,1:4].reshape(3*num_omega))

    ##-----------------------------------------------------------------------------#
    """ PERMUTE VERTS TO SORT OMEGA_I TO THE END and ADAPT TRIANGLES etc """ 
    nodes_inner = list(set(nodes) - set(boundary))
    nodes_rest = list(set(range(len(verts))) - set(nodes_inner) - set(boundary))
    verts = np.vstack((verts[nodes_inner], verts[boundary], verts[nodes_rest]))
    
    new_order = nodes_inner + list(boundary) + nodes_rest
    def permutation(i):
        return new_order.index(i)
    
    triangles_old = triangles
    triangles = np.zeros(np.shape(triangles_old), dtype = int)
    triangles[:,0] = triangles_old[:,0]
    
    for i in range(np.shape(triangles_old)[0]):
        for j in range(1,4):
            triangles[i,j] = int(permutation(triangles_old[i,j] ))

    omega = triangles[np.where(triangles[:,0] != 2)[0]]
    boundary = range(len(nodes_inner), len(nodes_inner)+len(boundary))
    nodes = list(range(len(nodes_inner)))+list(boundary)

    #--------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""      
    bdry_nodes = [nodes.index(i) for i in boundary] 
    
    #--------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""      
    interface = []#np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()     

    #------------------------------------------------------------------------------  
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, boundary, nodes, bdry_nodes, interface]    
    mesh = Mesh_slim(proc_mesh_data)
    #------------------------------------------------------------------------------       
    return mesh, proc_mesh_data





       
def prepare_mesh_slim(verts, lines, triangles, eps):
    verts = verts[:,0:2]
    # note that eps is determined already in the mesh, however: for 
    # homogeneous dirichtlet constraints it simply has to be smaller than
    # the pre-determined epsilon

    """ sort triangles """
    # we sort by labels
    # this is important such that omega[k,] = triangle[k,]
    # As a consequence we can use omega for the list 'support'
    triangles = triangles[triangles[:,0].argsort()]

    def norm(x):
        return np.max(np.abs(x), axis= 0)
    #--------------------------------------------------------------------------
    """ DECRYPTING LABELS: 1 = shape; 2 = omega\shape; 3 = omega_i """   
    omega = triangles[np.where(triangles[:,0] != 3)[0]]    

    #--------------------------------------------------------------------------
    """ SHAPE INTERFACE (subset 0:len(verts) )"""      
    interface = np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()   
    
    #--------------------------------------------------------------------------
    """ NODES (subset of verts)"""   

    omega1 = triangles[np.where(triangles[:,0] == 1)] 
    omega2 = triangles[np.where(triangles[:,0] == 2)] 
    nodes1 = list(np.unique(omega1[:,1:4]))
    nodes2 = list(np.unique(omega2[:,1:4]))

    G_N = []
    for i in interface:
        G_N += np.where(norm((verts-np.repeat(verts[i][:,np.newaxis], len(verts), axis = 1).transpose()).transpose())<=eps)[0].tolist()

    G_N1 = list( (set(nodes1)&set(G_N)) )
    G_N2 = list(  (set(nodes2)&set(G_N)) )
    
    nodes1 = list(set(nodes1) - set(G_N1) )
    nodes2 = list(set(nodes2) - set(G_N2) )
    G_N1 = list( (set(G_N1)-set(interface)) )
    G_N2 = list(  (set(G_N2)-set(interface)) )
    nodes = list(nodes1)  + list(G_N1)+ interface + list(G_N2) + list(nodes2) 
    nodes = np.array(nodes)

    #--------------------------------------------------------------------------
    """ BOUNDARY_verts (subset of verts); label = 9"""
    boundary = np.unique(lines[lines[:,0] == 9][:,1:3])
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
        triangles[:,0] = triangles_old[:,0]
        
        for i in range(np.shape(triangles_old)[0]):
            for j in range(1,4):
                triangles[i,j] = int(permutation(triangles_old[i,j] ))
        
    
        interface_old= interface
        interface = np.zeros(len(interface_old), dtype = int)
        for i in range(len(interface_old)):
            interface[i] = permutation(interface_old[i])
        
        omega = triangles[np.where(triangles[:,0] != 3)[0]]
        # plot omega
    #    for i in range(len(omega)):
    #        plt.gca().add_patch(plt.Polygon(verts[omega[i,1:]], closed=True, fill = True, color = 'blue', alpha = 0.2)) 
        
        boundary = np.array(range(len(nodes_inner), len(nodes_inner)+len(boundary)))
        nodes = np.array(range(len(nodes_inner)+len(boundary)))#+list(boundary)
#     PLOT verts
#    plt.plot(verts[nodes][:,0], verts[nodes][:,1], 'bx')
#    plt.plot(verts[interface][:,0], verts[interface][:,1], 'yd')
#    plt.plot(verts[len(nodes):][:,0], verts[len(nodes):][:,1], 'ro')
    #--------------------------------------------------------------------------
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""      
    bdry_nodes = [nodes.tolist().index(i) for i in boundary] 

    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts, boundary, nodes,bdry_nodes,interface]    
    mesh = Mesh_slim(proc_mesh_data)
    #------------------------------------------------------------------------------       
    return mesh, proc_mesh_data


class Mesh_1d:

    # object with all processed mesh data
    def __init__(self, proc_mesh_data):

        self.triangles = proc_mesh_data[0]
        self.omega = proc_mesh_data[1]
        self.verts = proc_mesh_data[2]
        self.boundary_verts = proc_mesh_data[3]
        self.nodes = proc_mesh_data[4]
        self.diam = proc_mesh_data[5]
        self.hash_table_approx = proc_mesh_data[6]
        self.bary = proc_mesh_data[7]
        self.boundary = proc_mesh_data[8]
        self.hash_table_bary = proc_mesh_data[9]
        self.shape_interface = proc_mesh_data[10]
        self.h = proc_mesh_data[11]


def prepare_mesh_reg_1d(h, delta, num_cores, **kwargs):

    interface_point = kwargs.get('interface_point', 'x')
    if interface_point == 'x':
        interface_point =  2*h
        labels = [1,1,3]
    else:
        labels = [1,2,3]
    
    if not (1/h).is_integer() or not (delta/h).is_integer():
        print()
        print('\n    ---- WARNING ----     \n')
        print('(1/h and delta/h have to be an integer !!!')
        print('\n    ---- WARNING ----     \n')
        print()

    if delta < h:
        delta_i = h
    else:
        delta_i = delta
        
    def norm(x):
        return np.abs(x)
    a = [0.]
    b = [1.]
    gridsize = [h]
    diam = h
    dim = 1 
    def E(z, L):
        p = [np.prod([L[k] for k in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        summe = 0
        for i in range(dim):
            summe = summe + z[i] * p[i]  
        return int(summe)
    def iE(k, L):
        p = [np.prod([L[l] for l in range(j+1,dim)]) for j in range(dim)] + [np.prod(L)]
        x = np.zeros(dim)
        for i in range(dim):
            x[i] = k//(int(p[i])) - (k//(int(p[i-1]))) * L[i]
        return x[0]
    
    roundof = 7 # verts will not give precisely the value but rather 0.1000009 (in order to make 
                # np.where () work we have to round of, roughly in the dimension of the gridsize)
    def ha(x):
        return np.array([gridsize[0]*x])
    
    a_i = [a[0]- delta_i]
    b_i = [b[0] + delta_i]
    N_i = [int(np.around((b_i[i]-a_i[i])/gridsize[i])) + 1 for i in range(dim)]
    
    #-----------------------------------------------------------------------------#
    """ VERTS """
    def fun(k):
        return (a_i[0] +  ha(iE(k,N_i)))[0]#, decimals = roundof)
    #    pool = Pool(processes=num_cores) 
    verts = np.around(np.array(map(fun, range(np.prod(N_i)) )), decimals = roundof)# np.array([  for k in range(np.prod(N_i))])
    #    pool.close()
    #    pool.join()
    #    pool.clear()
    #    print verts
    
    verts = np.sort(np.unique(np.append(verts, [interface_point]))) 
    # note: if we dont want interface, then interface = 2h, in this case no
    # new point is added to the grid
    """ OMEGA """
    k_0 =  np.where(verts==np.around(np.array(a) , decimals = roundof))[0][0] 
    k_i = np.where(verts==np.around(np.array(interface_point) , decimals = roundof))[0][0] 
    k_1 =  np.where(verts==np.around(np.array(b) , decimals = roundof))[0][0]
    omega = np.zeros((k_1-k_0, 3), dtype = int)
    for j in range(k_i - k_0):
        omega[j, ] = np.array([labels[0], k_0 + j, k_0 + j + 1])#  clockwise
    for j in range(k_1 - k_i):
        omega[k_i-k_0 + j, ] = np.array([labels[1], k_i + j, k_i + j + 1])#  clockwise
    """ OMEGA_i """
    omega_i = np.zeros((2*len(range(k_0)), 3), dtype = int)
    for j in range(k_0):
        omega_i[j,    ] = np.array([labels[2], j, j+1]) # clockwise
        omega_i[k_0 + j,    ] = np.array([labels[2], k_1 + j, k_1 +j+1])
    """ BOUNDARY """
    boundary = [k_0, k_1]
    """ TRIANGLES """
    triangles = np.vstack((omega, omega_i))
    """ NODES """
    num_omega = np.shape(omega)[0]
    nodes = np.unique(omega[:,1:3].reshape(2*num_omega))
    bary = (verts[triangles[:,1]] + verts[triangles[:,2]] ) /2.
    """ HASH TABLE_approx (= hash table bary without puffer)"""      
    hash_table_approx = []#[np.where(np.abs((bary-np.repeat(bary[i], len(bary))))<=(delta))[0].tolist() for i in range(len(bary))]
    """ HASH TABLE (subset 0:len(bary))"""
    hash_table_bary = []#[np.where(norm((bary-np.repeat(bary[i], len(bary)).transpose()).transpose())<=(delta+ diam))[0].tolist() for i in range(len(bary))]
    """ BOUNDARY_nodes (subset 0:len(nodes) )"""      
    bdry_nodes = [nodes.tolist().index(i) for i in boundary]
    """ SHAPE INTERFACE (subset 0:len(verts) )"""      
    interface = []#np.unique(lines[np.where(lines[:,0]==12)][:,1:]).tolist()
    """ BUILDING MESH CLASS """
    proc_mesh_data = [triangles, omega, verts,  boundary, nodes,  diam,  hash_table_approx, bary, bdry_nodes, hash_table_bary, interface, h]    
    mesh = Mesh_1d(proc_mesh_data)
    #------------------------------------------------------------------------------       
    return mesh, proc_mesh_data
   

"""-------------------------------------------------------------------------"""
"""                     END PREPARE MESH                                    """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#
"""-------------------------------------------------------------------------"""
"""                   ASSEMBLY FUNCTIONS                                    """
"""-------------------------------------------------------------------------"""
#==============================================================================
#                          QUADRATURE
#==============================================================================
if quad_order_outer == 'test':
    P = P_test#np.array([ [ 1./3., 1./3.]]).transpose()
    weights=  np.array([ 0.5])
    
elif quad_order_outer == 8:
    print('outer quad = 16 points')
    P = np.array([ [ 0.33333333,  0.33333333],
                   [ 0.45929259,  0.45929259],
                   [ 0.45929259,  0.08141482],
                   [ 0.08141482,  0.45929259],
                   [ 0.17056931,  0.17056931],
                   [ 0.17056931,  0.65886138],
                   [ 0.65886138,  0.17056931],
                   [ 0.05054723,  0.05054723],
                   [ 0.05054723,  0.89890554],
                   [ 0.89890554,  0.05054723],
                   [ 0.26311283,  0.72849239],
                   [ 0.72849239,  0.00839478],
                   [ 0.00839478,  0.26311283],
                   [ 0.72849239,  0.26311283],
                   [ 0.26311283,  0.00839478],
                   [ 0.00839478,  0.72849239]]).transpose()
                   
    weights=  np.array([ 0.14431560767779
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443])

elif quad_order_outer == 5:
    print('outer quad = 7 Gaussian points')
    P = np.array([[0.33333333333333,    0.33333333333333],
                  [0.47014206410511,    0.47014206410511],
                  [0.47014206410511,    0.05971587178977],
                  [0.05971587178977,    0.47014206410511],
                  [0.10128650732346,    0.10128650732346],
                  [0.10128650732346,    0.79742698535309],
                  [0.79742698535309,    0.10128650732346] ]).transpose()
    
    weights = np.array([0.22500000000000,
                        0.13239415278851,
                        0.13239415278851,
                        0.13239415278851,
                        0.12593918054483,
                        0.12593918054483,
                        0.12593918054483])
        

elif quad_order_outer == 3:            
    P = np.array([[ 1./3.,  1./3.],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()
    
    weights = np.array([-27./48., 25./48., 25./48., 25./48.])

elif quad_order_outer == 2: 
    P = np.array([[1./6.,    1./6.],
                  [1./6.,    2./3.],  
                  [2./3.,    1./6.]  ]).transpose()
    weights = 1./3 * np.ones(3)

elif quad_order_outer == 'GL-5-12':
    P = np.array([[1./21. * (7 - np.sqrt(7))    , 1./21. * (7 - np.sqrt(7))],
                  [1./21. * (7 - np.sqrt(7))    , 1- 2./21. * (7 - np.sqrt(7))  ],
                  [1- 2./21. * (7 - np.sqrt(7)) , 1./21. * (7 - np.sqrt(7))],
                  [1./42.*(21.-np.sqrt(21*(4*np.sqrt(7)-7))), 0. ],
                  [0. ,1.-1./42.*(21.-np.sqrt(21*(4*np.sqrt(7)-7)))],
                  [1.-1./42.*(21.-np.sqrt(21*(4*np.sqrt(7)-7))), 1./42.*(21.-np.sqrt(21*(4*np.sqrt(7)-7)))],
                  [1./42.*(21.+np.sqrt(21*(4*np.sqrt(7)-7))), 0. ],
                  [0. ,1.-1./42.*(21.+np.sqrt(21*(4*np.sqrt(7)-7)))],
                  [1.-1./42.*(21.+np.sqrt(21*(4*np.sqrt(7)-7))), 1./42.*(21.+np.sqrt(21*(4*np.sqrt(7)-7)))],
                  [0., 0.],
                  [1., 0.],
                  [0., 1.] ]).transpose()



    weights = 2*np.array([7./720. * (14. - np.sqrt(7.)), 7./720. * (14. - np.sqrt(7.)), 7./720. * (14. - np.sqrt(7.)),
                        1./720. * (7. + 4.*np.sqrt(7.)), 1./720. * (7. + 4.*np.sqrt(7.)), 1./720. * (7. + 4.*np.sqrt(7.)),
                        1./720. * (7. + 4.*np.sqrt(7.)), 1./720. * (7. + 4.*np.sqrt(7.)), 1./720. * (7. + 4.*np.sqrt(7.)),
                        1./720. * (8.-np.sqrt(7.)),1./720. * (8.-np.sqrt(7.)),1./720. * (8.-np.sqrt(7.))])


else:
    print('outer quad = 7 points incl. vertices')
    # 7-points rule including vertices of triangle
    P = np.array([[0.,0.],
               [0.5, 0.],
               [1., 0.],
               [1./3., 1./3.],
               [0.,0.5],
               [0.5, 0.5],
               [0.,1.] ]).transpose()

    weights = np.array([1./20., 4./30., 1./20., 9./20., 4./30., 4./30., 1./20.])

#===================        FOR INNER INTEGRAL     ============================
if quad_order_inner == 8:
    print('inner quad = 16 points ')
    P2 = np.array([ [ 0.33333333,  0.33333333],
                   [ 0.45929259,  0.45929259],
                   [ 0.45929259,  0.08141482],
                   [ 0.08141482,  0.45929259],
                   [ 0.17056931,  0.17056931],
                   [ 0.17056931,  0.65886138],
                   [ 0.65886138,  0.17056931],
                   [ 0.05054723,  0.05054723],
                   [ 0.05054723,  0.89890554],
                   [ 0.89890554,  0.05054723],
                   [ 0.26311283,  0.72849239],
                   [ 0.72849239,  0.00839478],
                   [ 0.00839478,  0.26311283],
                   [ 0.72849239,  0.26311283],
                   [ 0.26311283,  0.00839478],
                   [ 0.00839478,  0.72849239]]).transpose()
                   
    weights2=  np.array([ 0.14431560767779
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.09509163426728
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.10321737053472
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.03245849762320
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443
                       , 0.02723031417443])

elif quad_order_inner == 5:
    P2 = np.array([[0.33333333333333,    0.33333333333333],
                  [0.47014206410511,    0.47014206410511],
                  [0.47014206410511,    0.05971587178977],
                  [0.05971587178977,    0.47014206410511],
                  [0.10128650732346,    0.10128650732346],
                  [0.10128650732346,    0.79742698535309],
                  [0.79742698535309,    0.10128650732346] ]).transpose()
    
    weights2 = np.array([0.22500000000000,
                        0.13239415278851,
                        0.13239415278851,
                        0.13239415278851,
                        0.12593918054483,
                        0.12593918054483,
                        0.12593918054483])

elif quad_order_inner == 3:    
    print("inner quad = 4 points Gaussian")        
    P2 = np.array([[ 1./3.,  1./3.],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()
    
    weights2 = np.array([-27./48., 25./48., 25./48., 25./48.])

elif quad_order_inner == 2: 
    print("inner quad = 3 points Gaussian") 
    P2 = np.array([[1./6.,    1./6.],
                  [1./6.,    2./3.],  
                  [2./3.,    1./6.]  ]).transpose()
    weights2 = 1./3 * np.ones(3)        

else:
    print("inner quad = 1 point - barycenter + triangle vol") 
    P2 = np.array([[1./3., 1./3.]]).transpose()
    weights2 = np.array([1.0])


n = np.shape(P)[1]
n2 = np.shape(P2)[1]

X = np.tile(P,n)
Y = np.repeat(P,n, axis=1)
W = 0.25 * np.array([weights[i]*weights[r] for i in range(n) for r in range(n)])

def BASIS(v):
    return np.array([ 1. - v[0] - v[1], v[0], v[1]])

PSI = BASIS(P)
PSI_2 = BASIS(P2)

PSI_X = BASIS(X)
PSI_Y = BASIS(Y)

PSI_P = BASIS(P)
weights = 0.5 * weights
weights2 = 0.5 * weights2

def basis_0(v):
    return 1. - v[0] - v[1]
def basis_1(v):
    return v[0]
def basis_2(v):
    return v[1]
def basis_3(v):
    return v[1]-v[1]

basis = [basis_0, basis_1, basis_2, basis_3]


""" TEST FOR MINIMUM PRECISION"""
P_outer = P
weights_outer = weights
P_inner = P2
weights_inner = weights2
n_outer, n_inner = n, n2
X_outer, Y_inner = np.tile(P_outer,n_inner), np.repeat(P_inner,n_outer, axis=1)
W_minprec = np.array([weights_outer[r]*weights_inner[i] for i in range(n_inner) for r in range(n_outer)])
PSI_outer = BASIS(X_outer)
PSI_inner = BASIS(Y_inner)
#-----------------------------------------------------------------------------


"""#===========================================================================
                              QUADRATURE 1d
#==========================================================================="""
## quadpoints and weights for reference interval [0,1]
points1_1d = np.polynomial.legendre.leggauss(n1_1d)[0]    
weights1_1d = np.polynomial.legendre.leggauss(n1_1d)[1] 
p1_1d = 0.5 + 0.5 * points1_1d

points2_1d = np.polynomial.legendre.leggauss(n2_1d)[0]    
weights2_1d = np.polynomial.legendre.leggauss(n2_1d)[1] 
p2_1d = 0.5 + 0.5 * points2_1d

n1_1d = len(p1_1d)
n2_1d = len(p2_1d)

X_1d = np.tile(p1_1d,n1_1d)
Y_1d = np.repeat(p1_1d,n1_1d)
W_1d = 0.25 * np.array([weights1_1d[i]*weights1_1d[r] for i in range(n1_1d) for r in range(n1_1d)])

def BASIS_1d(v):
    return np.array([ 1-v, v])

PSI_1d = BASIS_1d(p1_1d)
PSI_2_1d = BASIS_1d(p2_1d)

PSI_X_1d = BASIS_1d(X_1d)
PSI_Y_1d = BASIS_1d(Y_1d)

PSI_P_1d = BASIS_1d(p1_1d)
weights1_1d = 0.5 * weights1_1d
weights2_1d = 0.5 * weights2_1d

def psi0(v):
    return 1.-v          
def psi1(v):
    return v
def psi2(v):
    return (v - v)

basis_1d = [psi0, psi1, psi2]

#==============================================================================
"""                 CHOOSE PARAMETER FOR ADAPTIVE RULE                      """
#==============================================================================
T_ref = [[0., 0.], [1., 0.], [0., 1.]]
""" plot quad_points """
#plt.gca().add_patch(plt.Polygon(T_ref , closed=True, fill = False))  
#for p in P.transpose():
#    plt.plot(p[0], p[1], 'rx')
#plt.axis('equal')
"""RONALD COOLS and ANN HAEGEMANS  5-7 embedded rule """    
P_radon = np.array([[ 1./3.,  1./3.],
              [ (6+np.sqrt(15))/21.,            (6+np.sqrt(15))/21. ],
              [ (6+np.sqrt(15))/21.,            (9. - 2. * np.sqrt(15))/21. ],
              [ (9. - 2. * np.sqrt(15))/21. ,   (6+np.sqrt(15))/21.  ], 
              [ (6-np.sqrt(15))/21.,            (6-np.sqrt(15))/21. ],
              [ (6-np.sqrt(15))/21.,            (9.+ 2. * np.sqrt(15))/21. ],
              [ (9.+ 2. * np.sqrt(15))/21. ,    (6-np.sqrt(15))/21.],
              [1./9.,                           1./9.],
              [1./9.,                           7./9.],
              [7./9.,                           1./9.],
              [1./9.,                           (4 + np.sqrt(15))/9.],
              [1./9.,                           (4 - np.sqrt(15))/9.],
              [(4 + np.sqrt(15))/9.,            1./9.],
              [(4 + np.sqrt(15))/9.,            (4 - np.sqrt(15))/9.],  
              [(4 - np.sqrt(15))/9.,            1./9.],
              [(4 - np.sqrt(15))/9.,            (4+ np.sqrt(15))/9.]])


weights7 = np.array([1773./17920., 
                      1./3. * (13558.*np.sqrt(15) - 37801.)/89600., 1./3. * (13558.*np.sqrt(15) - 37801.)/89600.,1./3. * (13558.*np.sqrt(15) - 37801.)/89600.,
                      1./3. * (-13558.*np.sqrt(15) - 37801.)/89600., 1./3. * (-13558.*np.sqrt(15) - 37801.)/89600., 1./3. * (-13558.*np.sqrt(15) - 37801.)/89600., 
                      1./3. * 19683./17920, 1./3. * 19683./17920,1./3. * 19683./17920,
                      1./6. * 6561./44800.,1./6. * 6561./44800.,1./6. * 6561./44800.,1./6. * 6561./44800.,1./6. * 6561./44800.,1./6. * 6561./44800.])
weights5 = 0.5 * np.array([9./40., 1./3. * (155. + np.sqrt(15))/400., 1./3. * (155. + np.sqrt(15))/400., 1./3. * (155. + np.sqrt(15))/400., 1./3. * (155. - np.sqrt(15))/400., 1./3. * (155. - np.sqrt(15))/400., 1./3. * (155. - np.sqrt(15))/400.,0,0,0,0,0,0,0,0,0])    

def tri_adapt(f, T, **kwargs):
    tol1_Radon_get = kwargs.get('tol1_Radon_get', tol1_Radon)
    tol2_Radon_get = kwargs.get('tol2_Radon_get', tol2_Radon)
#    plot = kwargs.get('plot', 0)

    T = np.array(T)

    M = np.array([T[1] - T[0], T[2] - T[0]]).transpose()
    def trans(y):   
#        plt.plot((T[0] + M.dot(y) )[0], (T[0] + M.dot(y) )[1], 'rx')
        return T[0] + M.dot(y) 
        
    det = abs(np.linalg.det(np.array([T[1] - T[0], T[2] - T[0]]).transpose()))

    I = np.array(map(lambda i: f(trans(P_radon[i])), range(np.shape(P_radon)[0]) )) 

    g_high = det * (I * weights7).sum()
    g_low =  det * (I * weights5).sum()
    diam = max(np.linalg.norm(T[0]-T[1]),np.linalg.norm(T[0]-T[2]), np.linalg.norm(T[2]-T[1]))
    
#    if plot:
#        plt.gca().add_patch(plt.Polygon(T , closed=True, fill = False))  
    
    if abs(g_high - g_low)/abs(g_low+1e-9) > tol1_Radon_get and diam>tol2_Radon_get:#
        new1 = list(0.5 * (np.array(T[0])+np.array(T[2])))
        new2 = list(0.5 * (np.array(T[0])+np.array(T[1])))
        new3 = list(0.5 * (np.array(T[1])+np.array(T[2])))

        T1 = [T[0], new2, new1]
        T2 = [new1, new2, new3]
        T3 = [new1, new3, T[2]]
        T4 = [new2, T[1], new3]
        
        return   tri_adapt(f, T1, tol2_Radon_get = tol2_Radon_get, tol1_Radon_get = tol1_Radon_get)  \
               + tri_adapt(f, T2, tol2_Radon_get = tol2_Radon_get, tol1_Radon_get = tol1_Radon_get)  \
               + tri_adapt(f, T3, tol2_Radon_get = tol2_Radon_get, tol1_Radon_get = tol1_Radon_get)  \
               + tri_adapt(f, T4, tol2_Radon_get = tol2_Radon_get, tol1_Radon_get = tol1_Radon_get) 
               
    return g_high    
  
"""-------------------------------------------------------------------------"""
"""                   ASSEMBLY FUNCTIONS                                    """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#
#=============================================================================#"""-------------------------------------------------------------------------"""
"""-------------------------------------------------------------------------"""
"""                     ASSEMBLY LAPLACE STIFFNESS MATRIX                   """
"""-------------------------------------------------------------------------"""
def Laplace(mesh,num_cores, **kwargs):
    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    triangles = mesh.elements
    nodes = mesh.nodes
    support = mesh.support
    nodes = list(nodes)
    gradient = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]
    
    def entry(k,j):
        res = 0.
        Skj = list(set(support[k]) & set(support[j]))
        for s in Skj:
            if triangles[s,0] != 3:
                S = triangles[s,1:].tolist()
                a, b= S.index(nodes[k]),S.index(nodes[j])
                T = [verts[S[i]] for i in range(3)]    
                Mat_k = np.array( [T[1] - T[0],T[2] - T[0] ]).transpose()
                det_k = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
                iMat_k = 1./det_k * np.array([ [Mat_k[1,1], -Mat_k[0,1]], [-Mat_k[1,0], Mat_k[0,0]]  ])
                res += 0.5 * abs(det_k) * gradient[a].dot(iMat_k.dot(iMat_k.transpose().dot(gradient[b])))
        return res

    L = np.zeros((len(nodes), len(nodes)))
    def fun_wrapper(indices):
        return entry(*indices)
    def fun(k):
        return np.array(map(fun_wrapper, [(k, j) for j in mesh.nhd[k]]))
    AUX = list(map(fun, range(len(nodes))))
    for k in xrange(len(nodes)):
        L[k, mesh.nhd[k]] = AUX[k]
    L = np.tril(L)
    L = L.transpose() + L - np.diag(L.diagonal())    
    
    return ss.csr_matrix(L)

def Laplace_1d(mesh, diff_coeff):

    verts = mesh.vertices
#    nodes = mesh.nodes
    gradient = [np.array([-1]), np.array([1])]

    L = ss.lil_matrix((len(mesh.vertices), len(mesh.vertices)), dtype = float)

    for i in range(len(mesh.elements)):
        label_i = mesh.elements[i, 0]
        T_i = mesh.elements[i, 1:].tolist()
        T_i_v = verts[T_i]#np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
        det_T_i = T_i_v[1] - T_i_v[0]
        iMat_i = 1./det_T_i
        for a in range(2):
            for b in range(2):
                kk, jj = T_i[a], T_i[b]
                L[kk, jj] += abs(det_T_i) * diff_coeff[label_i-1] * gradient[a] * iMat_i * iMat_i * gradient[b]

    return L.tocsr()

    
def Laplace_para(mesh, diff_coeff, num_cores, **kwargs):
    """
    actually not worth it, because:
    1) if matrix small: serial pretty fast, too much overhead elsewise
    2) if matrix is large: way of parallizing takes way to much RAM
    """
    
    deform = kwargs.get('deform', 0)
    # print(deform)
    # print('verts1', mesh.verts[50])
    verts = mesh.vertices + deform
    # print('verts2', verts[50])
    nodes = mesh.nodes
    gradient = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]

    # randomly shuffle indices: 
    # For areas where the mesh is fine, hash_i might be much larger 
    liste = range(len(mesh.omega))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, 1)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)

        for i in pieces[m].tolist():
            label_i = mesh.omega[i,0]
            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = verts[T_i]#np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
            # if i == 60:
            #     print(T_i_v)
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
            for a in range(3):
                for b in range(3):
                    kk, jj = np.where(nodes == T_i[a])[0][0], np.where(nodes == T_i[b])[0][0]
                    L[kk, jj] += diff_coeff[label_i-1] * 0.5 * abs(det_T_i) * gradient[a].dot(iMat_i.dot(iMat_i.transpose().dot(gradient[b])))

        return L

    Ls = list(map(aux, range(1)))#Pool(num_cores).

    L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
    for i in range(1):
        # print(Ls[i].A)
        L += Ls[i]

    return L.tocsr()
"""-------------------------------------------------------------------------"""
"""                 END ASSEMBLY LAPLACE STIFFNESS MATRIX                   """
"""-------------------------------------------------------------------------"""

#=============================================================================#
#=============================================================================#
#=============================================================================#
"""-------------------------------------------------------------------------"""
"""                       ASSEMBLY OF RHS                                   """
"""-------------------------------------------------------------------------"""
def source_term_discon(mesh, source, **kwargs):
    """
    source: function determining the source term
    omega: triangles in omega
    verts: associated nodes
    """
    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform

#    f0 = source[0]
#    f1 = source[1]

    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    n = np.shape(P)[1]
    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    num_nodes = len(mesh.nodes)
    res = np.zeros(num_nodes)

    labels_inner_domain = np.sort(np.unique(mesh.elements[:, 0]))[0:-1].tolist()

    for label in labels_inner_domain:
        omega = mesh.elements[np.where(mesh.elements[:, 0] == label)[0]]

        num_omega = len(omega)

        nodes = list(mesh.nodes)
        for i in range(num_omega):
            T_i = omega[i][1:].tolist()
            T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
            for k in range(3):
                Mat_k = np.array( [T_i_v[(k+1)%3] - T_i_v[k],T_i_v[(k-1)%3] - T_i_v[k] ]).transpose()
                det_k = abs(Mat_k[0,0] * Mat_k[1,1] - Mat_k[1,0] * Mat_k[0,1] )
                B = source[label-1] * np.ones(n)
                res[nodes.index(T_i[k])] += det_k * (B * PSI[k] * weights).sum()

    return res

"""-------------------------------------------------------------------------"""
"""                   END ASSEMBLY OF RHS                                   """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#
"""-------------------------------------------------------------------------"""
"""                       ASSEMBLY OF MASSMATRIX                            """
"""-------------------------------------------------------------------------"""
def mass_matrix2(mesh, **kwargs):

    deform = kwargs.get('deform', 0)

    verts = mesh.vertices + deform

    nodes = np.array(mesh.nodes)

    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    num_nodes = len(mesh.nodes)


    L = ss.lil_matrix((num_nodes,num_nodes), dtype = float)

    for i in range(len(mesh.omega)):
        T_i = mesh.omega[i, 1:].tolist()
        T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
        Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
        det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] )
        for a in range(3):
            for b in range(3):
                L[np.where(nodes == T_i[a])[0][0],np.where(nodes == T_i[b])[0][0]] += det_T_i * (PSI[a] * PSI[b] * weights).sum()

    return L.tocsr()

def mass_matrix2_DG(mesh, **kwargs):

    deform = kwargs.get('deform', 0)

    verts = mesh.vertices + deform

    nodes = np.array(mesh.nodes)

    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    num_nodes = len(mesh.nodes)

    L = ss.lil_matrix((3*len(mesh.omega),3*len(mesh.omega)), dtype = float)

    for i in range(len(mesh.omega)):
        T_i = mesh.omega[i, 1:].tolist()
        T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
        Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
        det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] )
        for a in range(3):
            for b in range(3):
                L[3*i + a, 3*i +b] = det_T_i * (PSI[a] * PSI[b] * weights).sum()

    return L.tocsr()

def mass_matrix_full(mesh, **kwargs):

    deform = kwargs.get('deform', 0)

    verts = mesh.vertices + deform

    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    # P = np.array([[0.33333333, 0.33333333],
    #               [0.45929259, 0.45929259],
    #               [0.45929259, 0.08141482],
    #               [0.08141482, 0.45929259],
    #               [0.17056931, 0.17056931],
    #               [0.17056931, 0.65886138],
    #               [0.65886138, 0.17056931],
    #               [0.05054723, 0.05054723],
    #               [0.05054723, 0.89890554],
    #               [0.89890554, 0.05054723],
    #               [0.26311283, 0.72849239],
    #               [0.72849239, 0.00839478],
    #               [0.00839478, 0.26311283],
    #               [0.72849239, 0.26311283],
    #               [0.26311283, 0.00839478],
    #               [0.00839478, 0.72849239]]).transpose()
    #
    # weights = np.array([0.14431560767779
    #                        , 0.09509163426728
    #                        , 0.09509163426728
    #                        , 0.09509163426728
    #                        , 0.10321737053472
    #                        , 0.10321737053472
    #                        , 0.10321737053472
    #                        , 0.03245849762320
    #                        , 0.03245849762320
    #                        , 0.03245849762320
    #                        , 0.02723031417443
    #                        , 0.02723031417443
    #                        , 0.02723031417443
    #                        , 0.02723031417443
    #                        , 0.02723031417443
    #                        , 0.02723031417443])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

    for i in range(len(mesh.elements)):
        T_i = mesh.elements[i, 1:].tolist()
        T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
        Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
        det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] )
        for a in range(3):
            for b in range(3):
                L[T_i[a], T_i[b]] += det_T_i * (PSI[a] * PSI[b] * weights).sum()

    return L.tocsr()

def mass_matrix_full_DG(mesh, **kwargs):

    deform = kwargs.get('deform', 0)

    verts = mesh.vertices + deform

    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    L = ss.lil_matrix((3 * len(mesh.elements), 3 * len(mesh.elements)), dtype = float)

    for i in range(len(mesh.elements)):
        T_i = mesh.elements[i, 1:].tolist()
        T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
        Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
        det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] )
        for a in range(3):
            for b in range(3):
                L[3*i + a, 3*i + b] += det_T_i * (PSI[a] * PSI[b] * weights).sum()

    return L.tocsr()

def mass_matrix_para(mesh, num_cores, **kwargs):

    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    nodes = np.array(mesh.nodes)
    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)
    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.omega))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)

        for i in pieces[m].tolist():

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = np.array([verts[T_i[0]],verts[T_i[1]], verts[T_i[2]]])
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] )
            for a in range(3):
                for b in range(3):
                    L[np.where(nodes == T_i[a])[0][0],np.where(nodes == T_i[b])[0][0]] += det_T_i * (PSI[a] * PSI[b] * weights).sum()

        return L

    Ls = Pool(num_cores).map(aux, range(num_cores))

    L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    return L

def mass_matrix_1d(mesh):

    num_nodes = len(mesh.nodes)

    L = ss.lil_matrix((num_nodes,num_nodes), dtype = float)

    for i in range(len(mesh.omega)):
        T_i = mesh.omega[i, 1:].tolist()
        det_T_i = mesh.h
        for a in range(2):
            for b in range(2):
                L[np.where(mesh.nodes == T_i[a])[0][0],np.where(mesh.nodes == T_i[b])[0][0]] += det_T_i * (PSI_P_1d[a] * PSI_P_1d[b] * weights1_1d).sum()

    return L.tocsr()

def mass_matrix_1d_full(mesh):

    num_nodes = len(mesh.vertices)

    L = ss.lil_matrix((num_nodes,num_nodes), dtype = float)

    for i in range(len(mesh.elements)):
        T_i = mesh.elements[i, 1:].tolist()
        det_T_i = mesh.h
        for a in range(2):
            for b in range(2):
                L[T_i[a], T_i[b]] += det_T_i * (PSI_P_1d[a] * PSI_P_1d[b] * weights1_1d).sum()

    return L.tocsr()


def assembly_coupling_1d(mesh, gam, retriangulate, num_cores):

    labels_domains = np.sort(np.unique(mesh.elements[:, 0]))
    nodes = np.array(mesh.nodes)

    liste = range(len(mesh.omega))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
        for i in pieces[m].tolist():

            label_i = mesh.omega[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]

            det_T_i = abs(mesh.h)

            i_triangles = i#np.where(np.all(mesh.triangles == mesh.f[i],axis=1))[0][0]

            hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = mesh.vertices[T_j]

                def iPhi_j(y):
                    return 1./mesh.h * ( y - np.repeat(T_j_v[0], n2_1d))

                def I1(x):
                    x_trans = (T_i_v[0]+ mesh.h * x)
                    integral, integral0, integral1 = 0., 0., 0.
                    aux = np.repeat(x_trans, n2_1d)

                    def inner(tri, gam_j):
                        tri = np.array(tri)
                        h_l = tri[1] - tri[0]
                        det_l = abs(h_l)
                        def Phi_l(y):
                            return np.repeat(tri[0], n2_1d) +  h_l * y

                        GAM = det_l * gam_j(aux, Phi_l(p2_1d)) * weights2_1d

                        if label_j != labels_domains[-1]:

                            return  GAM.sum(), (basis_1d[0](iPhi_j(Phi_l(p2_1d))) * GAM ).sum(), (basis_1d[1](iPhi_j(Phi_l(p2_1d))) * GAM ).sum()
                        else:
                            return  GAM.sum(), 0., 0.

                    tri = retriangulate(x_trans, T_j_v, eps_i, mesh.h )

                    if len(tri) != 0:

                        v, v0, v1= inner(tri, gam_j)
                        integral  = v
                        integral0 = v0
                        integral1 = v1

                    return np.array([integral0, integral1, integral])

                I = np.array(map(I1, p1_1d)).transpose()


                for a in range(2):
                    kk = np.where(nodes == T_i[a])[0][0]
                    for b in range(2):
                       if label_j != labels_domains[-1]:
                           L[kk, np.where(nodes == T_j[b])[0][0]] += -det_T_i * (PSI_P_1d[a] * I[b] * weights1_1d).sum()
                       L[kk, np.where(nodes == T_i[b])[0][0]] += det_T_i * (PSI_P_1d[a] * PSI_P_1d[b] * I[2] * weights1_1d).sum()
        return 2 * L



    if num_cores == 1:
        Ls = map(aux, range(num_cores))#
    else:
        p = Pool(num_cores)
        Ls = p.map(aux, range(num_cores))
        p.close()
        p.join()
        p.clear()

    L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls

    return L

def assembly_coupling_1d_full(mesh, gam, retriangulate, num_cores):
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))

    liste = range(len(mesh.elements))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.vertices), len(mesh.vertices)), dtype = float)
        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]

            det_T_i = abs(mesh.h)

            i_triangles = i#np.where(np.all(mesh.triangles == mesh.omega[i],axis=1))[0][0]

            hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = mesh.vertices[T_j]

                def iPhi_j(y):
                    return 1./mesh.h * ( y - np.repeat(T_j_v[0], n2_1d))

                def I1(x):
                    x_trans = (T_i_v[0]+ mesh.h * x)
                    integral, integral0, integral1 = 0., 0., 0.
                    aux = np.repeat(x_trans, n2_1d)

                    def inner(tri, gam_j):
                        tri = np.array(tri)
                        h_l = tri[1] - tri[0]
                        det_l = abs(h_l)
                        def Phi_l(y):
                            return np.repeat(tri[0], n2_1d) +  h_l * y

                        GAM = det_l * gam_j(aux, Phi_l(p2_1d)) * weights2_1d

#                        if label_j != labels_domains[-1]:

                        return  GAM.sum(), (basis_1d[0](iPhi_j(Phi_l(p2_1d))) * GAM ).sum(), (basis_1d[1](iPhi_j(Phi_l(p2_1d))) * GAM ).sum()
#                        else:
#                            return  GAM.sum(), 0., 0.

                    tri = retriangulate(x_trans, T_j_v, eps_i, mesh.h )

                    if len(tri) != 0:

                        v, v0, v1= inner(tri, gam_j)
                        integral  = v
                        integral0 = v0
                        integral1 = v1

                    return np.array([integral0, integral1, integral])

                I = np.array(map(I1, p1_1d)).transpose()


                for a in range(2):
                    kk = T_i[a]
                    for b in range(2):
                       L[kk, T_j[b]] += -det_T_i * (PSI_P_1d[a] * I[b] * weights1_1d).sum()
                       L[kk, T_i[b]] += det_T_i * (PSI_P_1d[a] * PSI_P_1d[b] * I[2] * weights1_1d).sum()
        return 2 * L



    if num_cores == 1:
        Ls = map(aux, range(num_cores))#
    else:
        p = Pool(num_cores)
        Ls = p.map(aux, range(num_cores))
        p.close()
        p.join()
        p.clear()

    L = ss.lil_matrix((len(mesh.vertices), len(mesh.vertices)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls

    return L


def assembly_coupling(mesh, gam, retriangulate, Norm, num_cores, **kwargs):

    labels_domains = np.sort(np.unique(mesh.elements[:, 0]))
    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    nodes = np.array(mesh.nodes)

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.omega))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.omega[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])

            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            i_triangles = np.where(np.all(mesh.elements == mesh.omega[i], axis=1))[0][0]
            hash_i = np.where(norm_dict[Norm]((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                def check_interaction(S,T, norm, eps):
                    """checks if T is subset of the interaction domain of S """ #  is this really an improvement here due to the overhead of "check_interaction"?
                    length_of_edges = np.array([np.linalg.norm(T[0]-T[1]),np.linalg.norm(T[0]-T[2]), np.linalg.norm(T[1]-T[2])] )
                    diam = np.max(length_of_edges)
                    return np.all(np.array([ np.any(np.array([norm(S[k]-T[i]) for i in range(3)] )< (max(eps-0.5*diam, 0))) for k in range(len(T))]) )

                if check_interaction(T_i_v,T_j_v, norm_dict[Norm], eps_i):#norm_dict[Norm](mesh.bary[i_triangles]-mesh.bary[j]) < eps_i -mesh.diam:
                    # no re-triangulation needed
                    def Phi_j(y):
                        return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                    for a in range(3):
                        kk = np.where(nodes == T_i[a])[0][0]
                        for b in range(3):
                           if label_j != labels_domains[-1]:
                               L[kk, np.where(nodes == T_j[b])[0][0]] += -det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           L[kk, np.where(nodes == T_i[b])[0][0]] += det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()
                else:

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))

                    def I1(x):
                        x_trans = (T_i_v[0]+Mat_i.dot(x))
                        integral, integral0, integral1, integral2 = 0., 0., 0., 0.
                        aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)

                        def inner(tri, gam_j):
                            tri = np.array(tri)
                            Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                            det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                            def Phi_l(y):
                                return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                            GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                            if label_j != labels_domains[-1]:
                                return  GAM.sum(), (basis[0](iPhi_j(Phi_l(P2))) * GAM ).sum(), (basis[1](iPhi_j(Phi_l(P2))) * GAM ).sum()  , (basis[2](iPhi_j(Phi_l(P2))) * GAM ).sum()
                            else:
                                return  GAM.sum(), 0., 0., 0.

                        tris = retriangulate(x_trans, T_j_v, Norm, eps_i )
                        if len(tris) != 0:
                            for tri in tris:
                                v, v0, v1, v2 = inner(tri, gam_j)
                                integral  += v
                                integral0 += v0
                                integral1 += v1
                                integral2 += v2

                                """plot for testing below"""
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False))

                        return np.array([integral0, integral1, integral2, integral])

                    I = np.array(list(map(I1, P.transpose()))).transpose()

                    for a in range(3):
                        kk = np.where(nodes == T_i[a])[0][0]
                        for b in range(3):
                           if label_j != labels_domains[-1]:
                               L[kk, np.where(nodes == T_j[b])[0][0]] += -det_T_i * (PSI_P[a] * I[b] * weights).sum()
                           L[kk, np.where(nodes == T_i[b])[0][0]] += det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

    L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L



def assembly_coupling_full_DG(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        STANDARD VERSION BY NOW !

        -----

       uses simple criterion
                ||E_a^bary - E_b^bary || < delta - h
       to decide whether subdivision or special treatment of outer triangle is
       needed
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((3 * len(mesh.elements), 3 * len(mesh.elements)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            i_triangles = i
            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                if norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i  -mesh.diam:
                    # no subdivision or outer integral treatment needed
                    def Phi_j(y):
                        return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                    for a in range(3):
                        for b in range(3):
                            #if label_j != labels_domains[-1]:
                            L[3*i+a, 3*j+b] += -det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                            # if i==j:
                            L[3*i+a, 3*i+b] += det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()
                else:

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))

                    def I1(x):
                        x_trans = (T_i_v[0]+Mat_i.dot(x))
                        integral, integral0, integral1, integral2 = 0., 0., 0., 0.
                        aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)

                        def inner(tri, gam_j):
                            tri = np.array(tri)
                            Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                            det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                            def Phi_l(y):
                                return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                            GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                            return  GAM.sum(), (basis[0](iPhi_j(Phi_l(P2))) * GAM ).sum(), (basis[1](iPhi_j(Phi_l(P2))) * GAM ).sum()  , (basis[2](iPhi_j(Phi_l(P2))) * GAM ).sum()

                        tris = retriangulate(x_trans, T_j_v, norm, eps_i )

                        if len(tris) != 0:
                            for tri in tris:
                                v, v0, v1, v2 = inner(tri, gam_j)
                                integral  += v
                                integral0 += v0
                                integral1 += v1
                                integral2 += v2

                                """plot for testing below"""
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False))

                        return np.array([integral0, integral1, integral2, integral])

                    I = np.array( list(map(I1, P.transpose())) ).transpose()
                    for a in range(3):
                        for b in range(3):

                           L[3*i+a, 3*j+b] += -det_T_i * (PSI_P[a] * I[b] * weights).sum()
                           L[3*i+a, 3*i+b] += det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#
   # Ls = list(map(aux, range(num_cores)))  #
#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((3 * len(mesh.elements), 3 * len(mesh.elements)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L

def assembly_coupling_full_standard(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        STANDARD VERSION BY NOW !

        -----

       uses simple criterion
                ||E_a^bary - E_b^bary || < delta - h
       to decide whether subdivision or special treatment of outer triangle is
       needed
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])


            """test for minimum precision"""
            # def Phi_i(y):
            #     return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n_inner*n_outer, axis=1) +  Mat_i.dot(y)

            i_triangles = i
            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                if norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i -mesh.diam:
                    # no subdivision or outer integral treatment needed

                    """test for minimum precision"""
                    # def Phi_j(y):
                    #     return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)
                    def Phi_j(y):
                        return np.repeat(T_j_v[0][:,np.newaxis], n_inner*n_outer, axis=1) +  Mat_j.dot(y)

                    GAM = det_T_i * abs(det_T_j) * W_minprec * gam_j(Phi_i(X_outer), Phi_j(Y_inner))

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
                           # L[kk, T_j[b]] += -det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           # L[kk, T_i[b]] +=  det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           """minimum precision variant"""
                           L[kk, T_j[b]] += -(PSI_outer[a] * PSI_inner[b] * GAM).sum()
                           L[kk, T_i[b]] +=  (PSI_outer[a] * PSI_outer[b] * GAM).sum()

                else:

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))

                    def I1(x):
                        x_trans = (T_i_v[0]+Mat_i.dot(x))
                        integral, integral0, integral1, integral2 = 0., 0., 0., 0.
                        aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)

                        def inner(tri, gam_j):
                            tri = np.array(tri)
                            Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                            det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                            def Phi_l(y):
                                return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                            GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                            return  GAM.sum(), (basis[0](iPhi_j(Phi_l(P2))) * GAM ).sum(), (basis[1](iPhi_j(Phi_l(P2))) * GAM ).sum()  , (basis[2](iPhi_j(Phi_l(P2))) * GAM ).sum()

                        tris = retriangulate(x_trans, T_j_v, norm, eps_i )

                        if len(tris) != 0:
                            for tri in tris:
                                v, v0, v1, v2 = inner(tri, gam_j)
                                integral  += v
                                integral0 += v0
                                integral1 += v1
                                integral2 += v2

                                """plot for testing below"""
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False))

                        return np.array([integral0, integral1, integral2, integral])

                    I = np.array(list(map(I1, P.transpose()))).transpose()

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
                           L[kk, T_j[b]] += -det_T_i * (PSI_P[a] * I[b] * weights).sum()
                           L[kk, T_i[b]] += det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L

def assembly_coupling_full_exactcaps(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        STANDARD VERSION BY NOW !

        -----

       uses simple criterion
                ||E_a^bary - E_b^bary || < delta - h
       to decide whether subdivision or special treatment of outer triangle is
       needed
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            i_triangles = i
            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                if norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i -mesh.diam:
                    # no subdivision or outer integral treatment needed
                    def Phi_j(y):
                        return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
#                           if label_j != labels_domains[-1]:
                           L[kk, T_j[b]] += -det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           L[kk, T_i[b]] += det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()
                else:

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))
                    def iPhi_j_1point(y):
                        return iMat_j.dot( y - T_j_v[0])
                    def I1(x):
                        x_trans = (T_i_v[0]+Mat_i.dot(x))
                        integral, integral0, integral1, integral2 = 0., 0., 0., 0.
                        aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)

                        def inner(tri, gam_j):
                            tri = np.array(tri)
                            Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                            det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                            def Phi_l(y):
                                return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                            GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                            return  GAM.sum(), (basis[0](iPhi_j(Phi_l(P2))) * GAM ).sum(), (basis[1](iPhi_j(Phi_l(P2))) * GAM ).sum()  , (basis[2](iPhi_j(Phi_l(P2))) * GAM ).sum()

                        def inner_cap(cap, gam_j):
                            GAM = gam_j(x_trans, cap[0]) * cap[1]
                            return  GAM.sum(), (basis[0](iPhi_j_1point(cap[0])) * GAM ).sum(), (basis[1](iPhi_j_1point(cap[0])) * GAM ).sum()  , (basis[2](iPhi_j_1point(cap[0])) * GAM ).sum()

                        tris, caps = retriangulate(x_trans, T_j_v, norm, eps_i )

                        if len(tris) != 0:
                            for tri in tris:
                                v, v0, v1, v2 = inner(tri, gam_j)
                                integral  += v
                                integral0 += v0
                                integral1 += v1
                                integral2 += v2
                        if len(caps) != 0:
                            for cap in caps:
                                v, v0, v1, v2 = inner_cap(cap, gam_j)
                                integral += v
                                integral0 += v0
                                integral1 += v1
                                integral2 += v2
                                """plot for testing below"""
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
                #                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False))

                        return np.array([integral0, integral1, integral2, integral])

                    I = np.array(list(map(I1, P.transpose()))).transpose()

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
                           L[kk, T_j[b]] += -det_T_i * (PSI_P[a] * I[b] * weights).sum()
                           L[kk, T_i[b]] += det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L


def assembly_coupling_full_bary(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        RETRIANGULATE OUTER INTEGRAL

        -----

       uses simple criterion
                ||E_a^bary - E_b^bary || < delta - h
       to decide whether subdivision or special treatment of outer triangle is
       needed
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)


    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)
            iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
            def iPhi_i(y):
                return iMat_i.dot( y - np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1))
            i_triangles = i

            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]
                def Phi_j(y):
                    return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                if norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i -mesh.diam:
                    # no subdivision or outer integral treatment needed

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
#                           if label_j != labels_domains[-1]:
                           L[kk, T_j[b]] += -abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           L[kk, T_i[b]] += abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()
                else:


                    tris = retriangulate(mesh.bary[j], T_i_v, norm, eps_i )

                    for tri in tris:

                        tri = np.array(tri)
                        Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                        det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                        def Phi_l(y):
                            return np.repeat(tri[0][:,np.newaxis], n**2, axis=1) +  Mat_l.dot(y)

                        for a in range(3):
                            kk = T_i[a]
                            for b in range(3):
    #                           if label_j != labels_domains[-1]:
                               L[kk, T_j[b]] += -abs(det_l) * abs(det_T_j) * ( basis[a](iPhi_i(Phi_l(X))) *  PSI_Y[b]  * W * gam_j(Phi_l(X),Phi_j(Y))).sum()
                               L[kk, T_i[b]] += abs(det_l) * abs(det_T_j) * ( basis[a](iPhi_i(Phi_l(X))) *  basis[b](iPhi_i(Phi_l(X)))  * W * gam_j(Phi_l(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L


def assembly_coupling_full_shifted(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        uses shifted balls
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)
            i_triangles = i

            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]
                def Phi_j(y):
                    return np.repeat(T_j_v[0][:,np.newaxis], n2**2, axis=1) +  Mat_j.dot(y)
                iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                def iPhi_j(y):
                    return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1))
                # check if close enough, so that integration over whole elements
                if norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i -mesh.diam:
                    # no subdivision or outer integral treatment needed

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
#                           if label_j != labels_domains[-1]:
                           L[kk, T_j[b]] += -abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           L[kk, T_i[b]] += abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

                # otherwise re-triangulate
                else:


                    tris = retriangulate(mesh.bary[i], T_j_v, norm, eps_i )

                    for tri in tris:

                        tri = np.array(tri)
                        Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                        det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                        def Phi_l(y):
                            return np.repeat(tri[0][:,np.newaxis], n**2, axis=1) +  Mat_l.dot(y)

                        for a in range(3):
                            kk = T_i[a]
                            for b in range(3):
    #                           if label_j != labels_domains[-1]:
                               L[kk, T_j[b]] += -abs(det_l) * abs(det_T_i) * ( PSI_X[a] *  basis[b](iPhi_j(Phi_l(Y)))  * W * gam_j(Phi_i(X),Phi_l(Y))).sum()
                               L[kk, T_i[b]] +=  abs(det_l) * abs(det_T_i) * ( PSI_X[a] *  PSI_X[b]   * W * gam_j(Phi_i(X),Phi_l(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L

def assembly_coupling_full_approx(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
        Combination of: Shifted ball and barycenter method

           = integrate over whole triangles with barycenters are closer than delta
    """
#    labels_domains = np.sort(np.unique(mesh.triangles[:,0]))
    deform = kwargs.get('deform', 0)
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose()) < eps_i)[0].tolist()
            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]
                def Phi_j(y):
                    return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                for a in range(3):
                    kk = T_i[a]
                    for b in range(3):
#                           if label_j != labels_domains[-1]:
                       L[kk, T_j[b]] += -abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                       L[kk, T_i[b]] +=  abs(det_T_i) * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

#    L =  np.zeros((len(verts),len(verts)), dtype = float)#ss.lil_matrix((len(verts),len(verts)), dtype = float)
    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L


def assembly_coupling_full_adaptive(mesh, gam, retriangulate, norm, num_cores, **kwargs):
    """
       uses simple criterion
                ||E_a^bary - E_b^bary || < delta - h
       to decide whether subdivision or special treatment of outer triangle is
       needed

       +++

       adaptive quadrature rule for outer triangle

    """
    hash_onthefly = kwargs.get('hash_onthefly', 0)
    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.elements))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

#        L =  np.zeros((len(verts),len(verts)), dtype = float)#
        L = ss.lil_matrix((len(verts),len(verts)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.elements[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.elements[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            if hash_onthefly:
                hash_i = np.where(norm((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
            else:
                hash_i = mesh.hash_table_bary[i_triangles]

            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                if norm(mesh.bary[i]-mesh.bary[j]) < eps_i - mesh.diam:
                    # no subdivision or outer integral treatment needed
                    def Phi_j(y):
                        return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                    for a in range(3):
                        kk = T_i[a]
                        for b in range(3):
                           L[kk, T_j[b]] += -det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()
                           L[kk, T_i[b]] +=  det_T_i * abs(det_T_j) * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()

                else:

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))

                    for a in range(3):
                        for b in range(3):

                            def I1(x):
                                x_trans = (T_i_v[0]+Mat_i.dot(x))
                                integral  = 0.
                                aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)
                                aux2 = np.repeat(x[:,np.newaxis], n2, axis=1)

                                def inner(tri, gam_j):
                                    tri = np.array(tri)
                                    Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                                    det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                                    def Phi_l(y):
                                        return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                                    GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                                    return (basis[a](aux2) * basis[b](aux2) * GAM ).sum()

                                tris = retriangulate(x_trans, T_j_v, norm, eps_i )

                                if len(tris) != 0:
                                    for tri in tris:
                                        integral += inner(tri, gam_j)

                                return integral

                            def I2(x):
                                x_trans = (T_i_v[0]+Mat_i.dot(x))
                                integral  = 0.
                                aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)
                                aux2 = np.repeat(x[:,np.newaxis], n2, axis=1)

                                def inner(tri, gam_j):
                                    tri = np.array(tri)
                                    Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                                    det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                                    def Phi_l(y):
                                        return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                                    GAM = det_l * gam_j(aux, Phi_l(P2)) * weights2

                                    return -  ( basis[a](aux2) * basis[b](iPhi_j(Phi_l(P2)))   * GAM ).sum()

                                tris = retriangulate(x_trans, T_j_v, norm, eps_i )

                                if len(tris) != 0:
                                    for tri in tris:
                                        integral += inner(tri, gam_j)

                                return integral

                            val1 = det_T_i * tri_adapt(I1, T_ref)
                            val2 = det_T_i * tri_adapt(I2, T_ref)
                            L[T_i[a], T_i[b]] += val1
                            L[T_i[a], T_j[b]] += val2

        return 2 * L

    p = Pool(num_cores)

    Ls = p.map(aux, range(num_cores))#

    L =  ss.lil_matrix((len(verts),len(verts)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L




def assembly_coupling_approx(mesh, gam, retriangulate, Norm, num_cores, **kwargs):

    labels_domains = np.sort(np.unique(mesh.elements[:, 0]))
    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    nodes = np.array(mesh.nodes)

    # randomly shuffle indices:
    # For areas where the mesh is fine, hash_i might be much larger
    liste = range(len(mesh.omega))
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):

        L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)

        for i in pieces[m].tolist():

            label_i = mesh.omega[i, 0]

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = verts[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = abs(Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1])
            def Phi_i(y):
                return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)

            hash_i = np.where(norm_dict[Norm]((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i))[0].tolist()
            for j in hash_i:

                label_j = mesh.elements[j, 0]
                gam_j = gam[str(label_i)+str(label_j)]

                T_j = mesh.elements[j, 1:].tolist()
                T_j_v = verts[T_j]
                Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                det_T_j = abs(Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1])
                def Phi_j(y):
                    return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)

                for a in range(3):
                    kk = np.where(nodes == T_i[a])[0][0]
                    for b in range(3):
                       L[kk, np.where(nodes == T_i[b])[0][0]] += det_T_i * det_T_j * ( PSI_X[a] *  PSI_X[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()
                       if label_j != labels_domains[-1]:
                           # CONVOLUTION PART
                           L[kk, np.where(nodes == T_j[b])[0][0]] += -det_T_i * det_T_j * ( PSI_X[a] *  PSI_Y[b]  * W * gam_j(Phi_i(X),Phi_j(Y))).sum()

        return 2 * L

    p = Pool(num_cores)
    Ls = list(p.map(aux, range(num_cores))) #

    L = ss.lil_matrix((len(mesh.nodes),len(mesh.nodes)), dtype = float)
    for i in range(num_cores):
        L += Ls[i]

    del Ls
    p.close()
    p.join()
    p.clear()

    return L



"""-------------------------------------------------------------------------"""
"""                   END ASSEMBLY                                          """
"""-------------------------------------------------------------------------"""

#=============================================================================#
#=============================================================================#
"""-------------------------------------------------------------------------"""
"""                             SOLVE SYSTEM                                """
"""-------------------------------------------------------------------------"""
def solve(mesh, A, b):
    #==========================================================================
    #               INCORPORATE DIRICHLET VOLUME CONSTRAINTS
    #==========================================================================
    A = A.tolil()
    num_nodes = len(mesh.nodes)
    for k in mesh.boundary:
        # adjust righthand side at particular value (2x because substract again)
        b[k] = 0
        A[k,:] = ss.eye(1, num_nodes, k).tocsr()
        A[:, k] = ss.eye(1, num_nodes, k ).tocsr().transpose()

    A = A.tocsr()
#    def func(x):
#        return A.dot(x)
#
#    u = cg(func,b,0,True)
    u = ssl.spsolve(A, b)

    return u
"""-------------------------------------------------------------------------"""
"""                          END SOLVE SYSTEM                               """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#



"""-------------------------------------------------------------------------"""
"""                          PLOT                                           """
"""-------------------------------------------------------------------------"""
#def myfunc(a,b, *args, **kwargs):
#      c = kwargs.get('c', None)
#      d = kwargs.get('d', None)
def plot(mesh, u, **kwargs):
    """
    verts: vertices of the triangulation
    nodes: number of the nodes in Omega
    z: array containing for each triangle 3 values, indicating value at the nodes
    """
    title = kwargs.get('title', '')
    vmin_max = kwargs.get('vmin_max', [min(u), max(u)])
    plt.figure(title)
    plt.tricontourf(mesh.vertices[mesh.nodes][:, 0], mesh.vertices[mesh.nodes][:, 1], u, 100, interpolation='gaussian', cmap =plt.cm.get_cmap('rainbow'), vmin = vmin_max[0], vmax = vmin_max[1]) # choose 20 contour levels, just to show how good its interpolation is
    plt.colorbar()
    plt.axis('equal')
    plt.show()

def plot_inner(mesh, u, **kwargs):
    """
    verts: vertices of the triangulation
    nodes: number of the nodes in Omega
    z: array containing for each triangle 3 values, indicating value at the nodes
    """
    nodes_inner = range(len(mesh.nodes)-len(mesh.boundary))
    title = kwargs.get('title', '')
    vmin_max = kwargs.get('vmin_max', [min(u), max(u)])
    plt.figure(title)
    plt.tricontourf(mesh.vertices[nodes_inner][:, 0], mesh.vertices[nodes_inner][:, 1], u, 100, interpolation='gaussian', cmap =plt.cm.get_cmap('rainbow'), vmin = vmin_max[0], vmax = vmin_max[1]) # choose 20 contour levels, just to show how good its interpolation is
    plt.colorbar()
    plt.axis('equal')

def plot_all(mesh, u, **kwargs):
    """
    verts: vertices of the triangulation
    nodes: number of the nodes in Omega
    z: array containing for each triangle 3 values, indicating value at the nodes
    """
    title = kwargs.get('title', '')
    vmin_max = kwargs.get('vmin_max', [min(u), max(u)])
    plt.figure(title)
    plt.tricontourf(mesh.vertices[:, 0], mesh.vertices[:, 1], u, 100, interpolation='gaussian', cmap =plt.cm.get_cmap('rainbow'), vmin = vmin_max[0], vmax = vmin_max[1]) # choose 20 contour levels, just to show how good its interpolation is
    plt.colorbar()
    plt.axis('equal')



def plot_mesh(mesh, **kwargs):
    """
    mesh: mesh class from above
    """
    new_figure = kwargs.get('new_figure', True)
    title = kwargs.get('title', '')
    linewidth = kwargs.get('linewidth', 1)
    verts = kwargs.get('verts', mesh.vertices)
    if new_figure:
        plt.figure(title)

    labels = np.sort(np.unique(mesh.elements[:, 0]).tolist())
    num_labels = len(labels)
    color = ['r', 'b', 'black', 'g', 'y', 'grey']
    for i in range(num_labels):
        plt.triplot(verts[:,0], verts[:,1], mesh.elements[np.where(mesh.elements[:, 0] == labels[i])][:, 1:], color = color[i], linewidth = linewidth)
#        tris_label = mesh.triangles[np.where(mesh.triangles[:,0]==labels[i])]
#        for k in range(len(tris_label)):
#            plt.gca().add_patch(plt.Polygon(verts[tris_label[k,1:]], closed=True, fill = False, color = color[i], linewidth = linewidth))

    plt.axis('equal')
    plt.show()

def plot_vertices(mesh, indices, color, new_figure):
    """
    mesh: mesh class from above
    indices: indices of mesh.vertices which shall be plotted
    color: string with one letter indicating color (e.g., 'b', 'g', 'y', 'r',..)
    new_figure: False or True
    """
    if new_figure:
        plt.figure('Mesh')
    plt.plot(mesh.vertices[indices][:, 0], mesh.vertices[indices][:, 1], color + 'x')
"""-------------------------------------------------------------------------"""
"""                      END PLOT                                           """
"""-------------------------------------------------------------------------"""




"""
*******************************************************************************
*******************************************************************************

                    R E  -  T R I A N G U L A T I O N
     
*******************************************************************************
*******************************************************************************
"""
#==========================================================================
#                               AUXILIARY FUNCTIONS
#==========================================================================
def intersect(line1, line2):
    """
    line1 = [a, b]
    line2 = [c, d]
    returns intersection point of line(a,b) \cap line(c,d)
    if intersection not unique (linear dependent), then return 0

    caution: method does check for intersection of the (infinite) lines not the
             convex combinations
    """


def are_intersecting_lines(line1, line2):
    """
    line1 = [a, b]
    line2 = [c, d]

    tests if two (finite !) line segments intersect or not

    caution: method does check for intersection of the convex combinations
    """
    a = line1[0]
    b = line1[1]
    c = line2[0]
    d = line2[1]
    # construct matrix (b-a, d-c)
    M = np.array([b-a,c-d]).transpose()
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]

    tol = 0.0000001

    if det != 0:
        iM = 1./det * np.array([ [M[1,1], -M[0,1]], [-M[1,0], M[0,0]]  ])
        lbd = iM.dot(c-a)#np.linalg.solve(M,c-a)

        if tol < lbd[0] < 1-tol and tol < lbd[1] < 1-tol :
            return 1
    else:
        return 0
    # test
#line1 = [np.ones(2), np.zeros(2)]
#line2 = [np.array([0,0]), np.array([2,0])]
#print are_intersecting_lines(line1, line2)

def which_vertices(vertices, T):
    """ test which vertices of the ball lie in interior of triangle T """
    """ ATTENTION: Vertices have to be sorted accordingly !!! """

     #sort
    origin = np.array(T[0])
    refvec = T[1]-T[0]#np.array([1, 0])
    def clockwiseangle_and_distance(point):
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        return angle, lenvector
    a = sorted(range(len(T)),key=lambda x:clockwiseangle_and_distance(T[x]))
    T = T[a][::-1]

    ipts =[]
    a = T[0]
    b = T[1]
    c = T[2]

    M = np.array([b-a, c-a]).transpose()
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    iM = 1./det * np.array([ [M[1,1], -M[0,1]], [-M[1,0], M[0,0]]  ])

    for vert in vertices:
        lbd = iM.dot(vert-a)#np.linalg.solve(M,line[0]-a)
        if 0<lbd[0]<1 and 0<lbd[1]<1 and (lbd[0]+lbd[1])<1:
            ipts += [vert]

    return ipts


def intersection_l1linf(x_trans, T_j, Norm, eps):
    """
    outputs vertices of the polygon that results from intersecting

        * the ball which is defined by its center 'x_trans', radius 'eps' and the Norm (='L1', 'Linf')
    and
        * the triangle T_j = [v0,v1,v2]

    """
    if Norm == 'L1':
        vertices = [x_trans + np.array([-eps,0]), x_trans +  np.array([0, eps]), x_trans + np.array([ eps, 0]),x_trans + np.array([ 0,-eps])]

        def norm(x):
            return np.sum(np.abs(x), axis = 0)#np.array([abs(x[i]) for i in range(len(x))]).sum()#

    elif Norm == 'Linf':
        vertices = [x_trans + np.array([-eps,-eps]), x_trans +  np.array([-eps,eps]), x_trans + np.array([ eps, eps]),x_trans + np.array([ eps,-eps])]

        def norm(x):
            return np.max(np.abs(x))

    """plot for testing below"""
#    for i in range(4):
#       plt.gca().add_patch(plt.Polygon([vertices[i], vertices[(i+1)%4]] , closed=False, fill = False, color = 'red'))

    # find out how many vertices of the triangle lie in the ball
    ipts = []
    for k in range(3):
        if norm(x_trans-T_j[k]) < eps:
            ipts = ipts + [k]

    # case 1 (3 interior) = element fully contained in the ball
    if len(ipts) == 3:
        points = T_j

    # case 2 = element only partly covered by the ball
    else:
        intersection_points = []
        sides = [[T_j[0], T_j[1]], [T_j[1], T_j[2]], [T_j[2], T_j[0]]]  # sides of triangle
        for a in range(4):
            for b in range(3):
                #--------------------------------------------------------
                # intersect two lines
                line1 = [vertices[a], vertices[(a + 1) % 4]]
                line2 = sides[b]
                a = line1[0]
                b = line1[1]
                c = line2[0]
                d = line2[1]
                # construct matrix (b-a, d-c)
                M = np.array([b - a, c - d]).transpose()
                det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
                if det == 0:
                    #        print 'there is no unique intersection'
                    p, lbd = np.zeros(2), 10. * np.ones(2)
                else:
                    iM = 1. / det * np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]])
                    lbd = iM.dot(c - a)  # np.linalg.solve(M,c-a)
                    p, lbd = a + lbd[0] * (b - a), lbd
                if 0 <= lbd[0] <= 1 and 0 <= lbd[1] <= 1:
                    intersection_points += [p]
                # --------------------------------------------------------
        points = [T_j[ipts[i]] for i in range(len(ipts))]
        points += intersection_points
        points += which_vertices(vertices, T_j)

    return points


def order_indices(b):
    origin = np.array(b[0])
    refvec = b[1] - b[0]  # np.array([1, 0])

    def clockwiseangle_and_distance(point):
        vector = [point[0] - origin[0], point[1] - origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0] / lenvector, vector[1] / lenvector]
        dotprod = normalized[0] * refvec[0] + normalized[1] * refvec[1]  # x1*x2 + y1*y2
        diffprod = refvec[1] * normalized[0] - refvec[0] * normalized[1]  # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        return angle, lenvector

    a = sorted(range(len(b)), key=lambda x: clockwiseangle_and_distance(b[x]))

    return a


# auxiliary functions
def intersection_points_l2(x, eps, T,norm):
    """
    compute intersection points of triangle and l2-ball
    """
    intersection_points = []
    sides = [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]]  # sides of triangle
    for i in range(3):
        side = sides[i]
        a = side[0]
        b = side[1]

        def f(lbd):
            return norm(a + lbd * (b - a) - x) - eps

        """USING pq formula"""
        c = a - x
        d = b - a
        p = np.dot(c, d) / np.dot(d, d)
        v = p ** 2 - (np.dot(c, c) - eps ** 2) / np.dot(d, d)
        if v >= 0:
            lbd1 = -p - np.sqrt(v)
            lbd2 = (np.dot(c, c) - eps ** 2) / np.dot(d, d) / lbd1  # -p +np.sqrt(v)
            if 0 <= lbd1 <= 1 and np.allclose(f(lbd1), 0.):
                intersection_points += [np.array(a + lbd1 * (b - a))]
            if 0 <= lbd2 <= 1 and np.allclose(a + lbd1 * (b - a), a + lbd2 * (b - a)) == False and np.allclose(f(lbd2),0.):  #
                intersection_points += [np.array(a + lbd2 * (b - a))]

    return intersection_points#




def intersection_exactl2_1(x_trans, T, eps):
    """
    POLYGON THAT RESULTS FROM LEAVING OUT CAPS
        +
    PUTTING 1 Triangles on the CAPS

    outputs vertices of the polygon that results from intersecting

        * the L2 ball which is defined by its center x_trans and eps
    and
        * the triangle T = [v0,v1,v2]

    """
    def norm(x):
        return np.sqrt(np.dot(x,x))

    """ POLYGON (without caps) """
    # (1) Find out interior points (= vertices of triangle that lie in the interior of the ball)
    ipts = []
    for k in range(3):
        if norm(x_trans-T[k]) < eps:
            ipts = ipts + [k]

    # CASE 1 (3 interior = triangle fully contained in the ball) -> finished
    if len(ipts) == 3:
        return [T]

    # CASE 2 (not all points are interior points, i.e., boundaries intersect or intersection empty)
    # we have to figure out intersection points and potentially add caps
    else:

        # (2) Compute intersection points (there do not have to be one, if there is no intersection)
        # we also label these points according to which side of the triangle they intersect
        # due to the labeling we cannot use the function "intersection_points_l2"
        intersection_points = []
        labels = []
        sides = [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]] # sides of triangle
        for i in range(3):
            c = sides[i][0] - x_trans
            d = sides[i][1] - sides[i][0]
            p = np.dot(c,d)/np.dot(d,d)
            v = p**2 - (np.dot(c,c)-eps**2) / np.dot(d,d)
            if v >= 0:
                # lbd1 and lbd2 are the two roots of ||s1 + lbd (s2-s1) - x_trans|| - eps = 0
                #                                    ||c + lbd * d|| - eps = 0
                # the following is the numerically more stable p/q formula implementation
                lbd1 = -p - np.sqrt(v)
                lbd2 = (np.dot(c, c) - eps ** 2) / np.dot(d, d) / lbd1

                if 0<=lbd1<=1:
                    intersection_points += [np.array(sides[i][0]+lbd1*d)]
                    labels += [i]
                if 0<=lbd2<=1:
                    intersection_points += [np.array(sides[i][0]+lbd2*d)]
                    labels += [i]

        # interior points + intersection points
        points = [T[ipts[i]] for i in range(len(ipts))] + intersection_points

        # empty intersection
        if len(points) < 3:
            tris = []

        # divide polygon resulting from points
        else:
            tris = divide_polygon(points)


        """ NOW ADD THE CAPS (put a single triangle into the cap) """
        num_int_points = len(intersection_points)

        # case not covered: ball very small and fully contained in a triangle, therefore this if statement
        if num_int_points>1:
            # we order intersection_point clockwise and adapt angles accordingly
            a = order_indices(intersection_points)
            intersection_points, labels = [intersection_points[a[i]] for i in range(len(a))], [labels[a[i]] for i in range(len(a))]

            # have to avoid that in this case two times the same triangle is put on cap
            if num_int_points==2:
                i = 0
                s1 = intersection_points[i]
                s2 = intersection_points[(i + 1) % num_int_points]
                s_bar = 0.5 * (s1 + s2)
                c_plus = x_trans + eps * ((s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                c_minus = x_trans - eps * ((s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                #
                # if cplus is in the triangle, then this is the cap
                aux = which_vertices([c_plus], T)
                if len(aux) > 0:
                    tris += [order([s1, s2, c_plus])]
                # if cminus is in the triangle, then this is the cap (note that both may not be in the triangle!)
                aux = which_vertices([c_minus], T)
                if len(aux) > 0:
                    tris += [order([s1, s2, c_minus])]
            else:
                for i in range(len(labels)):
                    if labels[i] != labels[(i+1) % num_int_points]:
                        s1 = intersection_points[i]
                        s2 = intersection_points[(i+1) % num_int_points]
                        s_bar = 0.5 * (s1+s2)
                        c_plus =  x_trans + eps * ((s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                        c_minus = x_trans - eps * ((s_bar - x_trans)  / np.linalg.norm(s_bar - x_trans))
                        #
                        # if cplus is in the triangle, then this is the cap
                        aux = which_vertices([c_plus], T)
                        if len(aux)>0:
                            tris += [order([s1, s2, c_plus])]
                        # if cminus is in the triangle, then this is the cap (note that both may not be in the triangle!)
                        aux = which_vertices([c_minus], T)
                        if len(aux)>0:
                            tris += [order([s1, s2, c_minus])]

    return tris

def intersection_exactl2_capsonly(x_trans, T, eps):

    def norm(x):
        return np.sqrt(np.dot(x,x))

    caps = []
    # (2) Compute intersection points (there do not have to be one, if there is no intersection)
    # we also label these points according to which side of the triangle they intersect
    # due to the labeling we cannot use the function "intersection_points_l2"
    intersection_points = []
    labels = []
    sides = [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]] # sides of triangle
    for i in range(3):
        c = sides[i][0] - x_trans
        d = sides[i][1] - sides[i][0]
        p = np.dot(c,d)/np.dot(d,d)
        v = p**2 - (np.dot(c,c)-eps**2) / np.dot(d,d)
        if v >= 0:
            lbd1 = -p - np.sqrt(v)
            lbd2 = (np.dot(c, c) - eps ** 2) / np.dot(d, d) / lbd1#-p +np.sqrt(v)
            if 0<=lbd1<=1:
                intersection_points += [np.array(sides[i][0]+lbd1*d)]
                labels += [i]
            if 0<=lbd2<=1:
                intersection_points += [np.array(sides[i][0]+lbd2*d)]
                labels += [i]
    """  CAPS """
    num_int_points = len(intersection_points)

    # we do not cover the case that the ball is fully contained in a triangle, therefore this if statement
    if num_int_points>1:
        # we order intersection_point clockwise and adapt angles accordingly
        a = order_indices(intersection_points)
        intersection_points, labels = [intersection_points[a[i]] for i in range(len(a))], [labels[a[i]] for i in range(len(a))]

        # sides of the "nocaps"-polygon clockwise
        sides = []
        for i in range(num_int_points):
            sides += [[np.array(intersection_points[i]), np.array(intersection_points[(i+1) % num_int_points])]]


        if num_int_points == 2:
            s1 = intersection_points[0]
            s2 = intersection_points[1]
            p1, p2 = s1 - x_trans, s2 - x_trans
            s_bar = 0.5 * (s1 + s2)

            alpha = 0.5 * np.arccos(np.dot(p1, p2) / eps ** 2)
            centroid = x_trans + 4 * eps * np.sin(alpha) ** 3 / (3 * (2 * alpha - np.sin(2 * alpha))) * (
                        (s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
            area = 0.25 * eps ** 2 * (2 * alpha - np.sin(2 * alpha))

            # plt.plot(s1[0], s1[1], 'rx')
            # plt.plot(s2[0], s2[1], 'yx')
            # plt.plot(x_trans[0], x_trans[1], 'bx')
            # plt.plot(centroid[0], centroid[1], 'ro')
            # plt.quiver(x_trans[0], x_trans[1],p1[0], p1[1],   angles='xy', scale_units='xy', scale=1)
            # plt.quiver(x_trans[0], x_trans[1],p2[0], p2[1],  color='b',angles='xy', scale_units='xy', scale=1)

            caps += [[alpha, centroid, area]]

        else:
            for i in range(len(labels)):
                #caps points always belong to different sides of the triangles
                if labels[i] != labels[(i+1) % num_int_points]:

                    s1 = intersection_points[i]
                    s2 = intersection_points[(i+1) % num_int_points]
                    p1, p2 = s1-x_trans, s2-x_trans
                    s_bar = 0.5 * (s1+s2)

                    alpha = 0.5 * np.arccos(np.dot(p1, p2) / eps ** 2)
                    centroid = x_trans + 4 * eps * np.sin(alpha) ** 3 / (3 * (2 * alpha - np.sin(2 * alpha))) * ( (s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                    area = 0.25 * eps ** 2 * (2 * alpha - np.sin(2 * alpha))

                    # plt.plot(s1[0], s1[1], 'rx')
                    # plt.plot(s2[0], s2[1], 'yx')
                    # plt.plot(x_trans[0], x_trans[1], 'bx')
                    # plt.plot(centroid[0], centroid[1], 'ro')
                    # plt.quiver(x_trans[0], x_trans[1],p1[0], p1[1],   angles='xy', scale_units='xy', scale=1)
                    # plt.quiver(x_trans[0], x_trans[1],p2[0], p2[1],  color='b',angles='xy', scale_units='xy', scale=1)

                    caps += [[alpha, centroid, area]]

    return caps


def intersection_approxl2(x_trans, T_j, eps):
    """
    POLYGON THAT RESULTS FROM LEAVING OUT CAPS

    outputs vertices of the polygon that results from intersecting

        * the L2 ball which is defined by its center x_trans and eps
    and
        * the triangle T_j = [v0,v1,v2]

    """
    def norm(x):
        return np.sqrt(np.dot(x,x))

    ipts = []
    for k in range(3):
        if norm(x_trans-T_j[k]) < eps:
            ipts = ipts + [k]

    # case 1 (3 interior)
    if len(ipts) == 3:
        tris = [T_j]

    else:
        points = [T_j[ipts[i]] for i in range(len(ipts))]
        points += intersection_points_l2(x_trans, eps, T_j, norm)

        if len(points) < 3:
            tris = []
        else:
            tris = divide_polygon(points)

    return tris


def intersection_l2_exactcaps(x_trans, T, eps):

    def norm(x):
        return np.sqrt(np.dot(x,x))

    caps = []
    # (2) Compute intersection points (there do not have to be one, if there is no intersection)
    # we also label these points according to which side of the triangle they intersect
    # due to the labeling we cannot use the function "intersection_points_l2"
    intersection_points = []
    labels = []
    sides = [[T[0], T[1]], [T[1], T[2]], [T[2], T[0]]] # sides of triangle
    for i in range(3):
        c = sides[i][0] - x_trans
        d = sides[i][1] - sides[i][0]
        p = np.dot(c,d)/np.dot(d,d)
        v = p**2 - (np.dot(c,c)-eps**2) / np.dot(d,d)
        if v >= 0:
            lbd1 = -p - np.sqrt(v)
            lbd2 = (np.dot(c, c) - eps ** 2) / np.dot(d, d) / lbd1#-p +np.sqrt(v)

            if 0<=lbd1<=1:
                intersection_points += [np.array(sides[i][0]+lbd1*d)]
                labels += [i]
            if 0<=lbd2<=1 and lbd1 != lbd2:
                intersection_points += [np.array(sides[i][0]+lbd2*d)]
                labels += [i]

    ipts = []
    for k in range(3):
        if norm(x_trans-T[k]) < eps:
            ipts = ipts + [k]

    # case 1 (3 interior)
    if len(ipts) == 3:
        tris = [T]

    else:
        points = [T[ipts[i]] for i in range(len(ipts))]
        points += intersection_points_l2(x_trans, eps, T, norm)

        if len(points) < 3:
            tris = []
        else:
            tris = divide_polygon(points)

    """  CAPS """
    num_int_points = len(intersection_points)

    # we do not cover the case that the ball is fully contained in a triangle, therefore this if statement
    if num_int_points>1:
        # we order intersection_point clockwise and adapt angles accordingly
        a = order_indices(intersection_points)
        intersection_points, labels = [intersection_points[a[i]] for i in range(len(a))], [labels[a[i]] for i in range(len(a))]

        # sides of the "nocaps"-polygon clockwise
        sides = []
        for i in range(num_int_points):
            sides += [[np.array(intersection_points[i]), np.array(intersection_points[(i+1) % num_int_points])]]

        if num_int_points == 2:
            s1 = intersection_points[0]
            s2 = intersection_points[1]
            p1, p2 = s1 - x_trans, s2 - x_trans
            s_bar = 0.5 * (s1 + s2)

            aux = np.dot(p1, p2) / eps ** 2
            if np.allclose(aux, 1.0, atol=1e-07) != True:
                alpha = 0.5 * np.arccos(aux)
                centroid = x_trans + 4 * eps * np.sin(alpha) ** 3 / (3 * (2 * alpha - np.sin(2 * alpha))) * (
                            (s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                area = 0.25 * eps ** 2 * (2 * alpha - np.sin(2 * alpha))

                # print(eps, np.dot(p1, p2) , np.dot(p1, p2) / eps ** 2)
                # plt.plot(s1[0], s1[1], 'rx')
                # plt.plot(s2[0], s2[1], 'yx')
                # plt.plot(x_trans[0], x_trans[1], 'bx')
                # plt.plot(centroid[0], centroid[1], 'ro')
                # plt.quiver(x_trans[0], x_trans[1],p1[0], p1[1],   angles='xy', scale_units='xy', scale=1)
                # plt.quiver(x_trans[0], x_trans[1],p2[0], p2[1],  color='b',angles='xy', scale_units='xy', scale=1)

                caps += [[centroid, area]]

        else:

            for i in range(len(labels)):
                #caps points always belong to different sides of the triangles
                if labels[i] != labels[(i+1) % num_int_points]:

                    s1 = intersection_points[i]
                    s2 = intersection_points[(i+1) % num_int_points]
                    p1, p2 = s1-x_trans, s2-x_trans
                    s_bar = 0.5 * (s1+s2)
                    aux = np.dot(p1, p2) / eps ** 2
                    if np.allclose(aux, 1.0, atol=1e-07) != True:
                        alpha = 0.5 * np.arccos(aux)
                        centroid = x_trans + 4 * eps * np.sin(alpha) ** 3 / (3 * (2 * alpha - np.sin(2 * alpha))) * ( (s_bar - x_trans) / np.linalg.norm(s_bar - x_trans))
                        area = 0.25 * eps ** 2 * (2 * alpha - np.sin(2 * alpha))
                        # if np.dot(p1, p2) == 0:
                        #     plt.plot(s1[0], s1[1], 'rx')
                        #     plt.plot(s2[0], s2[1], 'yx')
                        #     plt.plot(x_trans[0], x_trans[1], 'bx')
                        #     plt.plot(centroid[0], centroid[1], 'ro')
                        #     plt.quiver(x_trans[0], x_trans[1],p1[0], p1[1],   angles='xy', scale_units='xy', scale=1)
                        #     plt.quiver(x_trans[0], x_trans[1],p2[0], p2[1],  color='b',angles='xy', scale_units='xy', scale=1)

                        caps += [[centroid, area]]

    return tris, caps

#==============================================================================
def divide_polygon(b):
    """
    task:  divide domain defined by polygon into triangles
    input: a = [a0, a1, a2, a3, ...] nodes of the polygon
    output:  [], [] given the indices of the triangle

    idea: sort nodes counter clockwise
    """
    origin = np.array(b[0])
    refvec = b[1]-b[0]#np.array([1, 0])
    def clockwiseangle_and_distance(point):
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        return angle, lenvector

    a = sorted(range(len(b)),key=lambda x:clockwiseangle_and_distance(b[x]))
    t = []

    for i in range(len(a)-2):

        t  += [ [b[a[0]].tolist(), b[a[i+1]].tolist(), b[a[i+2]].tolist()] ]

    return t

### ADD BARYCENTER
def divide_polygon_with_barycenter(b):
    """
    task:  divide domain defined by polygon into triangles
    input: a = [a0, a1, a2, a3]  nodes of the polygon
    output:  [[], [],...], given the indices of the triangle

    idea: sort nodes counter clockwse
    """

    origin = np.array(b[0])
    refvec = b[1]-b[0]#np.array([1, 0])
    def clockwiseangle_and_distance(point):
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        return angle, lenvector

    a = sorted(range(len(b)),key=lambda x:clockwiseangle_and_distance(b[x]))
    t = []

    bary = np.zeros(2)
    for i in range(len(b)):
        bary += b[i]

    bary = (1./ float(len(b))) * bary

    for i in range(len(a)):
        t += [[b[a[i]].tolist(), bary, b[a[(i+1)%len(a)]].tolist()]]
        #t  += [ [b[a[0]].tolist(), b[a[i+1]].tolist(), b[a[i+2]].tolist()] ]

    return t

def order(b):
    origin = np.array(b[0])
    refvec = b[1]-b[0]#np.array([1, 0])
    def clockwiseangle_and_distance(point):
        vector = [point[0]-origin[0], point[1]-origin[1]]
        lenvector = math.hypot(vector[0], vector[1])
        if lenvector == 0:
            return -math.pi, 0
        normalized = [vector[0]/lenvector, vector[1]/lenvector]
        dotprod  = normalized[0]*refvec[0] + normalized[1]*refvec[1]     # x1*x2 + y1*y2
        diffprod = refvec[1]*normalized[0] - refvec[0]*normalized[1]     # x1*y2 - y1*x2
        angle = math.atan2(diffprod, dotprod)
        return angle, lenvector

    a = sorted(range(len(b)),key=lambda x:clockwiseangle_and_distance(b[x]))
    res = [b[a[i]] for i in range(len(b))]
    return res


#==============================================================================
"""  EXACT: L1 and Linf  """
#==============================================================================

def retriangulate_exactL1Linf(x, T, Norm, eps):

    points = intersection_l1linf(x, T, Norm, eps)
    if len(points) == 3:
        tris = [points]
    elif len(points) > 3:
        tris = divide_polygon(points)
    else:
        tris = []

    return tris

#==============================================================================
"""  L2 """
#==============================================================================
def retriangulate_exactL2(x, T, norm, eps):
    return intersection_exactl2_1(x, T, eps)#intersection_l2(x, T, eps)

def retriangulate_exactcaps(x, T, norm, eps):
    return intersection_l2_exactcaps(x, T, eps)

def retriangulate_approxL2(x, T, norm, eps):
    return intersection_approxl2(x, T, eps)

#==============================================================================
"""  APPROX_1: intersection is nonempty (all norms) """
#==============================================================================
def retriangulate_apx1(x, T, norm, eps):
    """I think the criterion is wrong !!! """
    aux = np.array([norm(x-t) for t in T])
    if np.any(aux<eps) or len(which_vertices([x], T)) == 1: # intersection is nonempty <=> one vertices lies in the interior of the ball
        return [T]
    else:
        return []

#==============================================================================
"""  APPROX_2: barycenter is in ball (all norms) """
#==============================================================================
def retriangulate_apx2(x, T, norm, eps):
    if norm((1./3.*(T[0]+T[1]+T[2])) - x) < eps:
        return [T]
    else:
        return []

#==============================================================================
retriangulate_dict = {'exact':      retriangulate_exactL1Linf,
                      'exact_L2':   retriangulate_exactL2,
                      'approx_L2':  retriangulate_approxL2, # leave out the caps
                      'approx1':    retriangulate_apx1, # intersection nonempty
                      'approx2':    retriangulate_apx2, # barycenter lies in ball
                      'exactcaps':  retriangulate_exactcaps # centroid rule for caps
                      }

""" TEST  RE - TRIANGULATE """
re_triang_test =0
if re_triang_test:
    linewidth = 2.5
    Norm = 2#'L1'#
    T0 = np.array([[-0.8,0.], [1.0, 0.], [0.,2]])
    color_ball = 'red'
    alpha_triangles = 0.1
    color_triangles = 'cyan'
    X = [np.array([0.3, 0.8]), np.array([0.25, 0.8]), np.array([0.1, 0.8]), np.array([-0.7, 0.1]), np.array([0.6, 1.4]), np.array([0.25, 0.4]), np.array([0.1, 0.5]), np.array([0.12, 0.5])]
    Eps = [0.4, 0.9, 0.3, 0.4, 0.8, 1., 0.8, 0.75]

    #X = [X[4], X[6]]
    #Eps = [Eps[4], Eps[6]]
    for i in range(len(X)):
        fig, ax = plt.subplots()
        plt.gca().add_patch(plt.Polygon(T0 , closed=True, fill = False, linewidth = linewidth))
        x = X[i]
        eps = Eps[i]
    #    plt.plot(x[0], x[1], 'yo')
        circle = plt.Circle(tuple(x), eps, color=color_ball, fill= False, linewidth = linewidth)

        # tris = retriangulate_approxL2(x, T0, Norm, eps)
        tris = intersection_exactl2_1(x, T0, eps)

        # caps = intersection_exactl2_capsonly(x, T0, eps)
        # print(caps)
        # for i in range(len(caps)):
        #     print(i, caps[i])
        #     centroid = caps[i][1]
        #     plt.plot(centroid[0], centroid[1], 'bo')


        print( len(tris))
        if len(tris) != 0:
            for T in tris:
                plt.gca().add_patch(plt.Polygon(T , closed=True, fill = True, alpha = alpha_triangles, color = color_triangles, linewidth = 0.1))
                plt.gca().add_patch(plt.Polygon(T , closed=True, fill = False, linewidth = 0.7*linewidth))

        if Norm == 2:
            ax.add_artist(circle)
        elif Norm == 'Linf':
            plt.gca().add_patch(plt.Polygon([x + np.array([-eps,-eps]), x +  np.array([-eps,eps]), x + np.array([ eps, eps]),x + np.array([ eps,-eps])] , closed=True, fill = False, color = color_ball, linewidth = linewidth))
        else:
            plt.gca().add_patch(plt.Polygon([x + np.array([-eps,0]), x +  np.array([0, eps]), x + np.array([ eps, 0]),x + np.array([ 0,-eps])] , closed=True, fill = False, color = color_ball, linewidth = linewidth))

        plt.axis('equal')

        plt.xlim(-1.5,1.5)
        plt.ylim(-0.5,2.5)
        plt.gca().set_adjustable("box")
    #    plt.autoscale()
        plt.xticks([])
        plt.yticks([])
        plt.axis('equal')

        plt.show()

"""#####################################################"""
"""                             1D                      """
"""#####################################################"""
def retriangulate_1d_approx(x, I, delta , h):
    a = I[0]
    b = I[1]

    if (a >= x+delta and b >= x+delta) or (a <= x-delta and b <= x-delta):
        intersection = []
    else:
        intersection = I

    return intersection

def retriangulate_1d_exact(x, I, delta , h):
    a = I[0]
    b = I[1]
    if np.abs(a-x) < delta and np.abs(b-x) < delta:
        intersection = [a,b]
    elif np.abs(a-x) < delta:
        intersection = [a, x+delta]
    elif np.abs(b-x) < delta:
        intersection = [x-delta, b]
    else:
        intersection = []
    return intersection

def retriangulate_1d_cap1(x, I, delta, h):
    a = I[0]
    b = I[1]
    delta_aux = delta +  0.1 * h
    if np.abs(a-x) < delta_aux and np.abs(b-x) < delta_aux:
        intersection = [a,b]
    elif np.abs(a-x) < delta_aux:
        intersection = [a, x+delta_aux]
    elif np.abs(b-x) < delta_aux:
        intersection = [x-delta_aux, b]
    else:
        intersection = []
    return intersection

def retriangulate_1d_cap2(x, I, delta, h):
    a = I[0]
    b = I[1]
    delta_aux = delta +  h**2
    if np.abs(a-x) < delta_aux and np.abs(b-x) < delta_aux:
        intersection = [a,b]
    elif np.abs(a-x) < delta_aux:
        intersection = [a, x+delta_aux]
    elif np.abs(b-x) < delta_aux:
        intersection = [x-delta_aux, b]
    else:
        intersection = []
    return intersection

def retriangulate_1d_cap3(x, I, delta, h):
    a = I[0]
    b = I[1]
    delta_aux = delta +  h**3
    if np.abs(a-x) < delta_aux and np.abs(b-x) < delta_aux:
        intersection = [a,b]
    elif np.abs(a-x) < delta_aux:
        intersection = [a, x+delta_aux]
    elif np.abs(b-x) < delta_aux:
        intersection = [x-delta_aux, b]
    else:
        intersection = []
    return intersection

retriangulate_dict_1d = {'exact':retriangulate_1d_exact,
                         'approx':retriangulate_1d_approx,
                         'cap1':retriangulate_1d_cap1,
                         'cap2':retriangulate_1d_cap2,
                         'cap3':retriangulate_1d_cap3}

"""-------------------------------------------------------------------------"""
"""                      RE - TRIANGULATE                                   """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#



"""-------------------------------------------------------------------------"""
"""                    READ MESH                                            """
"""-------------------------------------------------------------------------"""
"""
Read mesh generated by gmsh

output: array of 

        1) Vertices (points in 3d !!!)
        2) Triangles
        3) Lines
        
For the file format see the documentation:

http://gmsh.info/doc/texinfo/gmsh.html#File-formats

"""

"""-------------------------------------------------------------------------"""
"""                     READ MESH                                   """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#
def read_mesh(mshfile):
    """meshfile = .msh - file genrated by gmsh """

    fid = open(mshfile, "r")

    for line in fid:

        if line.find('$Nodes') == 0:
            # falls in der Zeile 'Nodes' steht, dann steht in der...
            line = fid.readline()  #...naechsten Zeile...
            npts = int(line.split()[0]) #..die anzahl an nodes

            Verts = np.zeros((npts, 3), dtype=float) #lege array for nodes an anzahl x dim

            for i in range(0, npts):
                # run through all nodes
                line = fid.readline() # put current line to be the one next
                data = line.split() # split line into its atomic characters
                Verts[i, :] = list(map(float, data[1:])) # read out the coordinates of the node by applying the function float() to the characters in data

        if line.find('$Elements') == 0:
            line = fid.readline()
            nelmts = int(line.split()[0]) # number of elements

            Lines = []
            Triangles = []
            #Squares = np.array([])

            for i in range(0, nelmts):
                line = fid.readline()
                data = line.split()
                if int(data[1]) == 1:
                    """ 
                    we store [physical group, node1, node2, node3], 
                    -1 comes from python starting to count from 0
                    """
                    # see ordering:

#                   0----------1 --> x

                    Lines += [int(data[3]), int(data[-2])-1, int(data[-1])-1]

                if int(data[1]) == 2:
                    """
                    we store [physical group, node1, node2, node3]
                    """
                    # see ordering:

#                    y
#                    ^
#                    |
#                    2
#                    |`\
#                    |  `\
#                    |    `\
#                    |      `\
#                    |        `\
#                    0----------1 --> x

                    Triangles += [int(data[3]), int(int(data[-3])-1), int(int(data[-2])-1), int(int(data[-1])-1)]

    return Verts, np.array(Lines).reshape(int(len(Lines)/3), 3), np.array(Triangles).reshape(int(len(Triangles)/4), 4)


"""-------------------------------------------------------------------------"""
"""                    CG                                  """
"""-------------------------------------------------------------------------"""
def cg(A, b, M, mute):
    """preconditioned cg method for

                A * u = b

        with M ~ A^-1 as preconditioner


        INPUT:
        A: function that does matrix*vector A(x) = A*x for
           A \in \R^{nxn} symmetric and positive definite

        b: right hand side, vector in \R^n

        M ~ A^-1: function that solves Ax = c approximately, that is, M(c) ~ x

        mute: if False then print option

        -----------------------------------------------------------------------
        REFERENCE:
         'Finite Element Methods and Fast Iterative Solvers'(A. Wathen), p.80
    """
    if M == 0:
        def M(x):
            return x

    tol = 10e-12
    maxiter = 2500
    n = len(b)
    u = np.zeros(n)
    r_alt = b - A(u)

    z_alt = M(r_alt)


    p = z_alt
    k = 0
    while k < maxiter:
        if mute == False:
            print('iteration STEP', k)
        alpha = np.dot(z_alt, r_alt) / np.dot(A(p), p)
        u = u + alpha * p
        r_neu = r_alt - alpha * A(p)

        if np.linalg.norm(r_neu)/np.linalg.norm(b) < tol:
            break

        z_neu = M(r_neu)

        beta = np.dot(z_neu, r_neu) / np.dot(z_alt, r_alt)
        p = z_neu + beta * p

        z_alt = z_neu
        r_alt = r_neu
        k = k+1

#    print
#    print 'number of cg iterations = ', k+1
#    print

    return u


def L_BFGS(k, m, s, y, ro, g, A):
    """
    m: number of vectors stored
    s_k = x_(k+1) - x_(k)
    y_k = gradient_f(x)_(k+1) - gradient_f(x)_(k)
    ro_k= 1./<y_k, s_k>
    g = gradient_f(x)_(k)
    M = matrix for scalarproduct, if 0 then identity

    s and y are stored as follows:
        s = np.array([ [ ...   s_1      ...],
                       [ ...   s_2      ...],

                               ...

                       [ ...   s_(k-1)  ...]  ])


    returns the decent direction

        z = H^(-1)*grad(f)

    such that the minimization iteration goes as follows:

        x_(k+1) = x_k - h * z

    """

    def M(x):
        return A.dot(x)

    q = g

    if k == 0:
        # first step = gradient
        return q

    else:

        I = min(k, m)

        alpha = np.zeros(k)

        # backward loop
        for i in range(I)[::-1]:
            alpha[i] = ro[i] * np.dot(s[i,], M(q))
            q = q - alpha[i] * y[i, :]

        # re-scale
        H_0 = np.dot(s[I - 1,], M(y[I - 1,])) / np.dot(y[I - 1,], M(y[I - 1,]))
        z = H_0 * q

        # forward loop
        for i in range(I):
            beta = ro[i] * np.dot(y[i,], M(z))
            z = z + s[i,] * (alpha[i] - beta)

        return z



# ==============================================================================
def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def load_sparse_csr(filename):
    loader = np.load(filename)
    return ss.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])
# ==============================================================================
"""-------------------------------------------------------------------------"""
"""                     NORM DICTIONARY                                     """
"""-------------------------------------------------------------------------"""
#=============================================================================#
#=============================================================================#
#=============================================================================#
def norm_2(x):
    return np.sqrt(x[0]**2 + x[1]**2)
def norm_1(x):
    return np.sum(np.abs(x), axis = 0)
def norm_inf(x):
    return np.max(np.abs(x), axis= 0)

norm_dict = {'L2': norm_2, 'L1': norm_1, 'Linf': norm_inf}


"""
*******************************************************************************
*******************************************************************************

     X       X  X   X          XXXXXX  X       XXXXXX
     X       X  XX  X          X       X       X    X
     X       X  X X X          XXX     X       XXXXXX
     X       X  X  XX          X       X       X    X
     XXXXXX  X  X   X          XXXXXX  XXXXXX  X    X
     
*******************************************************************************
*******************************************************************************
"""

"""========================================================================="""
"""                            MASS MATRIX    LinElas                              """
"""========================================================================="""
def LinElas_massmat(mesh):
    num_nodes = len(mesh.nodes)
    num_omega = len(mesh.omega)

    M_1d = ss.lil_matrix((num_nodes, num_nodes), dtype = float)
    P = np.array([[ 0.33333333,  0.33333333],
                  [ 0.2       ,  0.6       ],
                  [ 0.2       ,  0.2       ],
                  [ 0.6       ,  0.2       ]]).transpose()

    weights = 0.5 * np.array([-27./48, 25./48, 25./48, 25./48])

    def BASIS(v):
        return np.array([ 1. - v[0] - v[1], v[0], v[1]])

    PSI = BASIS(P)

    for l in range(num_omega):
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]
        det = abs(float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] ))

        for a in range(3):
            for b in range(3):
                a_nodes = np.where(mesh.nodes == T_num[a])[0][0] # find number of node to which element basis function at vertex a contributes
                b_nodes = np.where(mesh.nodes == T_num[b])[0][0]
                if a_nodes <= b_nodes:
                    M_1d[a_nodes, b_nodes] += det * (PSI[a] * PSI[b] * weights).sum()

    M = ss.lil_matrix((2*num_nodes, 2*num_nodes), dtype = float)
    D = ss.diags(M_1d.diagonal())
    M_1d = M_1d.transpose() + M_1d - D

    M[0:num_nodes, 0:num_nodes] = M_1d
    M[num_nodes:,  num_nodes:] = M_1d

    return M

"""========================================================================="""
"""                       STIFFNESS MATRIX     LinElas                             """
"""========================================================================="""
def LinElas_assembly(mu, lbd, mesh):

    num_omega = len(mesh.omega)
    num_nodes = len(mesh.nodes)

    A = ss.lil_matrix((2*num_nodes, 2*num_nodes), dtype = float)

    grad = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]

    def fun(l):
#    for l in range(num_omega):
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

        Mat = np.array( [T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
        det = abs(det)

        if isinstance(mu, float): # constant mu
            mu_loc = det * 0.5 * mu # = \int_T mu dx
        else: # locally varying mu
            idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

            # mu values on three vertices of the current triangle
            mu_vals = mu[idx]
            mu_loc = 1./6. * det * mu_vals.sum() # = \int_T mu dx, note that integral over reference element basis function is 1/6 for all


        for a in range(3):
            for b in range(3):
                a_nodes = np.where(mesh.nodes == T_num[a])[0][0]
                b_nodes = np.where(mesh.nodes == T_num[b])[0][0]

                grad_trans_a = iMat.transpose().dot(grad[a])
                grad_trans_b = iMat.transpose().dot(grad[b])

                # since A is symmetric we only assemble lower triangle
                # note: lbd == 0
                A[a_nodes, num_nodes + b_nodes] += 2 * (mu_loc * ( 0.5 * (grad_trans_a[1])*(grad_trans_b[0])) ) #bilin_2d(U_1, V_1[::-1], det, mu_loc)

                if a_nodes <= b_nodes:
                    A[a_nodes,             b_nodes]             += 2 * (mu_loc * (grad_trans_a[0]*grad_trans_b[0] + 0.5 * (grad_trans_a[1])*(grad_trans_b[1])) )# bilin_2d(U_1, V_1, det, mu_loc)
                    A[num_nodes + a_nodes, num_nodes + b_nodes] += 2 * (mu_loc * (grad_trans_a[1]*grad_trans_b[1] + 0.5 * (grad_trans_a[0])*(grad_trans_b[0])) )#bilin_2d(U_1[::-1], V_1[::-1], det, mu_loc)


    list(map(fun, range(num_omega)))
    # reflect lower triangular part of A on diagonal to construct full matrix
    A = A.tocsr()
    D = ss.diags(A.diagonal())
    A = A.transpose() + A - D
    A = A.tolil()

    # incorporate Dirichlet-data (replace column/row by unit vector)
    for k in mesh.boundary:

        A[k,:] = ss.eye(1, 2*num_nodes, k).tocsr()
        A[:, k] = ss.eye(1, 2*num_nodes, k ).tocsr().transpose()

        A[num_nodes + k,:] = ss.eye(1, 2*num_nodes, num_nodes + k).tocsr()
        A[:, num_nodes + k] = ss.eye(1, 2*num_nodes, num_nodes + k ).tocsr().transpose()

    return A

#def LinElas_assembly_orig(mu, lbd, mesh):
#
#    def bilin_2d(U,V,det, mu_loc):
#        """U, V = iMat^-T * grad (already from transformed formula) """
#        return det * 0.5 * (lbd*(U[0,0]+U[1,1])*(V[0,0]+V[1,1])) \
#                    + 2 * (mu_loc * (U[0,0]*V[0,0] + U[1,1]*V[1,1]  \
#                               + 0.5 * (U[0,1]+U[1,0])*(V[0,1]+V[1,0])) )
#
#    num_omega = len(mesh.omega)
#    num_nodes = len(mesh.nodes)
#
#    A = ss.lil_matrix((2*num_nodes, 2*num_nodes), dtype = float)
#
#    grad = [np.array([-1, -1]), np.array([1, 0]), np.array([0, 1])]
#
#    for l in range(num_omega):
#        T_num = mesh.omega[l,1:].tolist()
#        T = mesh.verts[T_num]
#
#        Mat = np.array( [T[1] - T[0],T[2] - T[0] ]).transpose()
#        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
#        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
#        det = abs(det)
#
#        if isinstance(mu, float): # constant mu
#            mu_loc = det * 0.5 * mu # = \int_T mu dx
#        else: # locally varying mu
#            idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]
#
#            # mu values on three vertices of the current triangle
#            mu_vals = mu[idx]
#            mu_loc = 1./6. * det * mu_vals.sum() # = \int_T mu dx, note that integral over reference element basis function is 1/6 for all
#        print mu_loc
#        for a in range(3):
#            for b in range(3):
#                a_nodes = np.where(mesh.nodes == T_num[a])[0][0]
#                b_nodes = np.where(mesh.nodes == T_num[b])[0][0]
#
#                U_1 = np.vstack((iMat.transpose().dot(grad[a]), np.zeros(2)))
#                U_2 = np.vstack((np.zeros(2), iMat.transpose().dot(grad[a])))
#                V_1 = np.vstack((iMat.transpose().dot(grad[b]), np.zeros(2)))
#                V_2 = np.vstack((np.zeros(2), iMat.transpose().dot(grad[b])))
#
#                # since A is symmetric we only assemble lower triangle
#                A[a_nodes, num_nodes + b_nodes] += bilin_2d(U_1, V_2, det, mu_loc)
#
#                if a_nodes <= b_nodes:
#                    A[a_nodes,             b_nodes]             += bilin_2d(U_1, V_1, det, mu_loc)
#                    A[num_nodes + a_nodes, num_nodes + b_nodes] += bilin_2d(U_2, V_2, det, mu_loc)
#
#    # reflect lower triangular part of A on diagonal to construct full matrix
#    A = A.tocsr()
#    D = ss.diags(A.diagonal())
#    A = A.transpose() + A - D
#    A = A.tolil()
#
#    # incorporate Dirichlet-data (replace column/row by unit vector)
#    for k in mesh.boundary:
#
#        A[k,:] = ss.eye(1, 2*num_nodes, k).tocsr()
#        A[:, k] = ss.eye(1, 2*num_nodes, k ).tocsr().transpose()
#
#        A[num_nodes + k,:] = ss.eye(1, 2*num_nodes, num_nodes + k).tocsr()
#        A[:, num_nodes + k] = ss.eye(1, 2*num_nodes, num_nodes + k ).tocsr().transpose()
#
#    return A

"""========================================================================="""
"""                          SOURCE TERM                                    """
"""========================================================================="""
def source_assembly(fx,fy,M, mesh):

    num_nodes = len(mesh.nodes)

    def source(v):
        return np.array([fx(v), fy(v)])

    B_phys = np.hstack((fx(mesh.vertices[mesh.nodes].transpose()), fy(mesh.vertices[mesh.nodes].transpose())))
    B = M.dot(B_phys)

    # incorporate Dirichlet-data
    for k in mesh.boundary:
        B[k] = 0
        B[num_nodes + k] = 0

    return B

"""========================================================================="""
"""                       PLOT ROUTINES                                     """
"""========================================================================="""
def plot_vecfield(U, mesh, title, **kwargs):
    reshape = kwargs.get('reshape', True)
    verts = kwargs.get('verts', [0])
    scale = kwargs.get('scale',1)
    interaction_domain = kwargs.get('interaction_domain', False)

    if reshape:
        U = U.reshape(len(mesh.nodes), 2 , order = 'F')


    if verts[0] == 0:
        verts_here = mesh.vertices
    else:
        verts_here = verts

    if interaction_domain:
        X, Y =  verts_here[:,0],  verts_here[:,1]
    else:
        X, Y =  verts_here[mesh.nodes][:,0],  verts_here[mesh.nodes][:,1]


    U_nor = np.linalg.norm(U, axis=1)
    norm = matplotlib.colors.Normalize()
    cm = matplotlib.cm.rainbow
    sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)

    plt.figure(title)
    plt.triplot(verts_here[:,0], verts_here[:,1], mesh.elements[:, 1:], color ='black', linewidth =0.3, alpha = 0.8)
#    for i in range(len(mesh.omega)):
#        plt.gca().add_patch(plt.Polygon(verts_here[mesh.omega[i,1:]].tolist(), closed=True, fill = False, color = 'black', linewidth = 0.08))
#    colormap = matplotlib.cm.rainbow

    norm.autoscale(U_nor)

    sm.set_array([])
    if scale:
        plt.quiver(X, Y, U[:,0], U[:,1],  color=cm(norm(U_nor)), angles='xy', scale_units='xy', scale=1)  #scale=5*np.max(U), scale_units='inches')
    else:
        plt.quiver(X, Y, U[:,0], U[:,1],  color=cm(norm(U_nor)))  #scale=5*np.max(U), scale_units='inches')
    plt.colorbar(sm)
    plt.axis('equal')

def plot_newmesh(mesh, U):
    U = U.reshape(len(mesh.nodes), 2 , order = 'F')
    U_new = np.zeros((len(mesh.vertices), 2))
    U_new[mesh.nodes, :] = U
    verts_new = copy(mesh.vertices)
    verts_new += U_new
    U_nor = np.linalg.norm(U_new, axis=1)

    plt.figure('Deformed Mesh')
    for i in range(len(mesh.omega)):
        plt.gca().add_patch(plt.Polygon(verts_new[mesh.omega[i,1:]].tolist(), closed=True, fill = False, color = 'black', linewidth = 0.15))
    plt.tricontourf(verts_new[:,0],verts_new[:,1], mesh.elements[:, 1:], U_nor, 150, interpolation='gaussian', cmap =plt.cm.get_cmap('rainbow'))
    plt.colorbar()




"""
*******************************************************************************
*******************************************************************************

      S H A P E   S P E C I F I C
     
*******************************************************************************
*******************************************************************************
"""


def interpolate(u1, mesh1, mesh2, **kwargs):
    """
    linearly interpolate u1 (based on mesh1) onto mesh2
    """
    deform = kwargs.get('deform', 0)

    u1[mesh1.boundary] = np.zeros(len(mesh1.boundary))
    verts2 = mesh2.vertices + deform

    u2 = si.griddata(mesh1.vertices[mesh1.nodes], u1, verts2[mesh2.nodes])
    u2[mesh2.boundary] = np.zeros(len(mesh2.boundary))

    return u2


def plot_shape(mesh, **kwargs):
    color = kwargs.get('color', 'grey')
    fill = kwargs.get('fill', True)
    alpha = kwargs.get('alpha', 0.4)
    omega_1 = mesh.elements[np.where(mesh.elements[:, 0] == 1)[0]]

    for k in range(len(omega_1)):
        plt.gca().add_patch(plt.Polygon(mesh.vertices[omega_1[k, 1:]], closed=True, fill = fill, color = color, alpha = alpha))
        plt.gca().add_patch(plt.Polygon(mesh.vertices[omega_1[k, 1:]], closed=True, fill = False, color ='black', linewidth = 0.3))




def target_function(mesh, source, u_bar, mesh_bar, eps, Norm, gam, num_cores, retriangulate,local,approx, diff_coeff, **kwargs):

    lap_reg = kwargs.get('lap_reg', 0)
    deform = kwargs.get('deform', 0)

    nu = kwargs.get('nu', 0)

    M = mass_matrix2(mesh, deform = deform)

    # compute u
    if local:
        A = Laplace_para(mesh,diff_coeff, num_cores, deform = deform)
    else:
        if approx:
            A = assembly_coupling_approx(mesh, gam,  retriangulate, Norm, num_cores, deform = deform)
        else:
            A = assembly_coupling(mesh, gam,  retriangulate, Norm, num_cores, deform = deform)

        if lap_reg > 0:
            A += lap_reg * Laplace_para(mesh, [1,1], num_cores, deform = deform)

#    if local:
#        A = Laplace_para(mesh,diff_coeff, num_cores, deform = deform)
#    else:
#        if approx:
#            A = assembly_coupling_approx(mesh, gam,  retriangulate, Norm, num_cores, deform = deform) + lap_reg * Laplace_para(mesh,[1,1], 1, deform = deform)
#        else:
#            A = assembly_coupling(mesh, gam, retriangulate, Norm, num_cores, deform = deform) + lap_reg * Laplace_para(mesh, [1,1], 1, deform = deform)

    b = source_term_discon(mesh, source, deform = deform)
    u = solve(mesh,A,b)

    # interpolate u_bar
    u_bar = interpolate(u_bar, mesh_bar, mesh, deform = deform)

    return 0.5 * np.dot(u-u_bar, M.dot(u-u_bar)) + nu * j_reg(mesh, deform = deform)


def target_function_adj(mesh, source, u_bar, mesh_bar, eps, Norm, gam, num_cores, retriangulate,local,approx, diff_coeff, **kwargs):

    lap_reg = kwargs.get('lap_reg', 0)
    deform = kwargs.get('deform', 0)

    nu = kwargs.get('nu', 0)

    M = mass_matrix2(mesh, deform = deform)

    # compute u
    if local:
        A = Laplace_para(mesh,diff_coeff, num_cores, deform = deform)
    else:
        if approx:
            A = assembly_coupling_approx(mesh, gam,  retriangulate, Norm, num_cores, deform = deform).transpose()
        else:
            A = assembly_coupling(mesh, gam,  retriangulate, Norm, num_cores, deform = deform).transpose()

        if lap_reg > 0:
            A += lap_reg * Laplace_para(mesh, [1,1], num_cores, deform = deform)

    b = source_term_discon(mesh, source, deform = deform)
    u = solve(mesh,A,b)

    # interpolate u_bar
    u_bar = interpolate(u_bar, mesh_bar, mesh, deform = deform)

    return 0.5 * np.dot(u-u_bar, M.dot(u-u_bar)) + nu * j_reg(mesh, deform = deform)


def target_function_klarified(mesh, source, u_bar, mesh_bar, eps, Norm, gam, num_cores, retriangulate,local,approx, diff_coeff, **kwargs):

    lap_reg = kwargs.get('lap_reg', 0)
    deform = kwargs.get('deform', 0)

    nu = kwargs.get('nu', 0)

    M = mass_matrix2(mesh, deform = deform)

    # compute u
    if local:
        A = Laplace_para(mesh,diff_coeff, num_cores, deform = deform)
    else:

        A = assemble(mesh, Px, weightsxy, weightsxy, eps , deltaVertices = deform)[0][:,0:mesh.nV_Omega]
        A = ss.csr_matrix(A)
        if lap_reg > 0:
            A += lap_reg * Laplace_para(mesh, [1,1], num_cores, deform = deform)

    b = source_term_discon(mesh, source, deform = deform)
    u = solve(mesh,A,b)

    # interpolate u_bar
    u_bar = interpolate(u_bar, mesh_bar, mesh, deform = deform)

    return 0.5 * np.dot(u-u_bar, M.dot(u-u_bar)) + nu * j_reg(mesh, deform = deform)


def is_self_intersecting(mesh, **kwargs):
    """
    tests if polygon which determines interface is simple closed or not
    brute-force approach

    0: simple closed
    1: self-intersecting
    """

    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    decide = 0

    interface_omega = []
    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        if bool(set(mesh.shape_interface) & set(T_num)) and len(list(set(mesh.shape_interface) & set(T_num))) ==2 and mesh.omega[l,0]==1:
            interface_omega += [l]

    for l in interface_omega:

        T_num = mesh.omega[l,1:].tolist()
        T_interface = list(set(T_num)&set(mesh.shape_interface))
        xk, xj = verts[T_interface[0]], verts[T_interface[1]]
        line_1 = [xk, xj]

        for k in interface_omega:
            if k != l:
                T_num = mesh.omega[k,1:].tolist()
                T_interface = list(set(T_num)&set(mesh.shape_interface))
                xk, xj = verts[T_interface[0]], verts[T_interface[1]]
                line_2 = [xk, xj]

                if are_intersecting_lines(line_1, line_2):
                    """ plot """
    #                plt.figure()
    #                plt.gca().add_patch(plt.Polygon(line_1, color = 'red'))
    #                plt.gca().add_patch(plt.Polygon(line_2))
                    decide = 1
                    break
        if decide:
            break

    return decide

def is_out_of_omega(mesh, **kwargs):
    """

    """

    deform = kwargs.get('deform', 0)
    verts = mesh.vertices + deform
    decide = 0

    tol = 0.01

    for l in range(len(mesh.shape_interface)):
        point = verts[mesh.shape_interface[l]]
        if np.max(np.abs(point - 0.5 * np.ones(2))) > 0.5-tol:
            decide = 1
            break

    return decide


def j_reg(mesh, **kwargs):

    deform = kwargs.get('deform', 0)
    Length = 0
    verts = mesh.vertices + deform

    interface_omega = []
    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        if bool(set(mesh.shape_interface) & set(T_num)) and len(list(set(mesh.shape_interface) & set(T_num))) ==2 and mesh.omega[l,0]==1:
            interface_omega += [l]

    for l in interface_omega:

        T_num = mesh.omega[l,1:].tolist()

        T_interface = list(set(T_num)&set(mesh.shape_interface))

        xk, xj = verts[T_interface[0]], verts[T_interface[1]]

        """ plot """
#        plt.gca().add_patch(plt.Polygon([xk,xj]))

        Length += np.linalg.norm(xk - xj)

    return Length


def shape_derivative_peri_reg(mesh):

    num_nodes = len(mesh.nodes)

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    # find all elements in omega_1 which connect to the interface
    interface_omega_1 = []
    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        if bool(set(mesh.shape_interface) & set(T_num)) and len(list(set(mesh.shape_interface) & set(T_num))) ==2 and mesh.omega[l,0]==1:
            interface_omega_1 += [l]

    # find all elements in omega_1 which connect to the interface
    interface_omega_2 = []
    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        if bool(set(mesh.shape_interface) & set(T_num)) and len(list(set(mesh.shape_interface) & set(T_num))) ==2 and mesh.omega[l,0]==2:
            interface_omega_2 += [l]


    dV = np.zeros(2*len(mesh.nodes))

    for counter in range(len(interface_omega_1)):

        l = interface_omega_1[counter]

        T_num = mesh.omega[l,1:].tolist()

        T_interface = list(set(T_num)&set(mesh.shape_interface))

        xk, xj = mesh.vertices[T_interface[0]], mesh.vertices[T_interface[1]]
        length = np.linalg.norm(xk - xj)

        T = mesh.vertices[T_num]

        Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])

        grad_trans = iMat.transpose().dot(gradient.transpose()).transpose()

        midpoint = 0.5 * (xk+xj)
        n1 = np.array([-(xk-xj)[1], (xk-xj)[0]])
        n2 = -n1
        bary = 1./3. * (T[0]+T[1]+T[2])
        if np.linalg.norm(midpoint+n1-bary) > np.linalg.norm(midpoint+n2-bary):
            normal_vec = n1
        else:
            normal_vec = n2

        """ for plot """
#        plt.plot(bary[0], bary[1], 'bo')
#        plt.gca().add_patch(plt.Polygon(T, closed=True,  color = 'grey', alpha = 0.2, fill = True, linewidth = 2))
#        plt.gca().add_patch(plt.Polygon([midpoint, midpoint+normal_vec], closed=True, fill = False, color = 'red', alpha = 1, linewidth = 2))

        for a in [T_num.index(T_interface[aa]) for aa in range(2)]:

            a_nodes = np.where(mesh.nodes == T_num[a])[0][0]

            div_V_x = gradient[a,:].dot(iMat[:,0])#Div_V[a, 0] # = div(psi_a, 0)
            div_V_y = gradient[a,:].dot(iMat[:,1])#Div_V[a, 1] # = div(0, psi_a)

            # Jacobian of already transformed vector fields
            jac_V_x = np.vstack((grad_trans[a], np.zeros(2)))  # = Jac(psi_a, 0)
            jac_V_y = np.vstack((np.zeros(2)  , grad_trans[a]))# = Jac(0, psi_a)

            dV[a_nodes] += length * (div_V_x - normal_vec.dot(jac_V_x.dot(normal_vec)))
            dV[num_nodes + a_nodes] += length * (div_V_y - normal_vec.dot(jac_V_y.dot(normal_vec)))

    return dV


def a_loc(mesh, u, v):

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    a_loc = 0

    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

        Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])

        det = abs(det)

        idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

        u_vals = u[idx]
        v_vals = v[idx]

        grad_trans     = iMat.transpose().dot(gradient.transpose()).transpose()
        gradient_u     = grad_trans.transpose().dot(u_vals)
        gradient_v     = grad_trans.transpose().dot(v_vals)

        a_loc += 0.5 * det * gradient_u.dot(gradient_v)

    return a_loc

def shape_derivative_laplace(mesh, u, v, u_bar, source, diff_coeff):

    num_nodes = len(mesh.nodes)

    weights = weights2
    PSI = PSI_2

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    dV = np.zeros(2*len(mesh.nodes))

    for l in range(len(mesh.omega)):
        label_l = mesh.omega[l, 0]
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

#        interface = [mesh.nodes.index(mesh.shape_interface)]

        # theoretically only vector fields with support intersecting the interface
        # have contribution to the shape derivative
        # in order to reduce numerical noise we only assemble for precisely those

        if 1:#bool(set(mesh.shape_interface) & set(T_num)):#

#            plt.gca().add_patch(plt.Polygon(T, fill = False, closed = True ) )

            Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
            det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
            iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
#            print mesh.omega[l,0], det
            det = abs(det)

#            def Phi_T(y):
#                return np.repeat(T[0][:,np.newaxis], len_P**2, axis=1) +  Mat.dot(y)

            idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

            # u and u_bar values on three vertices of the current triangle
            u_vals = u[idx]
            v_vals = v[idx]
            u_bar_vals = u_bar[idx]

            # u and u_bar evaluated at the quadrature points P, X_P and Y_P
            u_vals_P = PSI.transpose().dot(u_vals)     # for single integral
            v_vals_P   = PSI.transpose().dot(v_vals)   # for single integral
            u_bar_vals_P = PSI.transpose().dot(u_bar_vals)

            """plot for test if interpolation works"""
            """interpolated function values > should look like the other plots"""
    #        P_trans = np.repeat(T[0][:,np.newaxis], len_P, axis=1) +  Mat.dot(P)
    #        plt.tricontourf(P_trans[0,], P_trans[1,],u_bar_vals_P,100,interpolation='gaussian',cmap =plt.cm.get_cmap('rainbow'), vmin = min(u_bar), vmax = max(u_bar) )

            # note these need to be already transformed gradients M^-T * grad
            grad_trans     = iMat.transpose().dot(gradient.transpose()).transpose()
            gradient_u_bar = grad_trans.transpose().dot(u_bar_vals)
            gradient_u     = grad_trans.transpose().dot(u_vals)
            gradient_v     = grad_trans.transpose().dot(v_vals)

            # 3x2 array containing transformed divergence values of all basis funcs
            # for example Div_V[1,2] = div((0, psi_1)), Div_V[2,0] = div((psi_2, 0))
#            Div_V = div(iMat)

            # constant value of part 1b (constant in V)
            # note that source f is constant
            # it is f1 = source[0] on shape which is labeled 1
            # it is f2 = source[1] on shape which is labeled 2
            part_1b_const = ((0.5 * (u_vals_P - u_bar_vals_P)**2 - source[mesh.omega[l,0]-1] * v_vals_P) * weights ).sum()

            # per element we have 6 basis functions (3 in each dimension)
            for a in range(3):

                # given the element and the basis function, a_nodes gives the
                # component of dV to which element basis function psi contributes to
                # a_nodes = index for basis function in x-coordinate, i.e., (psi_a, 0)
                # num_nodes + a_nodes = index for basis function in  y-coordinate, i.e., (0, psi_a)
                a_nodes = np.where(mesh.nodes == T_num[a])[0][0]

                # contribution of each term coming from basisfunction in each dimension
                # (x-axis and y-axis)
                # will be added together and form dV[a_nodes] (=X) and dV[num_nodes + a_nodes] (=Y)
                X, Y = 0,0

                # divergence from already transformed vector fields
                div_V_x = gradient[a,:].dot(iMat[:,0])#Div_V[a, 0] # = div(psi_a, 0)
                div_V_y = gradient[a,:].dot(iMat[:,1])#Div_V[a, 1] # = div(0, psi_a)

                # Jacobian of already transformed vector fields
                jac_V_x = np.vstack((grad_trans[a], np.zeros(2)))  # = Jac(psi_a, 0)
                jac_V_y = np.vstack((np.zeros(2)  , grad_trans[a]))# = Jac(0, psi_a)

                # vaules of: (transformed gradient \nabla u_bar)^T (V)
                # it is a linear combination
                grad_u_bar_x = gradient_u_bar[0] * PSI[a] # if V = (psi_a, 0)
                grad_u_bar_y = gradient_u_bar[1] * PSI[a] # if V = (0, psi_a)

                # 1a
                X += - det * ( ( (u_vals_P - u_bar_vals_P) * grad_u_bar_x ) * weights ).sum()
                Y += - det * ( ( (u_vals_P - u_bar_vals_P) * grad_u_bar_y ) * weights ).sum()

                # 1b
                X +=   det * div_V_x * part_1b_const
                Y +=   det * div_V_y * part_1b_const

                # 2a
                X += - diff_coeff[label_l-1] * 0.5 * det * gradient_u.dot((jac_V_x + jac_V_x.transpose()).dot(gradient_v))
                Y += - diff_coeff[label_l-1] * 0.5 * det * gradient_u.dot((jac_V_y + jac_V_y.transpose()).dot(gradient_v))

                # 2b
                X += diff_coeff[label_l-1] * div_V_x * 0.5 * det * gradient_u.dot(gradient_v)
                Y += diff_coeff[label_l-1] * div_V_y * 0.5 * det * gradient_u.dot(gradient_v)

#                if T_num[a] in mesh.shape_interface:
#                    print a_nodes, T_num[a],X,Y
#                    plt.plot(mesh.verts[a_nodes][0], mesh.verts[a_nodes][1], 'ro')
                dV[a_nodes] += X
                dV[num_nodes + a_nodes] += Y

    # incorporate Dirichlet-data
    # (actually not necessary since we only consider those intersecting with interface)
    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0


    return dV



def div_V(mesh):

    num_nodes = len(mesh.nodes)
    gradient = np.array([[-1, -1], [1, 0], [0, 1]])
    dV = np.zeros(2*len(mesh.nodes))

    for l in range(len(mesh.omega)):

        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]
        Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])

        for a in range(3):
            a_nodes = np.where(mesh.nodes == T_num[a])[0][0]
            dV[a_nodes] += gradient[a,:].dot(iMat[:,0])
            dV[num_nodes + a_nodes] +=  gradient[a,:].dot(iMat[:,1])

    return dV

def shape_derivative_laplace_partly_a(mesh, u, v, u_bar, source, diff_coeff):
    num_nodes = len(mesh.nodes)

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    dV = np.zeros(2*len(mesh.nodes))

    for l in range(len(mesh.omega)):
        label_l = mesh.omega[l, 0]

        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

        Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
        det = abs(det)

        idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

        u_vals = u[idx]
        v_vals = v[idx]

        grad_trans     = iMat.transpose().dot(gradient.transpose()).transpose()
        gradient_u     = grad_trans.transpose().dot(u_vals)
        gradient_v     = grad_trans.transpose().dot(v_vals)


        for a in range(3):

            a_nodes = np.where(mesh.nodes == T_num[a])[0][0]

            X, Y = 0,0

            div_V_x = gradient[a,:].dot(iMat[:,0])#Div_V[a, 0] # = div(psi_a, 0)
            div_V_y = gradient[a,:].dot(iMat[:,1])#Div_V[a, 1] # = div(0, psi_a)

            X += diff_coeff[label_l-1] * div_V_x * 0.5 * det * gradient_u.dot(gradient_v)
            Y += diff_coeff[label_l-1] * div_V_y * 0.5 * det * gradient_u.dot(gradient_v)

            dV[a_nodes] += X
            dV[num_nodes + a_nodes] += Y


    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0


    return dV

def shape_derivative_laplace_partly_b(mesh, u, v, u_bar, source, diff_coeff):

    num_nodes = len(mesh.nodes)

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    dV = np.zeros(2*len(mesh.nodes))

    for l in range(len(mesh.omega)):
        label_l = mesh.omega[l, 0]
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

        Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
        det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
        iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
        det = abs(det)

        idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

        u_vals = u[idx]
        v_vals = v[idx]

        grad_trans     = iMat.transpose().dot(gradient.transpose()).transpose()
        gradient_u     = grad_trans.transpose().dot(u_vals)
        gradient_v     = grad_trans.transpose().dot(v_vals)

        for a in range(3):

            a_nodes = np.where(mesh.nodes == T_num[a])[0][0]

            X, Y = 0,0

            jac_V_x = np.vstack((grad_trans[a], np.zeros(2)))  # = Jac(psi_a, 0)
            jac_V_y = np.vstack((np.zeros(2)  , grad_trans[a]))# = Jac(0, psi_a)

            X += - diff_coeff[label_l-1] * 0.5 * det * gradient_u.dot((jac_V_x + jac_V_x.transpose()).dot(gradient_v))
            Y += - diff_coeff[label_l-1] * 0.5 * det * gradient_u.dot((jac_V_y + jac_V_y.transpose()).dot(gradient_v))

            dV[a_nodes] += X
            dV[num_nodes + a_nodes] += Y

    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0

    return dV



def shape_derivative_intersection(mesh, u, v, u_bar, source):

    num_nodes = len(mesh.nodes)

    # for quadrature
    weights = weights2
    PSI = PSI_2

    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    dV = np.zeros(2*len(mesh.nodes))

    for l in range(len(mesh.omega)):
        T_num = mesh.omega[l,1:].tolist()
        T = mesh.vertices[T_num]

        if True:#bool(set(interface) & set(T_num)):

            Mat = np.array([ T[1] - T[0],T[2] - T[0] ]).transpose()
            det = float((T[1] - T[0])[0] * (T[2] - T[0])[1] - (T[1] - T[0])[1] * (T[2] - T[0])[0] )
            iMat = 1./det * np.array([ [Mat[1,1], -Mat[0,1]], [-Mat[1,0], Mat[0,0]]  ])
            det = abs(det)

            idx = [np.where(mesh.nodes == j)[0][0] for j in T_num]

            u_vals = u[idx]
            v_vals = v[idx]
            u_bar_vals = u_bar[idx]

            u_vals_P = PSI.transpose().dot(u_vals)     # for single integral
            v_vals_P = PSI.transpose().dot(v_vals)
            u_bar_vals_P = PSI.transpose().dot(u_bar_vals)

            grad_trans     = iMat.transpose().dot(gradient.transpose()).transpose()
            gradient_u_bar = grad_trans.transpose().dot(u_bar_vals)

            objective = ((0.5 * (u_vals_P - u_bar_vals_P)**2 ) * weights ).sum()

            for a in range(3):

                a_nodes = np.where(mesh.nodes == T_num[a])[0][0]

                X, Y = 0,0

                div_V_x = gradient[a,:].dot(iMat[:,0])
                div_V_y = gradient[a,:].dot(iMat[:,1])

                grad_u_bar_x = gradient_u_bar[0] * PSI[a] # if V = (psi_a, 0)
                grad_u_bar_y = gradient_u_bar[1] * PSI[a] # if V = (0, psi_a)

                # 1
                X += - det * ( ( (u_vals_P - u_bar_vals_P) * grad_u_bar_x ) * weights ).sum()
                Y += - det * ( ( (u_vals_P - u_bar_vals_P) * grad_u_bar_y ) * weights ).sum()

#                # 2
#                X += det * div_V_x * objective
#                Y += det * div_V_y * objective

                # 3
                """annihilated???"""
                X += - det * div_V_x * ((source[mesh.omega[l,0]-1] * v_vals_P) * weights ).sum()
                Y += - det * div_V_y * ((source[mesh.omega[l,0]-1] * v_vals_P) * weights ).sum()

                dV[a_nodes]             += X
                dV[num_nodes + a_nodes] += Y

    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0

    return dV

def shape_derivative_nonlocal_divV_bilin(mesh, retriangulate, Norm, gam, u, v, num_cores):

    num_nodes = len(mesh.nodes)
    gradient = np.array([[-1, -1], [1, 0], [0, 1]])

    liste = range(len(mesh.omega)) #interface_omega
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole thing
    def aux(m):
        dV = np.zeros(2*len(mesh.nodes))
        for i in pieces[m]:

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]

            label_i = mesh.omega[i, 0]
            eps_i = gam['eps'+str(label_i)]

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
            det_T_i = abs(det_T_i)

            i_triangles = np.where(np.all(mesh.elements == mesh.omega[i], axis=1))[0][0]
            hash_i = np.where(norm_dict[Norm]((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()

            idx = [np.where(mesh.nodes == j)[0][0] for j in T_i]

            u_vals = u[idx]
            v_vals = v[idx]

            u_vals_P = PSI.transpose().dot(u_vals)     # for single integral
            v_vals_P = PSI.transpose().dot(v_vals)     # for single integral

            for a in range(3):
                a_nodes = np.where(mesh.nodes == T_i[a])[0][0]

                X_val, Y_val = 0,0

                div_V_x = gradient[a,:].dot(iMat_i[:,0])
                div_V_y = gradient[a,:].dot(iMat_i[:,1])

                for j in hash_i:
                    label_j = mesh.elements[j, 0]
                    gam_ij = gam[str(label_i)+str(label_j)]

                    T_j = mesh.elements[j, 1:].tolist()
                    T_j_v = mesh.vertices[T_j]
                    Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                    det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]

                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    def iPhi_j(y):
                        return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))


                    if label_j != 3:
                        idx_j = [np.where(mesh.nodes == j)[0][0] for j in T_j]
                        # u values on three vertices of T_j
                        u_vals_j = u[idx_j]

                    def u_j(x):
                        return BASIS(x).transpose().dot(u_vals_j)#u_vals_j[0] * basis[0](x) + u_vals_j[1] * basis[1](x) + u_vals_j[1] * basis[1](x)


                    def I1(x):
                        x_trans = (T_i_v[0]+Mat_i.dot(x))
                        integral_L2, integral_convu = 0., 0.

                        aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)

                        def inner(tri, gam_ij):
                            tri = np.array(tri)
                            Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
                            det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
                            def Phi_l(y):
                                return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)

                            GAM = det_l * gam_ij(aux, Phi_l(P2)) * weights2

                            if label_j != 3:
                                return  GAM.sum(), (u_j(iPhi_j(Phi_l(P2))) * GAM ).sum()
                            else:
                                return  GAM.sum(), 0.

                        tris = retriangulate(x_trans, T_j_v, Norm, eps_i )
                        if len(tris) != 0:
                            for tri in tris:
                                val_L2, val_convu = inner(tri, gam_ij)
                                integral_L2    += val_L2
                                integral_convu += val_convu

                        return np.array([integral_L2, integral_convu])

                    I = np.array(list(map(I1, P.transpose()))).transpose()

                    ### weighted L2 product part ###
                    L2 = det_T_i * (u_vals_P * v_vals_P * I[0] * weights).sum()
                    X_val += div_V_x * L2
                    Y_val += div_V_y * L2

                    ### convolution part ###
                    if label_j != 3:
                        convu = det_T_i * (v_vals_P * I[1] * weights).sum()
                        X_val += - div_V_x * convu
                        Y_val += - div_V_y * convu

                dV[a_nodes] += 2 *  X_val
                dV[num_nodes + a_nodes] += 2 * Y_val

        return dV

    p = Pool(num_cores)
    dVs = p.map(aux, range(num_cores))

    dV = np.zeros(2*len(mesh.nodes))
    for i in range(num_cores):
        dV += dVs[i]

    del dVs
    p.close()
    p.join()
    p.clear()

    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0

    return dV

def shape_derivative_nonlocal_divV_bilin_approx(mesh, gam, u, v, num_cores):
    gradient = np.array([[-1, -1], [1, 0], [0, 1]])
    num_nodes = len(mesh.nodes)

    liste = range(len(mesh.omega))#mesh.omega#interface_omega
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):
        dV = np.zeros(2*len(mesh.nodes))

        for i in pieces[m]:
            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]
            label_i = mesh.omega[i, 0]

            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1]
            iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
            det_T_i = abs(det_T_i)

            def Phi_i(y):
                return  Mat_i.dot(y)  + np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1)

            i_triangles = np.where(np.all(mesh.elements == mesh.omega[i], axis=1))[0][0]
            hash_i = np.where(norm_dict[Norm]((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()

            idx = [np.where(mesh.nodes == j)[0][0] for j in T_i]

            # u and u_bar values on three vertices of the current triangle
            u_vals = u[idx]
            v_vals = v[idx]

            for a in range(3):
                a_nodes = np.where(mesh.nodes == T_i[a])[0][0]

                X_val, Y_val = 0,0

                div_V_x = gradient[a,:].dot(iMat_i[:,0])
                div_V_y = gradient[a,:].dot(iMat_i[:,1])

                for j in hash_i:
                    label_j = mesh.elements[j, 0]
                    gam_ij = gam[str(label_i)+str(label_j)]

                    T_j = mesh.elements[j, 1:].tolist()
                    T_j_v = mesh.vertices[T_j]
                    Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()
                    det_T_j = abs(Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1])
                    def Phi_j(y):
                        return Mat_j.dot(y) + np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1)

                    if label_j != 3:
                        idx_j = [np.where(mesh.nodes == j)[0][0] for j in T_j]

                        # u values at the vertices of T_j
                        u_vals_j = u[idx_j]

                    u_vals_X_P   = PSI_X.transpose().dot(u_vals) # for double integral
                    v_vals_X_P   = PSI_X.transpose().dot(v_vals) # for double integral
                    u_vals_Y_P_j = PSI_Y.transpose().dot(u_vals_j) # for double integral

                    ### weighted L2 product part ###
                    L2 = det_T_i * det_T_j * ( u_vals_X_P *  v_vals_X_P  * W * gam_ij(Phi_i(X),Phi_j(Y))).sum()
                    X_val += div_V_x * L2
                    Y_val += div_V_y * L2

                    ### convolution part ###
                    if label_j != 3:
                        convu = det_T_i * det_T_j * ( v_vals_X_P *  u_vals_Y_P_j  * W * gam_ij(Phi_i(X),Phi_j(Y))).sum()
                        X_val += - div_V_x * convu
                        Y_val += - div_V_y * convu

                dV[a_nodes] += X_val
                dV[num_nodes + a_nodes] += Y_val

        return 2 * dV

    p = Pool(num_cores)
    dVs = p.map(aux, range(num_cores))

    dV = np.zeros(2*len(mesh.nodes))
    for i in range(num_cores):
        dV += dVs[i]

    del dVs
    p.close()
    p.join()
    p.clear()

    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0

    return dV


def shape_derivative_nonlocal_divV_bilin_approx_2(mesh, gam, u, v, num_cores):
    """
    bilinear form + (div V(x) + div V(y))
    """
    gradient = np.array([[-1, -1], [1, 0], [0, 1]])
    num_nodes = len(mesh.nodes)

    liste = range(len(mesh.omega))#mesh.omega#interface_omega
    random.shuffle(list(liste))
    pieces = np.array_split(liste, num_cores)

    # For every pieces compute the whole matrix
    def aux(m):
        dV = np.zeros(2*len(mesh.nodes))
        
        for i in pieces[m]:
            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]
            label_i = mesh.omega[i, 0]
    
            T_i = mesh.omega[i, 1:].tolist()
            T_i_v = mesh.vertices[T_i]
            Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
            det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] 
            iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
            det_T_i = abs(det_T_i)            
    
            def Phi_i(y):
                return  Mat_i.dot(y)  + np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1)
    
            i_triangles = np.where(np.all(mesh.elements == mesh.omega[i], axis=1))[0][0]
            hash_i = np.where(norm_dict[Norm]((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose())<=(eps_i + mesh.diam))[0].tolist()
    
            idx = [np.where(mesh.nodes == j)[0][0] for j in T_i]
            
            # u and u_bar values on three vertices of the current triangle
            u_vals = u[idx]
            v_vals = v[idx]

            for a in range(3):
                a_nodes = np.where(mesh.nodes == T_i[a])[0][0] 

                X_val, Y_val = 0,0

                div_V_x = gradient[a,:].dot(iMat_i[:,0])
                div_V_y = gradient[a,:].dot(iMat_i[:,1])

                for j in hash_i:
                    label_j = mesh.elements[j, 0]
                    gam_ij = gam[str(label_i)+str(label_j)]
                    
                    T_j = mesh.elements[j, 1:].tolist()
                    T_j_v = mesh.vertices[T_j]
                    Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()    
                    det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1]
                    iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
                    det_T_j = abs(det_T_j)
                    def Phi_j(y):
                        return Mat_j.dot(y) + np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) 
    
                    if label_j != 3:
                        idx_j = [np.where(mesh.nodes == j)[0][0] for j in T_j]
                        
                        # u values at the vertices of T_j
                        u_vals_j = u[idx_j]
                    
                    u_vals_X_P   = PSI_X.transpose().dot(u_vals) # for double integral                    
                    v_vals_X_P   = PSI_X.transpose().dot(v_vals) # for double integral
                    u_vals_Y_P_j = PSI_Y.transpose().dot(u_vals_j) # for double integral
                    
                    # add div V(y) if \neq 0
                    if T_i[a] in T_j:
                        b = T_j.index(T_i[a])
                        div_V_x += gradient[b,:].dot(iMat_j[:,0])
                        div_V_y += gradient[b,:].dot(iMat_j[:,1])

                    ### weighted L2 product part ###
                    L2 = det_T_i * det_T_j * ( u_vals_X_P *  v_vals_X_P  * W * gam_ij(Phi_i(X),Phi_j(Y))).sum()
                    X_val += div_V_x * L2
                    Y_val += div_V_y * L2
        
                    ### convolution part ###
                    if label_j != 3:
                        convu = det_T_i * det_T_j * ( v_vals_X_P *  u_vals_Y_P_j  * W * gam_ij(Phi_i(X),Phi_j(Y))).sum()
                        X_val += - div_V_x * convu
                        Y_val += - div_V_y * convu

                dV[a_nodes] += X_val
                dV[num_nodes + a_nodes] += Y_val
         
        return 2 * dV
     
    p = Pool(num_cores)           
    dVs = p.map(aux, range(num_cores))
    
    dV = np.zeros(2*len(mesh.nodes))
    for i in range(num_cores):
        dV += dVs[i]
        
    del dVs
    p.close()
    p.join()
    p.clear()

    for k in mesh.boundary:
        dV[k] = 0
        dV[num_nodes + k] = 0

    return dV   












            
#=============================================================================#
#=============================================================================#
"""-------------------------------------------------------------------------"""
"""                   TEST ASSEMBLY                           """
"""-------------------------------------------------------------------------"""
test_ball_volume =0
if test_ball_volume:

    h, delta, Norm, num_cores = 0.025, 0.2, 'L2', 8
    mesh, mesh_data = prepare_mesh_reg(h, delta, Norm, num_cores)

    ##### GMSH FILE adaption with correct interaction horizon for outer boundary
    # import os
    # fil_target = 'circle'
    # textfile = open('' + fil_target + '.geo', 'r')
    # data = textfile.readlines()
    #
    # tmpfile = open('test.txt', 'w+')
    # tmpfile.write('delta = ' + str(delta) + ';\n')
    #
    # for line in data[1:]:
    #     tmpfile.write(line)
    #
    # tmpfile.close()
    #
    # os.system('rm ' + fil_target + '.geo')
    # current_path = os.path.dirname(os.path.abspath(__file__))
    # os.system('mv ' + current_path + '/test.txt ' + current_path + '/' + fil_target + '.geo')
    # ##### GMSH FILE adation END #####
    #
    # os.system('gmsh ' + fil_target + '.geo -2 -clscale ' + str(h) + ' -o ' + fil_target + '.msh')
    # verts, lines, triangles =  read_mesh('' + fil_target + '.msh')
    # mesh, mesh_data_target = prepare_mesh(verts, lines, triangles, delta, Norm)
    ##### GMSH FILE

    i = int(len(mesh.verts) * 1.5) // 2
    # a, b = 0, 2
    # print(mesh.triangles)
    T_i = mesh.triangles[i, 1:].tolist()
    T_i = mesh.verts[T_i]
    x_i = T_i[0]#0.5*(T_i[0]+T_i[1])
    plt.plot(x_i[0], x_i[1], 'ro')

    hash_i = np.where(norm_dict['L2']((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose()) < delta+mesh.diam)[0].tolist()
    ball_vol = 0
    for j in hash_i:
        T_j = mesh.triangles[j, 1:].tolist()
        T_j = mesh.verts[T_j]
        plt.gca().add_patch(plt.Polygon(T_j, closed=True, fill=False, color='black'))

        # tris = retriangulate_dict[ball](x_i, T_j,'', delta)
        # caps = intersection_exactl2_capsonly(x_i, T_j, delta)
        tris, caps = intersection_l2_exactcaps(x_i, T_j, delta)
        for T_i_v in tris:
            T_i_v = np.array(T_i_v)
            Mat_i = np.array([T_i_v[1] - T_i_v[0], T_i_v[2] - T_i_v[0]]).transpose()
            det_T_i = Mat_i[0, 0] * Mat_i[1, 1] - Mat_i[1, 0] * Mat_i[0, 1]
            plt.gca().add_patch(plt.Polygon(T_i_v , closed=True, fill=True, color='red', alpha =0.5))
            plt.gca().add_patch(plt.Polygon(T_i_v, closed=True, fill=False, color='black'))
            ball_vol += 0.5 * abs(det_T_i)
        if len(caps)!=0:
            for cap in caps:
                centroid = cap[0]
                plt.plot(centroid[0], centroid[1], 'bo')
                ball_vol += cap[1]
    plt.gca().add_patch(plt.Polygon(T_i, closed=True, fill=True, color='yellow', alpha = 0.3))
    print(ball_vol, np.pi*delta**2)
    plt.axis('equal')
    plt.show()

test_assembly=0
if test_assembly:
    plot = 1
    
    h, delta, Norm, num_cores = 0.0125, 0.1, 'L2', 1
    mesh, mesh_data = prepare_mesh_reg(h, delta, Norm, num_cores)
    ball = 'exact_L2'

    i, j = int(len(mesh.verts)*1.5 )/2, 71
    a, b = 0,2
      
    label_i = mesh.triangles[i, 0]
    eps_i = delta
    
    T_i = mesh.triangles[i, 1:].tolist()
    T_i_v = mesh.verts[T_i]
    if plot:
        fig, ax = plt.subplots()
        plt.gca().add_patch(plt.Polygon(T_i_v , closed=True, fill = True, color = 'yellow'))
    
    #for cc in range(3):
    #    circle1 = plt.Circle(tuple(T_i_v[cc]), delta, color='r', fill = False, linewidth = 3)
    #    ax.add_artist(circle1)
    #circle1 = plt.Circle(tuple(mesh.bary[i]), delta, color='b', fill = False, linewidth = 3)
    #ax.add_artist(circle1)
    #plt.plot(mesh.bary[i][0], mesh.bary[i][1], 'bo')
    
    Mat_i = np.array( [T_i_v[1] - T_i_v[0],T_i_v[2] - T_i_v[0] ]).transpose()
    det_T_i = Mat_i[0,0] * Mat_i[1,1] - Mat_i[1,0] * Mat_i[0,1] 
    def Phi_i(y):
        return np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1) +  Mat_i.dot(y)
    iMat_i = 1./det_T_i * np.array([ [Mat_i[1,1], -Mat_i[0,1]], [-Mat_i[1,0], Mat_i[0,0]]  ])
    def iPhi_i(y):
        return iMat_i.dot( y - np.repeat(T_i_v[0][:,np.newaxis], n**2, axis=1))
    def iPhi_i2(y):
        return iMat_i.dot( y - np.repeat(T_i_v[0][:,np.newaxis], n, axis=1))
    def iPhi_i0(y):
        return iMat_i.dot( y - T_i_v[0] )
    i_triangles = i
    hash_i = mesh.hash_table_bary[i_triangles]

    overall =0
    hash_i = np.where(norm_dict['L2']((mesh.bary-np.repeat(mesh.bary[i][:,np.newaxis], len(mesh.bary), axis = 1).transpose()).transpose()) < eps_i)[0].tolist()     

    for j in hash_i:
        
        
        
        label_j = mesh.triangles[j, 0]
        T_j = mesh.triangles[j, 1:].tolist()
        T_j_v = mesh.verts[T_j]
        

        
        if plot:
            plt.gca().add_patch(plt.Polygon(T_j_v , closed=True, fill = True, color = 'orange', alpha = 0.25))
#            ax.annotate(str(j),mesh.bary[j], size = 7)
        
        """plot barycenter of inner triangle"""
#        plt.plot(mesh.bary[j][0], mesh.bary[j][1], 'go')
#        circle1 = plt.Circle(tuple(mesh.bary[j]), delta, color='g', fill = False, linewidth = 3)
#        ax.add_artist(circle1)

        Mat_j = np.array( [T_j_v[1] - T_j_v[0],T_j_v[2] - T_j_v[0] ]).transpose()        
        det_T_j = Mat_j[0,0] * Mat_j[1,1] - Mat_j[1,0] * Mat_j[0,1] 
                            
#        if np.linalg.norm(mesh.bary[i_triangles]-mesh.bary[j]) < eps_i - mesh.diam:
#            # no subdivision or outer integral treatment needed
#            def Phi_j(y):
#                return np.repeat(T_j_v[0][:,np.newaxis], n**2, axis=1) +  Mat_j.dot(y)
#                
#            if plot:
#                plt.gca().add_patch(plt.Polygon(T_j_v , closed=True, fill = True, alpha = 0.7))
#                plt.gca().add_patch(plt.Polygon(T_j_v , closed=True, fill = False)) 
#            
#            overall += det_T_j  
#
#
#        else:
#            tris = retriangulate_dict[ball](mesh.bary[j], T_i_v, norm_dict[Norm], eps_i )
#            
#            for tri in tris:
#                tri = np.array(tri)
#                
#                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True, alpha = 0.3))
#                plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False))                 
#                
#                Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
#                det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0])
#                def Phi_l(y):
#                    return np.repeat(tri[0][:,np.newaxis], n, axis=1) +  Mat_l.dot(y)
#                def Phi_l0(y):
#                    return tri[0] +  Mat_l.dot(y)

#                print iPhi_i2(Phi_l(P))
#                print 
#                print P
#                plt.gca().add_patch(plt.Polygon(T_ref , closed=True, fill = True, alpha = 0.3))
#                plt.gca().add_patch(plt.Polygon(T_ref , closed=True, fill = False))   
#                plt.gca().add_patch(plt.Polygon([iPhi_i0(tri[i]) for i in range(3)] , closed=True, fill = True, alpha = 0.3))
#                plt.gca().add_patch(plt.Polygon([iPhi_i0(tri[i]) for i in range(3)] , closed=True, fill = False))                   
#                for i in range(n):
#                    x = iPhi_i2(Phi_l(P))[:,i]
#                    plt.plot(x[0], x[1], 'ro')
#                           if label_j != labels_domains[-1]:
#                       L[kk, T_j[b]] += -abs(det_l) * abs(det_T_j) * ( basis[a](iPhi_i(Phi_l(X))) *  PSI_Y[b]  * W * gam_j(Phi_l(X),Phi_j(Y))).sum()
#                       L[kk, T_i[b]] += abs(det_l) * abs(det_T_j) * ( basis[a](iPhi_i(Phi_l(X))) *  basis[b](iPhi_i(Phi_l(X)))  * W * gam_j(Phi_l(X),Phi_j(Y))).sum()#det_T_i * (PSI_P[a] * PSI_P[b] * I[3] * weights).sum()


#            iMat_j = 1./det_T_j * np.array([ [Mat_j[1,1], -Mat_j[0,1]], [-Mat_j[1,0], Mat_j[0,0]]  ])
#            def iPhi_j(y):
#                return iMat_j.dot( y - np.repeat(T_j_v[0][:,np.newaxis], n2, axis=1))
#            #--------------------------------------------------------------------------
#        
#            def I1(x):
#                x_trans = (T_i_v[0]+Mat_i.dot(x))
#                integral = 0
#                aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)
#                aux2 = np.repeat(x[:,np.newaxis], n2, axis=1)
#                
#                if plot:
#                    plt.plot(x_trans[0], x_trans[1], 'ro')
#                    circle1 = plt.Circle(tuple(x_trans), delta, color='b', fill = False, linewidth = 1)
#                    ax.add_artist(circle1)
#    
#                def inner(tri, gam_j):
#                    tri = np.array(tri)
#                    Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
#                    det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
#                    def Phi_l(y):
#                        return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)
#                        
#                    
#                    
#                    return det_l #* (  weights2).sum() #det_l * ((basis[b](aux2) -  basis[b](iPhi_j(Phi_l(P2))) )* weights2).sum() 
#    
#                tris = retriangulate_dict[ball](x_trans, T_j_v, norm_dict[Norm], eps_i )
#                if len(tris) != 0:
#                    for tri in tris:     
#                        
#                        integral += inner(tri,1)
#                        """plot for testing below"""
#                        if plot:
#                            plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True, alpha = 0.3))
#                            plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False)) 
#            
#                return integral
#        
#        
#            I = np.array(map(I1, P.transpose())).transpose()
#
#            overall +=  I[0]
#    
#    print np.abs(0.5*overall-np.pi*delta**2)


#            print 'standard', det_T_i * (PSI_P[a] * I * weights).sum()
            
    #--------------------------------------------------------------------------
    #        def I1(x):
    #            x_trans = (T_i_v[0]+Mat_i.dot(x))
    #            integral  = 0.
    #            aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)
    #            aux2 = np.repeat(x[:,np.newaxis], n2, axis=1)
    ##            plt.plot(x_trans[0], x_trans[1], 'rx')
    #                    
    #            def inner(tri, gam_j):
    #                tri = np.array(tri)
    #                Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
    #                det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
    #                def Phi_l(y):
    #                    return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)
    #    
    #                return det_l * (basis[a](aux2) * (basis[b](aux2) -  basis[b](iPhi_j(Phi_l(P2))) )* weights2).sum() 
    #    
    #    
    #            tris = retriangulate_dict[ball](x_trans, T_j_v, norm_dict[Norm], eps_i )
    #    
    #            
    #            if len(tris) != 0:
    #                for tri in tris:     
    #                    integral += inner(tri, 1)
    #                    
    #    
    #                    """plot for testing below"""
    ##                    plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
    ##                    plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False)) 
    #    
    #    
    #    
    #            return integral
    #    
    #        I = map(I1, P.transpose())
    #        
    #        val = det_T_i * tri_adapt(I1, T_ref, tol2_Radon_get =0.01, plot = 1)
    #        
    #        print np.abs(val- det_T_i *(I*weights).sum())/np.abs(val)
    #        print 'adapt', val
    #        print 'nonadapt', det_T_i *(I*weights).sum()
            
    #        def I1(x):
    #            x_trans = (T_i_v[0]+Mat_i.dot(x))
    #            integral, integral0, integral1, integral2 = 0., 0., 0., 0.
    #            aux = np.repeat(x_trans[:,np.newaxis], n2, axis=1)
    #                    
    #            def inner(tri, gam_j):
    #                tri = np.array(tri)
    #                Mat_l = np.array( [tri[1] - tri[0],tri[2] - tri[0] ]).transpose()
    #                det_l = abs((tri[1] - tri[0])[0] * (tri[2] - tri[0])[1] - (tri[1] - tri[0])[1] * (tri[2] - tri[0])[0] )
    #                def Phi_l(y):
    #                    return np.repeat(tri[0][:,np.newaxis], n2, axis=1) +  Mat_l.dot(y)
    #                    
    #                GAM = det_l   * weights2   
    #
    #                return  GAM.sum(), (basis[0](iPhi_j(Phi_l(P2))) * GAM ).sum(), (basis[1](iPhi_j(Phi_l(P2))) * GAM ).sum()  , (basis[2](iPhi_j(Phi_l(P2))) * GAM ).sum()  
    #
    #
    #            tris = retriangulate_dict[ball](x_trans, T_j_v, norm_dict[Norm], eps_i )
    #
    #            if len(tris) != 0:
    #                for tri in tris:     
    #                    v, v0, v1, v2 = inner(tri, 1)
    #                    integral  += v
    #                    integral0 += v0
    #                    integral1 += v1
    #                    integral2 += v2
    #                    
    #                    """plot for testing below"""
    ##                    plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = True))
    ##                    plt.gca().add_patch(plt.Polygon(tri , closed=True, fill = False)) 
    #        
    #            return np.array([integral0, integral1, integral2, integral])
    #
    #        I = np.array(map(I1, P.transpose())).transpose()
    #
    #        print 'standard old', det_T_i * (PSI_P[a] * ( PSI_P[b]* I[3] - I[b] )* weights).sum()
    #
    #-------------------------------
    if plot:
                
        for ii in mesh.triangles[:,1:]:
            plt.gca().add_patch(plt.Polygon(mesh.verts[ii], closed=True, fill = False , alpha  = 0.25)) 
    #        for ii in mesh.omega[:,1:]:
    #            plt.gca().add_patch(plt.Polygon(mesh.verts[ii], closed=True, fill = True , color = 'gray',alpha  = 0.05)) 
        ax.axis('equal')     
           
        #
        #plt.figure('I')
        #plt.gca().add_patch(plt.Polygon(T_ref , closed=True, fill = False))  
        #for p in P.transpose():
        #    plt.plot(p[0], p[1], 'rx')
        #plt.tricontourf(P[0,:],P[1,:],I,100,interpolation='nearest',cmap =plt.cm.get_cmap('rainbow')) # choose 20 contour levels, just to show how good its interpolation is
        #plt.colorbar()
        #plt.axis('equal')
           
#    plt.gca().add_patch(plt.Polygon(T_i_v , closed=True, fill = True, color = 'yellow'))
    
