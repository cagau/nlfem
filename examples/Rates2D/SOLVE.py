"""
Created on Mon Feb 25 14:59:05 2019

@author: vollmann
"""
import python.bib3 as bib
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as ss
from pathos.multiprocessing import ProcessingPool as Pool
import python.conf as conf

def source(x):
    return -2. * (x[1] + 1)

def g_d(x):
    return x[0]**2 * x[1] + x[1]**2

def main(num_fem_sols):
    #==============================================================================
    #                                   INPUT
    #==============================================================================
    print("\n__________________________________________________________\nSOLVE\n")
    H = [.1* 2**-k for k in range(0,num_fem_sols)]

    num_cores = 8
    timestr = 'overlap'
    delta = conf.delta
    ball= 'johnCaps'
    Norm = 'L2'
    new_mesh = 1
    suffix_string = ''
    plot = 0
    folder = 'results/'+timestr+'/'
    mesh_str = 'regular_'
    #==============================================================================
    #                 GO
    #==============================================================================
    for h in H:
        print( '\n-------------------------------------------------')
        M = bib.load_sparse_csr(folder + 'M_' + str(h) + '.npz')

        mesh = bib.Mesh(np.load(folder + 'mesh_data_' + mesh_str + 'h_' + str(h) + '_delta_' + str(delta)+'.npy', allow_pickle=True))

        print('start rhs')
        p = Pool(num_cores)
        aux = np.array(p.map(source, mesh.verts[mesh.nodes] ))
        b = M.dot(aux)
        print( 'rhs computed')

        params = '_' + Norm + '_h' + str(h) + '_delta' + str(delta)[0:5] + '_' + 'ball' + ball + '_' +  str(suffix_string)

        A = bib.load_sparse_csr(folder+'A'+params+'.npz')
        A = A.tolil()
        # DIRICHTLET
        num_nodes = len(mesh.nodes)

        M = bib.load_sparse_csr(folder + 'M_full_' + str(h) + '.npz')
        nodes_inner = range(len(mesh.nodes)-len(mesh.boundary))
        nodes_rest = range(len(nodes_inner), len(mesh.verts))

        A_omom = A[nodes_inner[0]:nodes_inner[-1]+1, nodes_inner[0]:nodes_inner[-1]+1]
        M_omom = M[nodes_inner[0]:nodes_inner[-1]+1, nodes_inner[0]:nodes_inner[-1]+1]
        A_omomi = A[nodes_inner[0]:nodes_inner[-1]+1, nodes_inner[-1]+1:]

        #b_john = np.load("f_h"+str(h)+".npy")
        #b[nodes_inner] = b_john
        #print("Replace right side by John's right side")
        # Klar, 6.2.2019_2020
        #plt.plot(b, c= "b")
        #plt.plot(b_john, c="r")
        #plt.legend()
        #plt.show()
        print("Nodes Inner ", b[nodes_inner].shape, "\nShpae A_Omega ",A_omomi.shape,"\nNumber of Verts", mesh.verts[nodes_rest].shape)

        b_aux = b[nodes_inner] - A_omomi.dot(np.array(list(map(g_d,mesh.verts[nodes_rest]))))

        # SOLVE AND SAVE
        A_omom = A_omom.tocsr()
        A_omom = ss.csr_matrix(A_omom)
        bib.save_sparse_csr(folder+'A_omom'+params, A_omom.tocsr())
        bib.save_sparse_csr(folder+'M_omom'+params, M_omom.tocsr())

        print( 'start solving')
        u = np.linalg.solve(A_omom.todense(), b_aux)
        np.save(folder+'u'+params, u)#np.concatenate((u, map(g_d, mesh.verts[nodes_rest]))))
        del A


        #-------------------------------------------------------------#
        #                       PLOT                                  #
        #-------------------------------------------------------------#
        if plot:
            exactSol = np.array(list(map(g_d,mesh.verts[nodes_inner[0]:nodes_inner[-1]+1])))
            bib.plot_all(mesh, np.concatenate((u, list(map(g_d, mesh.verts[nodes_rest])))), title = 'u'+params+ '   APPROX')
            bib.plot_all(mesh, np.concatenate( (exactSol - u, np.zeros(len(nodes_rest)))), title = 'u'+params+ '   APPROX')

            labels_domain = np.sort(np.unique(mesh.triangles[:,0])).tolist()
            for label in labels_domain:
               omega = mesh.triangles[np.where(mesh.triangles[:,0] == label)[0]]
               #plot_tri(omega[:,1:], closed = True, fill = False, color = 'black')
            plt.axis('equal')
            plt.show()

        #####
        u_exact = np.array(list(map(g_d, mesh.verts[nodes_inner])))
        print ("L2 Error ", np.sqrt(np.dot(u-u_exact, M_omom.dot(u-u_exact))))

if __name__ == "__main__":
    main(2)