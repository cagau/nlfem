
"""
Created on Mon Feb 25 13:31:10 2019

@author: vollmann
"""
import assemble
import numpy as np
import python.bib3 as bib
import pickle as pkl
import scipy.sparse as sp
import python.nlocal as nlocal
import python.conf as conf

#==============================================================================
#                                   INPUT
#==============================================================================
def main(num_fem_sols):
    print("\n__________________________________________________________\nASSEMBLE\n")
    H = [.1* 2**-k for k in range(0,num_fem_sols)]

    delta = conf.delta
    ball= 'johnCaps'
    Norm = 'L2'
    suffix_string = ''
    norm = "L2"
    folder = 'results/overlap/'
    mesh_str = 'regular_'

    for h in H:
        output_mesh_data = folder + 'mesh_data_' + mesh_str + 'h_' + str(h) + '_delta_' + str(delta)
        #------------------------------------------------------------------
        #                            PRINT                                #
        #------------------------------------------------------------------
        print( '-------------------------------------------------')
        print( 'h     = ', h,  '\ndelta = ', delta)
        output_A = '_' + Norm + '_h' + str(h) + '_delta' + str(delta)[0:5] + '_' + 'ball' + ball + '_' +  str(suffix_string)

        #==============================================================================
        #                 MESH
        #==============================================================================
        # generate regular grid on [0,1]^2
        mesh, mesh_data = bib.prepare_mesh_reg(h, delta, Norm, num_cores=8)
        np.save(output_mesh_data, mesh_data)
        print( '\nLen(nodes): ', len(mesh.nodes))
        print( 'Len(verts): ', len(mesh.verts), '\n')
        #==============================================================================
        #              ASSEMBLY AND SAVE
        #==============================================================================

        A, f = assemble.assemble(nlocal.Mesh(mesh, conf.ansatz, conf.boundaryConditionType), conf.py_Px, conf.py_Py, conf.dx, conf.dy, conf.delta)
        A = sp.csr_matrix(A)
        bib.save_sparse_csr(folder+'A'+output_A, A)
        #del A

        M = bib.mass_matrix2(mesh)
        bib.save_sparse_csr(folder+'M_'+str(h), M.tocsr())
        M = bib.mass_matrix_full(mesh)
        bib.save_sparse_csr(folder+'M_full_'+str(h), M.tocsr())
        #del M

if __name__ == "__main__":
    main(2)