
"""
Created on Mon Feb 25 13:31:10 2019

@author: vollmann
"""

import sys
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/DATA/Dropbox/JOB/PYTHON/BIB")
import warnings
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
import numpy as np
import bib3 as bib
from time import time
from time import strftime
import os
np.set_printoptions(linewidth = 200, precision = 4)

#==============================================================================
#                                   INPUT
#==============================================================================
#INPUT
num_cores = 8

hash_onthefly = 1

approx = 0

geo_file = 'target_shape'#'circle_large'

#--------------------------------------#
#   PARAMS TO COPY INTO SOLVE FILE     #
#--------------------------------------#
#start_copy
Delta = [0.1]
Ball= ['approx_L2']
Norm = 'L2'  # choose from ['L1', 'L2', 'Linf']
H = [0.1 * 2**-k for k in range(0,3)]
new_mesh = 1
overwrite = 0# if True: no extra folder is generated and existing data overwritten
# folder_string = 'bary_outerexact'
suffix_string = ''
reg_mesh = 1 # create regular grid on [0,1]^2
#end_copy
#--------------------------------------#

norm = bib.norm_dict[Norm]


    
#END_INPUT
#==============================================================================
#                              SAVE
#==============================================================================
if overwrite:
    folder = 'results/'
else:
    timestr = strftime('%d-%m-%Y_%H%M%S')


    folder = 'results/TESTAREA/'#+folder_string+'/'#+timestr


    os.system('mkdir '+folder)

mesh_str = 'geo'
if reg_mesh:
    mesh_str = 'regular_'
    
#------------------------------------------------------------------------------   
print('\nNorm  = ', Norm, '\n\n')
for h in H:
    
    for delta in Delta:
        
        # Kernel
        def phi(r):
           return 1#(1 - ((r/delta )**2) )
        g11 = 0.001 * 3. / (4 * delta ** 4)
        g22 = 100 * 3. / (4 * delta ** 4)
        loclim = 4. / (np.pi * delta ** 4)

        def gam11(x, y):
            s = norm(x - y)
            return np.where(s >= delta, 0, loclim)#loclim #np.where(s > delta, 0, )#g11 #* (1 - ((s / delta) ** 2)) # np.where(s > eps2, 0, g11)

        def gam22(x, y):
            s = norm(x - y)
            return np.where(s >= delta, 0, loclim)#np.where(s > delta, 0,   ) #* (1 - ((s/delta )**2) )

        def gam13(x, y):
            s = norm(x - y)
            return np.where(s >= delta, 0, loclim)#
            
#        def gam1(x,y):
#            s = norm(x-y)
#            return 4. / (np.pi * delta **4) * phi(s) #np.where(s > delta, 0 , phi(s)) #
            
        gam = {'11': gam11, '22': gam22, '12': gam11, '21': gam22, '13': gam13, '32': gam22, '23': gam22, '33': gam22, 'eps1': delta, 'eps2': delta, 'eps3':delta}

        for ball in Ball:
    
            output_mesh_data = folder + 'mesh_data_' + mesh_str + 'h_' + str(h) + '_delta_' + str(delta)
            
            
            #------------------------------------------------------------------
            #                            PRINT                                #  
            #------------------------------------------------------------------
            print( '-------------------------------------------------')
            print( 'h     = ', h,  '\ndelta = ', delta, '\nball = ', ball)
        
            if overwrite:
                output_A = ''
            else:
                output_A = '_' + Norm + '_h' + str(h) + '_delta' + str(delta)[0:5] + '_' + 'ball' + ball + '_' +  str(suffix_string)
            

            """---------------------#
            #      LET'S GO         #
            #---------------------"""
            #==============================================================================
            #                 MESH
            #============================================================================== 
            if new_mesh: 
                #generate new mesh
                if reg_mesh:
                    # generate regular grid on [0,1]^2
                    mesh, mesh_data = bib.prepare_mesh_reg(h, delta, Norm, num_cores)

                else:
                    ##### GMSH FILE adaption with correct interaction horizon for outer boundary

                    textfile = open('mesh/'+geo_file+'.geo', 'r')
                    data = textfile.readlines()
                    
                    tmpfile = open('mesh/test.txt', 'w+')
                    tmpfile.write('delta = ' + str(delta) + ';\n')   
                    
                    for line in data[1:]:
                        tmpfile.write(line)
                    
                    tmpfile.close()
                    
                    os.system('rm mesh/'+geo_file+'.geo')
                    current_path = os.path.dirname(os.path.abspath(__file__))
                    os.system('mv '+current_path+'/mesh/test.txt ' + current_path +'/mesh/'+geo_file+'.geo')
                    ##### GMSH FILE adation END #####

                    # generate mesh based on geo-file 
                    os.system('gmsh mesh/'+geo_file+'.geo -2 -clscale ' + str(h) + ' -o mesh/' + geo_file + '.msh')

                    verts, lines, triangles = bib.read_mesh('mesh/'+geo_file+'.msh')
                    mesh, mesh_data = bib.prepare_mesh(verts, lines, triangles, delta, Norm)
                # save mesh
                np.save(output_mesh_data, mesh_data)
                print( '\n mesh prepared and carefully saved \n')
                del mesh_data
            else:
                mesh = bib.Mesh(np.load(output_mesh_data+'.npy'))
            print( '\n Len(nodes): ', len(mesh.nodes))
            print( ' Len(verts): ', len(mesh.verts), '\n')
            
            
            #==============================================================================
            #              ASSEMBLY AND SAVE
            #==============================================================================  
            t1 = time()
            if approx:
                A = bib.assembly_coupling_approx_full(mesh, gam, Norm,num_cores)#bib.assembly_coupling_full_approx(mesh, gam, bib.retriangulate_dict[ball], norm, num_cores, hash_onthefly = hash_onthefly)
            else:

                if ball == 'exactcaps':
                    A = bib.assembly_coupling_full_exactcaps(mesh, gam, bib.retriangulate_dict[ball], norm, num_cores,
                                                             hash_onthefly=hash_onthefly)
                else:

                    A = bib.assembly_coupling_full_standard(mesh, gam, bib.retriangulate_dict[ball], norm, num_cores, hash_onthefly = hash_onthefly)
                    # A = bib.assembly_coupling_full_bary(mesh, gam, bib.retriangulate_dict[ball], norm, num_cores, hash_onthefly = hash_onthefly)
                    # A = bib.assembly_coupling_full_shifted(mesh, gam, bib.retriangulate_dict[ball], norm, num_cores, hash_onthefly = hash_onthefly)
#                print A.A == B.A
#                print np.allclose(A.A, B.A)
            print( '\n time for A:', time()-t1 ,'   ('+ball+')\n')
            print( '-------------------------------------------------')
            
            bib.save_sparse_csr(folder+'A'+output_A, A.tocsr())
#            np.save(folder+'A'+output_A, A)
            del A


    # assembly mass matrix
    M = bib.mass_matrix2(mesh)
    bib.save_sparse_csr(folder+'M_'+str(h), M.tocsr())
    M = bib.mass_matrix_full(mesh)
    bib.save_sparse_csr(folder+'M_full_'+str(h), M.tocsr())
    del M    
#==============================================================================
#              PRINT INPUT - PARAMETERS INTO TEXT-FILE
#==============================================================================  
current_script = open('DEPRICATED_ASSEMBLY.py', 'r')
textfile = open(folder+'parameters.txt', 'w')

for line in current_script:
    
    if line.find('#INPUT') == 0:
        line = current_script.readline()
        while True:
            if line.find('#END_INPUT')== -1:
                textfile.write(line)   
                line = current_script.readline()
            else:
                break

current_script.close() 
textfile.close()

# if not overwrite:
#     timestr_txt = open('results/timestr.txt','a')
#     timestr_txt.write('\n' +folder_string+timestr)
#     timestr_txt.close()

