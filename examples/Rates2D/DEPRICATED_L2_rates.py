# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:19 2019

@author: vollmann
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("/home/vollmann/Dropbox/JOB/PYTHON/BIB")
sys.path.append("/media/vollmann/DATA/Dropbox/JOB/PYTHON/BIB")

import bib3 as bib
#from pathos.multiprocessing import ProcessingPool as Pool
import scipy.optimize as optimization
from scipy.sparse import diags
import conf2
import xlwt

np.set_printoptions(linewidth = 300)

folder = 'results/TEST_john/'#bary_7points_08-10-2019_151429/'#adaptive_barycenter_2_18-06-2019_151612/'#adaptive_approx2_14-06-2019_161243/' #adaptive_barycenter_2_18-06-2019_151612
ball  = 'exact_L2'
#num_cores = 8

num_fem_sols = 3
k_finest = num_fem_sols - 1



def u_exact(x):
    return x[0]**2 * x[1] + x[1]**2

H = [0.1 * 2**-k for k in range(0,num_fem_sols)]
h_finest = 0.1 * 2**-(k_finest)


def interpolate(u):
    """
    interpolates on regular grid on the square [0,1]^2
    arrays must contain values on the boundary    
    """
    L = int(np.sqrt(len(u)))    
    u = u.reshape((L,L))#np.reshape(u, (L,L), order = 'F')[::-1]# 
    L_new = 2*L - 1

    U = np.zeros((L_new, L_new))

    for i in range(L-1):
        for j in range(L-1):
            
            U[2*i, 2*j]    = u[i,j]
            U[2*i, 2*j +1] = 0.5  * (u[i,j] + u[i,j+1])
            U[2*i+1,2*j]   = 0.5  * (u[i,j] + u[i+1,j])
            U[2*i+1,2*j+1] = 0.25 * (u[i,j] + u[i,j+1] + u[i+1,j]+ u[i+1,j+1])

    for j in range(L-1):
        U[-1, 2*j]    = u[-1,j]
        U[-1, 2*j +1] = 0.5  * (u[-1,j] + u[-1,j+1])

    for i in range(L-1):
        U[2*i, -1]    = u[i,-1]
        U[2*i+1,-1]   = 0.5  * (u[i,-1] + u[i+1,-1])

    U[-1,-1] = u[-1,-1]

    return U.reshape((L_new)**2) 
#-----------------------------------------------------------------------------#

# Load the solutions + extend them by the boundary data
U = []
for h in H:
    k = H.index(h)
    params = '_L2_h' + str(h) + '_delta0.1_ball' + ball + '_'
    u = np.load(conf2.folder + 'u_' + str(k) + '.npy')
    
    if h == 0.1:
        u = np.concatenate((u[9:], u[0:9]))
        
    L = int(np.sqrt(len(u)))  
    n = L+2#int(1./h)+1
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    xv, yv = np.meshgrid(x, y)
    U_aux = np.array([u_exact(np.array([yv[i,j], xv[i,j]])) for i in range(n) for j in range(n)]).reshape(n,n)
    # plt.figure(str(1+h))
    # plt.imshow(u.reshape((L,L)))
    # plt.figure(str(2+h))
    # plt.imshow(U_aux)
    U_aux[1:-1, 1:-1] = u.reshape((L,L))
    plt.figure(str(3+h))
    plt.imshow(U_aux.reshape((n,n)))
    
    U_aux = U_aux.reshape(n**2)
    
    U += [U_aux]
    
#-----------------------------------------------------------------------------#

"""load """
# mesh    = bib.prepare_mesh_reg_slim(h, 0.1, 'L2', 8)[0]#
# mesh = bib.Mesh_slim(np.load('results/MASS_MATRIX_REG/mesh_data_slim_regular_h_'+str(h_finest)+'_delta_0.1.npy', allow_pickle = True))
massmat = bib.load_sparse_csr(conf2.folder +'M_omom_' + str(k_finest) + '.npz')

#plt.imshow(massmat.A, interpolation = 'nearest')

#bib.save_sparse_csr(folder+'M'+params,massmat)

# nodes_inner = list(range(len(mesh.nodes)-len(mesh.boundary)))
# nodes_rest = list(range(len(nodes_inner), len(mesh.verts)))

#p = Pool(8)
#solution = np.array(p.map(u_exact, mesh.verts[nodes_inner]))
#print '\nexact solution computed\n'

""" compoute on the fly"""
n = int(1./h_finest)+1
x = np.linspace(0,1,n)[1:-1]
XY= np.meshgrid(x,x)
XY = conf2.transform(XY)
solution = u_exact(XY).reshape((n-2)**2, order ='F')
print( '\nexact solution computed\n')
#
#massmat  = h_finest**2 * diags([1./12., 1./12.,1./12., 0.5, 1./12., 1./12., 1./12.], [-(n-2), -(n-2)-1,-1, 0, 1, (n-2), (n-2)+1], shape=((n-2)**2, (n-2)**2)) 

#-----------------------------------------------------------------------------#
#plt.figure()
#plt.imshow(massmat2.A, interpolation = 'nearest')

#print np.allclose(sol2, solution)
#print np.allclose(massmat.A, massmat2.A)
#plt.figure()
#plt.imshow(massmat.A- massmat2.A, interpolation = 'nearest')
#plt.colorbar()


#-----------------------------------------------------------------------------#
# Interpolate all to finest mesh
for i in range(0, len(U)):

    u_aux = U[i]

    for j in range(i, k_finest):
        u_aux = interpolate(u_aux)

    L = int(np.sqrt(len(u_aux)))    
    u_aux = u_aux.reshape((L,L))[1:-1,1:-1]
    u_aux = u_aux.reshape((L-2)**2)
    U[i] = u_aux
    
#    L = int(np.sqrt(len(u_aux)))    
#    plt.figure(str(2+H[i]))
#    plt.imshow(u_aux.reshape((L,L)))
    
#    bib.plot_inner(mesh, u_aux, title = str(i))


#-----------------------------------------------------------------------------#
def dist_L2(u,v):
    return np.sqrt( np.dot(u-v, massmat.dot(u-v) ) )
    
error_l2 = np.zeros(len(U))

for i in range(len(U)):
    # L = int(np.sqrt(len(solution)))
    # plt.imshow((U[i]-solution).reshape((L, L)))
    # plt.show()
   # bib.plot_inner(mesh, np.abs(u_aux-solution), title = str(i))
   # bib.plot_inner(mesh, np.abs(u_aux), title = str(i))
    error_l2[i] = dist_L2(U[i],solution)



  

##==============================================================================
#""" L2"""
##==============================================================================

print()
print()
print( 'L2 ERROR')
for a in error_l2:
    print( a)
 
print( 'RATES')
rates_l2 = [0]
for i in range(len(error_l2)-1):
    rates_l2 += [np.log(error_l2[i]/error_l2[i+1])/np.log(2)]
    print( np.log(error_l2[i]/error_l2[i+1])/np.log(2))
rates_l2 = np.array(rates_l2)


leftout = 0
xdata = np.log(np.array([2**i for i in range(len(U)-leftout)]) )#np.array([0.6 * 2**-(i+1) for i in range(9)])# np.log(np.array([1./(0.2 * 2**-(i)) for i in range(9)]))
ydata = np.log(error_l2)# L2#
x0   = np.zeros(2)
sigma = np.ones(len(U) -leftout)

def func(x, a, b):
    return a + b*x

a, b = tuple(optimization.curve_fit(func, xdata, ydata, x0, sigma)[0])
print ()
print( 'GRADIENT OF LINEAR FIT: ', b)

## PLOT ###
def fit2(x):
    return a + b*x 

#plt.figure()
### plot log of error
#plt.plot(xdata, ydata, 'rx' )
### plot linear fit
#plt.plot(xdata, fit2(xdata))
#plt.title('solution: l2 error')
#plt.xlabel('log(h)')
#plt.ylabel('log(error)')

print()
print()


book = xlwt.Workbook(encoding="utf-8")
sheet1 = book.add_sheet("Sheet 1")
header = ["h1", "h2", "L2-Error", "Rate", "Assembly time [s]"]#, "time-Rate"]
output_ass_time = conf2.folder     + 'ass_time' + '.npy'
ass_time = np.array(np.load(output_ass_time))
data = np.concatenate((np.array(conf2.H1)[:,np.newaxis], np.array(conf2.H2)[:,np.newaxis],np.around( error_l2[:,np.newaxis], decimals=7), np.around( rates_l2[:,np.newaxis], decimals=2) , np.around( ass_time[:,np.newaxis], decimals=2) ),  axis=1)
print("data           ", data)
for i in range(len(header)): # columns
    sheet1.write(1, 1+i, header[i])
    for j in range(len(error_l2)): # rows
        sheet1.write(1 + j + 1, 1 + i, data[j,i])
book.save(folder+"rates.ods")
