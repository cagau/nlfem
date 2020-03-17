# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:37:19 2019

@author: vollmann
"""
import numpy as np
import examples.Rates2D.bib3 as bib
import scipy.optimize as optimization

def u_exact(x):
    return x[0]**2 * x[1] + x[1]**2

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

def main(num_fem_sols):
    print("\n__________________________________________________________\nL2 RATES\n")
    folder = 'results/overlap/'
    ball  = 'johnCaps'
    k_finest = 7
    H = [.1* 2**-k for k in range(0,num_fem_sols)]
    h_finest = 0.1 * 2**-(k_finest)

    U = []
    for h in H:
        params = '_L2_h' + str(h) + '_delta0.1_ball' + ball + '_'
        u = np.load(folder+ 'u'+params+'.npy')

        if h==0.1:
            u = np.concatenate((u[9:], u[0:9]))

        L = int(np.sqrt(len(u)))
        n = L+2#int(1./h)+1
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        xv, yv = np.meshgrid(x, y)
        U_aux = np.array([u_exact(np.array([yv[i,j], xv[i,j]])) for i in range(n) for j in range(n)]).reshape(n,n)
        U_aux[1:-1, 1:-1] = u.reshape((L,L))
        U_aux = U_aux.reshape(n**2)
        U += [U_aux]

    #-----------------------------------------------------------------------------#

    """load """
    # mesh    = bib.prepare_mesh_reg_slim(h, 0.1, 'L2', 8)[0]#
    # mesh = bib.Mesh_slim(np.load('results/MASS_MATRIX_REG/mesh_data_slim_regular_h_'+str(h_finest)+'_delta_0.1.npy', allow_pickle = True))
    massmat = bib.load_sparse_csr('results/MASS_MATRIX_REG/M_omom' + str(h_finest) + '.npz')#bib.mass_matrix2(mesh)#load_sparse_csr(folder+'M_full_'+str(h)+'.npz')

    # Compoute on the fly
    print( '\nCompute exact solution...')
    n = int(1./h_finest)+1
    x = np.linspace(0,1,n)[1:-1]
    XY= np.meshgrid(x,x)
    solution = u_exact(XY).reshape((n-2)**2, order ='F')


    # Interpolate all to finest mesh
    print( 'Interpolate...')
    for i in range(0, len(U)):

        u_aux = U[i]

        for j in range(i, k_finest):
            u_aux = interpolate(u_aux)

        L = int(np.sqrt(len(u_aux)))
        u_aux = u_aux.reshape((L,L))[1:-1,1:-1]
        u_aux = u_aux.reshape((L-2)**2)
        U[i] = u_aux

    #-----------------------------------------------------------------------------#
    def dist_L2(u,v):
        return np.sqrt( np.dot(u-v, massmat.dot(u-v) ) )
    error_l2 = np.zeros(len(U))
    for i in range(len(U)):
        error_l2[i] = dist_L2(U[i],solution)

    ##==============================================================================
    #""" L2"""
    ##==============================================================================
    print()
    print( 'L2 ERROR')
    for a in error_l2:
        print( a)

    print( 'RATES')
    rates = []
    for i in range(len(error_l2)-1):
        rates.append(np.log(error_l2[i]/error_l2[i+1])/np.log(2))
        print(rates[-1])

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

    print()
    return {"h": H, "L2 Error": error_l2, "Rates": [0] + rates}

if __name__ == "__main__":
    main(2)