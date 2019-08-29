#-*- coding:utf-8 -*-
#cython: language_level=3
#cython: boundscheck=False, cdivision=True
# Setting this compiler directive will given a minor increase of speed.

cimport cython
cimport numpy as np
from cython.parallel import prange
from libcpp.queue cimport queue
import numpy as np
from numpy.linalg import LinAlgError
from libc.math cimport sqrt, pow, pi
from conf import py_P, weights, delta, outerIntMethod, innerIntMethod
import time
from libc.stdlib cimport malloc, free, abort#, new, delete

ctypedef np.int32_t npint32_t

def assemble(mesh):
    """**Assembly routine**

    :param mesh: clsFEM. Mesh containing the data. All other data is read from *conf.py*.
    :return: list of nd.array. Returns discretized matrix A and right side f.
    """

    ## Data Matrix ----------------------------------------
    # Allocate Matrix A and right side f
    py_Ad = np.zeros((mesh.K_Omega, mesh.K))
    py_fd = np.zeros(mesh.K_Omega)

    ## BFS ------------------------------------------------
    # Allocate Graph of Neighbours
    py_Neighbours = np.ones((mesh.J, 4), dtype=np.int32)*mesh.J
    # Further definitions
    py_visited = np.zeros(mesh.J, dtype=np.int32)
    py_Mis_interact = np.zeros(3, dtype=np.int32)

    # Define Basis-----------------------------------------
    psi0 = (1 - py_P[:, 0] - py_P[:, 1])
    psi1 = py_P[:, 0]
    psi2 = py_P[:, 1]

    cdef:
        int J = mesh.J, J_Omega = mesh.J_Omega, L = mesh.L, L_Omega = mesh.L_Omega, a=0, b=0, aAdxj =0
        queue[int] c_queue
        int nP = py_P.shape[0]
        int aTdx=0, bTdx=0, sTdx=0, i=0, k=0, j=0, h=0

        np.ndarray[int, ndim=1, mode="c"] visited = py_visited
        np.ndarray[npint32_t, ndim=1, mode="c"] Mis_interact = py_Mis_interact
        np.ndarray[npint32_t, ndim=2, mode="c"] Neighbours = py_Neighbours
        np.ndarray[npint32_t, ndim=2, mode="c"] c_Triangles = np.ascontiguousarray(mesh.T, np.int32)

        npint32_t * NTdx
        npint32_t * aAdx
        npint32_t * bAdx

        double sqdelta = delta**2
        double aTdet, bTdet
        double termf[3]
        double bTE[2*3]
        double termLocal[3*3]
        double termNonloc[3*3]

        np.ndarray[double, ndim=1, mode="c"] dx = weights
        np.ndarray[double, ndim=1, mode="c"] dy = weights
        np.ndarray[double, ndim=1, mode="c"] fd = py_fd
        np.ndarray[double, ndim=1, mode="c"] aTE = np.zeros(2*3)
        np.ndarray[double, ndim=2, mode="c"] Ad = py_Ad
        np.ndarray[double, ndim=2, mode="c"] c_Verts = mesh.V
        np.ndarray[double, ndim=2, mode="c"] P = np.ascontiguousarray(py_P)
        np.ndarray[double, ndim=2, mode="c"] psi = np.ascontiguousarray(np.array([psi0, psi1, psi2]))

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(J):
        neigs = set_neighbour(J, &c_Triangles[0,0], &c_Triangles[aTdx,0])
        n = len(neigs)
        for i in range(n):
            Neighbours[aTdx, i] = neigs[i]


    start = time.time()

    # Loop over triangles ----------------------------------------------------------------------------------------------
    for aTdx in range(J_Omega):

        # Get index of ansatz functions in matrix A.-------------------
        # Continuous Galerkin
        aAdx = &c_Triangles[aTdx,0]
        # Discontinuous Galerkin
        # - Not implemented -

        # Copy coordinates of
        for j in range(2):
            aTE[2*0+j] = c_Verts[c_Triangles[aTdx,0], j]
            aTE[2*1+j] = c_Verts[c_Triangles[aTdx,1], j]
            aTE[2*2+j] = c_Verts[c_Triangles[aTdx,2], j]
        aTdet = cy_absDet(&aTE[0])

        # integrate over all elements
        c_doublevec_tozero(&termf[0], 3)
        f(&aTE[0], aTdet, &P[0,0], nP, &dx[0], &psi[0,0], &termf[0])

        # then assign to fd
        for a in range(3):
            # Assembly only happens in the interior of Omega only
            if c_Triangles[aTdx, a] < L_Omega:
                aAdxj = aAdx[a]
                fd[aAdxj] += termf[a]

        c_queue.push(aTdx)
        c_intvec_tozero(&visited[0], J)

        while not c_queue.empty():
            sTdx = c_queue.front()
            c_queue.pop()
            NTdx =  &Neighbours[sTdx,0]

            for j in range(4):
                bTdx = NTdx[j]
                if bTdx < J:
                    for j in range(2):
                        bTE[2*0+j] = c_Verts[c_Triangles[bTdx,0], j]
                        bTE[2*1+j] = c_Verts[c_Triangles[bTdx,1], j]
                        bTE[2*2+j] = c_Verts[c_Triangles[bTdx,2], j]
                    bTdet = cy_absDet(&bTE[0])

                    if visited[bTdx]==0:
                        inNbhd(&aTE[0], &bTE[0], sqdelta, &Mis_interact[0])
                        if c_intvec_any(&Mis_interact[0], 3):
                            c_queue.push(bTdx)
                            bAdx = &c_Triangles[bTdx,0]
                            c_doublevec_tozero(&termLocal[0], 3*3)
                            c_doublevec_tozero(&termNonloc[0], 3*3)
                            A(&aTE[0], aTdet, &bTE[0], bTdet, &P[0,0], nP, &dx[0], &dy[0], &psi[0,0], sqdelta,
                              c_intvec_all(&Mis_interact[0], 3), termLocal, termNonloc)

                            for a in range(3):
                                if c_Triangles[aTdx, a] < L_Omega:
                                    aAdxj = aAdx[a]
                                    for b in range(3):
                                        Ad[aAdxj, aAdx[b]] += termLocal[3*a+b]
                                        Ad[aAdxj, bAdx[b]] -= termNonloc[3*a+b]
                    visited[bTdx] = 1
        #print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / J_Omega * 100), "%", end="\r", flush=True)
    print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / J_Omega * 100), "%\n")
    total_time = time.time() - start
    print("Time needed", "{:1.2e}".format(total_time), " Sec")

    return py_Ad*2, py_fd

cdef list set_neighbour(int rows, npint32_t * Triangles, npint32_t * Vdx):
    """
    Find neighbour for index Tdx.

    :param Triangles:
    :param Vdx:
    :return:
    """
    cdef:
        int i, j, k, n
    idx = []

    for i in range(rows):
        n = 0
        for j in range(3):
            for k in range(3):
                if Triangles[3*i+j] == Vdx[k]:
                    n+=1
        if n >= 2:
            idx.append(i)

    return idx

cdef void f(double * aTE,
            double aTdet,
            double * P,
            int nP,
            double * dx,
            double * psi,
            double * termf):
    cdef:
        int i,a
        double x[2]

    for a in range(3):
        for i in range(nP):
            cy_toPhys(aTE, &P[2*i], &x[0])
            termf[a] += psi[7*a + i] * c_fPhys(&x[0]) * aTdet * dx[i]

cdef void A(double * aTE, double aTdet, double * bTE, double bTdet,
            double * P,
            int nP,
            double * dx,
            double * dy,
            double * psi,
            double sqdelta, bint is_allInteract,
      double * cy_termLocal, double * cy_termNonloc):
    cdef:
        int i=0, j=0, a, b
        double kerd, innerIntLocal, innerIntNonloc

    if is_allInteract:
        for a in range(3):
            for b in range(3):
                for i in range(nP):
                    innerIntLocal=0
                    innerIntNonloc=0
                    for j in range(nP):
                        innerIntLocal += c_kernelPhys(&P[2*i], &P[2*j], sqdelta) * dy[j]
                        innerIntNonloc += c_kernelPhys(&P[2*i], &P[2*j], sqdelta) * dy[j] * psi[7*b+j]
                    cy_termLocal[3*a+b] += psi[7*a+i] * psi[7*b+i] * innerIntLocal * dx[i] * aTdet*bTdet
                    cy_termNonloc[3*a+b] += psi[7*a+i] * innerIntNonloc * dx[i] * aTdet*bTdet
    else:
        cy_outerInt_full(&aTE[0], aTdet, &bTE[0], bTdet, P, nP, dx, dy, psi, sqdelta, cy_termLocal, cy_termNonloc)

cdef void cy_outerInt_full(double * aTE, double aTdet,
                           double * bTE, double bTdet,
                           double * P,
                           int nP,
                           double * dx,
                           double * dy,
                           double * psi,
                           double sqdelta,
                           double * cy_termLocal,
                           double * cy_termNonloc):
    cdef:
        int i=0, k=0, Rdx=0, a=0, b=0
        double x[2]
        double innerLocal=0
        double innerNonloc[3]
        double RT[9*3*2]

    for k in range(nP):
        cy_toPhys(&aTE[0], &P[2*k], &x[0])
        Rdx = cy_retriangulate(&x[0], bTE, sqdelta, &RT[0])
        cy_innerInt_retriangulate(x, bTE, P, nP, dy, sqdelta, Rdx, &RT[0], &innerLocal, &innerNonloc[0])
        for b in range(3):
            for a in range(3):
                cy_termLocal[3*a+b] += aTdet * psi[7*a+k] * psi[7*b+k] * dx[k] * innerLocal #innerLocal
                cy_termNonloc[3*a+b] += aTdet * psi[7*a+k] * dx[k] * innerNonloc[b] #innerNonloc

cdef cy_innerInt_retriangulate(double * x,
                               double * T,
                               double * P,
                               int nP, double * dy, double sqdelta, int Rdx,
                               double * RT, double * innerLocal, double * innerNonloc):
    cdef:
        int nRT = 0, i=0, k=0, rTdx=0, b=0
        double psi_rp=0, ker=0, rTdet=0
        double ry[2]
        double rp[2]

    innerLocal[0] = 0
    for b in range(3):
        innerNonloc[b] = 0
    if Rdx == 0:
        return

    for rTdx in range(Rdx):
        for i in range(nP):
            #cy_toPhys(RT[rTdx*3:rTdx*3+3, :], &P[2*i], &ry[0])
            cy_toPhys(&RT[2*3*rTdx], &P[2*i], &ry[0])
            cy_toRef(T, ry, rp)
            ker = c_kernelPhys(&x[0], &ry[0], sqdelta)
            rTdet = cy_absDet(&RT[2*3*rTdx])
            innerLocal[0] += (ker * dy[i]) * rTdet # Local Term
            for b in range(3):
                psi_rp = cy_evalPsi(rp, b)
                innerNonloc[b] += (psi_rp * ker * dy[i]) * rTdet # Nonlocal Term

cdef double cy_evalPsi(double * p, int psidx):
    if psidx == 0:
        return 1 - p[0] - p[1]
    elif psidx == 1:
        return p[0]
    elif psidx == 2:
        return p[1]
    else:
        abort()
        #raise ValueError("in cy_evalPsi. Invalid psi index.")

cdef int cy_retriangulate(double * x_center, double * TE, double sqdelta, double * out_RT):
        """ Retriangulates a given triangle.

        :param x_center: nd.array, real, shape (2,). Center of normball, e.g. pyhsical quadtrature point.
        :param T: clsTriangle. Triangle to be retriangulated.
        :return: list of clsTriangle.
        """
        # C Variables and Arrays.
        cdef:
            int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0
            double v=0, lam1=0, lam2=0, t=0, term1=0, term2=0
            double c_p[2]
            double c_q[2]
            double c_a[2]
            double c_b[2]
            double c_y1[2]
            double c_y2[2]
            # An upper bound for the number of intersections between a circle and a triangle is 9
            # Hence we can hardcode how much space needs to bee allocated
            double c_R[9][2]
            # Hence 9*3 is an upper bound to encode all resulting triangles
            #double c_RE[9*3][2]
        for i in range(9):
            for k in range(2):
                c_R[i][k] = 0.0

        for k in range(3):
            edgdx0 = k
            edgdx1 = (k+1) % 3

            for i in range(2):
                c_p[i] = TE[2*edgdx0+i]
                c_q[i] = TE[2*edgdx1+i]
                c_a[i] = c_q[i] - x_center[i]
                c_b[i] = c_p[i] - c_q[i]
            v = pow(c_vecdot(&c_a[0], &c_b[0], 2),2) - (c_vecdot(&c_a[0], &c_a[0], 2) - sqdelta)*c_vecdot(&c_b[0],&c_b[0], 2)

            if v >= 0:
                term1 = -c_vecdot(&c_a[0], &c_b[0], 2)/c_vecdot(&c_b[0], &c_b[0], 2)
                term2 = sqrt(v)/c_vecdot(&c_b[0], &c_b[0], 2)
                lam1 = term1 + term2
                lam2 = term1 - term2
                for i in range(2):
                    c_y1[i] = lam1*(c_p[i]-c_q[i]) + c_q[i]
                    c_y2[i] = lam2*(c_p[i]-c_q[i]) + c_q[i]

                if c_vecdist_sql2(&c_p[0], &x_center[0], 2) <= sqdelta:
                    for i in range(2):
                        c_R[Rdx][i] = c_p[i]
                    Rdx += 1
                if 0 <= lam1 <= 1:
                    for i in range(2):
                        c_R[Rdx][i] = c_y1[i]
                    Rdx += 1
                if (0 <= lam2 <= 1) and (scaldist_sql2(lam1, lam2) >= 1e-9):
                    for i in range(2):
                        c_R[Rdx][i] = c_y2[i]
                    Rdx += 1
            else:
                if c_vecdist_sql2(c_p, &x_center[0], 2)  <= sqdelta:
                    for i in range(2):
                        c_R[Rdx][i] = c_p[i]
                    Rdx += 1

        # Construct List of Triangles from intersection points
        if Rdx < 3:
            # In this case the content of the array out_RE will not be touched.
            return 0
        else:
            for k in range(Rdx - 2):
                for i in range(2):
                    # i is the index which runs first, then h (which does not exist here), then k
                    # hence if we increase i, the *-index (of the pointer) inreases in the same way.
                    # if we increase k, there is quite a 'jump'
                    out_RT[2 * (3 * k + 0) + i] = c_R[0][i]
                    out_RT[2 * (3 * k + 1) + i] = c_R[k + 1][i]
                    out_RT[2 * (3 * k + 2) + i] = c_R[k + 2][i]
            # Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

            return Rdx - 2 # So that, it acutally contains the number of triangles in the retriangulation

# Define Right side f
cdef double c_fPhys(double * x):
        """ Right side of the equation.

        :param x: nd.array, real, shape (2,). Physical point in the 2D plane
        :return: real
        """
        # f = 1
        return 1.0

cdef void cy_toPhys(double * E, double * p, double * out_x):
    cdef:
        int i=0
    for i in range(2):
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i]

cdef double c_kernelPhys(double * x, double * y, double sqdelta):
    return 4 / (pi * pow(sqdelta, 2))

cdef double c_vecdist_sql2(double * x, double * y, int length):
    """
    Computes squared l2 distance
    
    :param x: * double array
    :param y: * dobule array
    :param length: int length of vectors
    :return: double
    """
    cdef:
        double r=0
        int i=0
    for i in range(length):
        r += pow((x[i] - y[i]), 2)
    return r

cdef double scaldist_sql2(double x, double y):
    """
    Computes squared l2 distance
    
    :param x: double array
    :param y: double array
    :return: double
    """
    return pow((x-y), 2)

cdef double c_vecdot(double * x, double * y, int length):
    """
    Computes scalar product of two vectors.
    
    :param x: * double array
    :param y: * double array
    :param lengt: int length of vectors
    :return: double
    """
    cdef:
        double r=0
        int i=0
    for i in range(length):
        r += x[i]*y[i]
    return r

cdef void cy_solve2x2(double * A, double * b, double * x):
    cdef:
        int i=0, dx0 = 0, dx1 = 1
        double l=0, u=0

    # Column Pivot Strategy
    if abs(A[0]) < abs(A[2]):
        dx0 = 1
        dx1 = 0

    # Check invertibility
    if A[2*dx0] == 0:
        abort()
        #raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Decomposition
    l = A[2*dx1]/A[2*dx0]
    u = A[2*dx1+1] - l*A[2*dx0+1]

    # Check invertibility
    if u == 0:
        abort()
        #raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Solve
    x[1] = (b[dx1] - l*b[dx0])/u
    x[0] = (b[dx0] - A[2*dx0+1]*x[1])/A[2*dx0]
    return

cdef void cy_toRef(double * E, double * phys_x, double * ref_p):
    cdef:
        double M[2*2]
        double b[2]
        int i=0, j=0

    for i in range(2):
        M[2*i] = E[2*1+i] - E[2*0+i]
        M[2*i+1] = E[2*2+i] - E[2*0+i]
        b[i] = phys_x[i] - E[2*0+i]

    cy_solve2x2(&M[0], &b[0], &ref_p[0])
    return

cdef double cy_absDet(double * E):
    cdef:
        double M[2][2]
        int i=0
    for i in range(2):
        M[i][0] = E[2*1+i] - E[2*0+i]
        M[i][1] = E[2*2+i] - E[2*0+i]
    return abs(M[0][0]*M[1][1] - M[0][1]*M[1][0])

cdef void baryCenter(double * E, double * bary):
    cdef:
        int i
    bary[0] = 0
    bary[1] = 0
    for i in range(3):
        bary[0] += E[2*i+0]
        bary[1] += E[2*i+1]
    bary[0] = bary[0]/3
    bary[1] = bary[1]/3

cdef void inNbhd(double * aTE, double * bTE, double sqdelta, npint32_t * M):
    """
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. Compares the barycenter of bT with the vertices of Ea.

    :param aT: clsTriangle, Triangle a
    :param bT: clsTriangle, Triangle b
    :param delta: Size of L-2 ball
    :param v: Verbose mode.
    :return: ndarray, bool, shape (3,) Entry i is True if Ea[i] and the barycenter of Eb interact.
    """

    cdef:
        int i=0
        double bary[2]
    baryCenter(bTE, &bary[0])
    for i in range(3):
        M[i] = (c_vecdist_sql2(&aTE[2*i], &bary[0], 2) <= sqdelta)

    return

cdef void c_doublevec_tozero(double * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
        vec[i]  = 0.

cdef void c_intvec_tozero(int * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
        vec[i]  = 0

cdef int c_intvec_any(npint32_t * vec, int len):
    cdef int i=0
    for i in range(len):
            if vec[i] != 0:
                return 1
    return 0

cdef int c_intvec_all(npint32_t * vec, int len):
    cdef int i=0
    for i in range(len):
            if vec[i] == 0:
                return 0
    return 1