#-*- coding:utf-8 -*-
#cython: language_level=3
#cython: binding=True

# #cython: boundscheck=False, cdivision=True
# Setting this compiler directive will given a minor increase of speed.

cimport cython
from libcpp.queue cimport queue#, list
import numpy as np
from numpy.linalg import LinAlgError
from libc.math cimport sqrt, pow, pi
from conf import py_P, py_weights, delta, outerIntMethod, innerIntMethod
import time

def assemble(mesh):
    """**Assembly routine**

    :param mesh: clsFEM. Mesh containing the data. All other data is read from *conf.py*.
    :return: list of nd.array. Returns discretized matrix A and right side f.
    """

    #{"Verts": self.V, "Triangles": self.T, "J":self.J, "J_Omega":self.J_Omega,"L":self.L, "L_Omega":self.L_Omega, "K":self.K, "K_Omega":self.K_Omega}
    cdef:
        int J = mesh.J, J_Omega = mesh.J_Omega, L = mesh.L, L_Omega = mesh.L_Omega

    # Allocate Matrix A and right side f
    py_Ad = np.zeros((mesh.K_Omega, mesh.K))
    py_fd = np.zeros(mesh.K_Omega)

    # Define Basis
    psi0 = 1 - py_P[0, :] - py_P[1, :]
    psi1 = py_P[0, :]
    psi2 = py_P[1, :]
    py_psi = np.array([psi0, psi1, psi2])

    py_Neighbours = np.ones((mesh.J, 4), dtype=np.int32)*mesh.J
    py_visited = np.zeros(mesh.J, dtype=np.int32)
    py_Mis_interact = np.zeros(3, dtype=np.int32)

    cdef:
        queue[int] c_queue
        int nP = py_P.shape[1]
        int aTdx=0, bTdx=0, sTdx=0, i=0, k=0, j=0, h=0

        int [:] visited = py_visited
        int [:] Mis_interact = py_Mis_interact
        int [:,:] Neighbours = py_Neighbours
        int [:,:] Triangles = mesh.T
        int * NTdx
        int * aAdx
        int * bAdx

        double sqdelta = delta**2
        double [:] fd = py_fd
        double [:] dx = py_weights
        double [:] dy = py_weights
        double [:] termf = np.zeros(3)
        double [:,:] Ad = py_Ad
        double [:,:] P = py_P
        double [:,:] psi = py_psi
        double [:,:] Verts = mesh.V
        double [:,:] termLocal = np.zeros((3,3))
        double [:,:] termNonloc = np.zeros((3,3))
        cy_clsTriangle aT, bT



    neigs = []
    for aTdx in range(J):
        Vdx = Triangles[aTdx]
        neigs = set_neighbour(Triangles, Vdx)
        n = len(neigs)
        for i in range(n):
            Neighbours[aTdx, i] = neigs[i]

    # Loop over triangles --------------------------------------------------------------------------------------------------
    start = time.time()
    for aTdx in range(J_Omega): # Laufe über 0 bis KT_Omega (der Index der Dreiecke in Omega).
        aAdx = &(Triangles[aTdx])[0] # Index for both Omega or OmegaI
        aT = cy_clsTriangle(Verts[Triangles[aTdx][0]], Verts[Triangles[aTdx][1]], Verts[Triangles[aTdx][2]])

        # integrate over all elements
        cy_doublevec_tozero(termf, 3)
        f(aT, P, nP, dx, psi, termf)

        # then assign to fd
        for a in range(3):
            # Assembly only happens in the interior of Omega only
            if Triangles[aTdx, a] < L_Omega:
                aAdxj = aAdx[a]
                fd[aAdxj] += termf[a]

        c_queue.push(aTdx)
        cy_intvec_tozero(visited, J)

        while not c_queue.empty():
            sTdx = c_queue.front()
            c_queue.pop()
            NTdx =  &(Neighbours[sTdx])[0]

            for j in range(4):
                bTdx = NTdx[j]
                if bTdx < J:
                    bT = cy_clsTriangle(Verts[Triangles[bTdx][0]], Verts[Triangles[bTdx][1]], Verts[Triangles[bTdx][2]])

                    if visited[bTdx]==0:
                        inNbhd(aT, bT, sqdelta, Mis_interact)
                        if cy_intvec_any(Mis_interact, 3):
                            c_queue.push(bTdx)
                            bAdx = &(Triangles[bTdx])[0]
                            cy_doublearr_tozero(termLocal, 3, 3)
                            cy_doublearr_tozero(termNonloc, 3, 3)
                            A(aT, bT, P, nP, dx, dy, psi, sqdelta, cy_intvec_all(py_Mis_interact, 3), termLocal, termNonloc)

                            for a in range(3):
                                if Triangles[aTdx, a] < L_Omega:
                                    aAdxj = aAdx[a]
                                    for b in range(3):
                                        Ad[aAdxj, aAdx[b]] += termLocal[a][b]
                                        Ad[aAdxj, bAdx[b]] -= termNonloc[a][b]
                    visited[bTdx] = 1
        print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / J_Omega * 100), "%", end="\r", flush=True)
    print("aTdx: ", aTdx, "\t Neigs: ", np.sum(visited), "\t Progress: ", round(aTdx / J_Omega * 100), "%\n")
    total_time = time.time() - start
    print("Time needed", "{:1.2e}".format(total_time), " Sec")

    return py_Ad*2, py_fd

cdef list set_neighbour(int [:,:] Triangles, int [:] Vdx):
    """
    Find neighbour for index Tdx.

    :param Triangles:
    :param Vdx:
    :return:
    """
    cdef:
        int i, j, k, n
        int rows = len(Triangles)
    idx = []

    for i in range(rows):
        n = 0
        for j in range(3):
            for k in range(3):
                if Triangles[i,j] == Vdx[k]:
                    n+=1
        if n >= 2:
            idx.append(i)

    return idx

cdef void f(cy_clsTriangle aT, double [:,:] P, int nP, double [:] dx, double [:,:] psi, double [:] termf):
    cdef:
        int i,a
        double x[2]
        double absDet = aT.absDet()

    for a in range(3):
        for i in range(nP):
            aT.cy_toPhys(P[:, i], x)
            termf[a] += psi[a, i] * c_fPhys(&x[0] ) * absDet * dx[i]

cdef void A(cy_clsTriangle aT, cy_clsTriangle bT, double [:,:] P, int nP,
      double [:] dx, double [:] dy, double [:,:] psi, double sqdelta, bint is_allInteract,
      double [:, :] cy_termLocal, double [:,:] cy_termNonloc):
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
                        innerIntLocal += c_kernelPhys(&(P[:, i])[0], &(P[:, j])[0], sqdelta) * dy[j]
                        innerIntNonloc += c_kernelPhys(&(P[:, i])[0], &(P[:, j])[0], sqdelta) * dy[j] * psi[b, j]
                    cy_termLocal[a][b] += psi[a, i] * psi[b, i] * innerIntLocal * dx[i] * aT.absDet()*bT.absDet()
                    cy_termNonloc[a][b] += psi[a,i] * innerIntNonloc * dx[i] * aT.absDet()*bT.absDet()
    else:
        cy_outerInt_full(aT, bT, P, nP, dx, dy, psi, sqdelta, cy_termLocal, cy_termNonloc)

cdef void cy_outerInt_full(cy_clsTriangle aT, cy_clsTriangle bT,
                           double [:,:] P, int nP, double [:] dx, double [:] dy, double [:,:] psi,
                           double sqdelta,
                           double [:, :] cy_termLocal, double [:,:] cy_termNonloc):
    cdef:
        int i=0, k=0, Rdx=0
        double x[2]
        double innerLocal=0
        double innerNonloc[3]
        double [:,:] RT = np.zeros((9*3, 2))

    for k in range(nP):
        #cy_toPhys(aTE, P[:,k], x)
        aT.cy_toPhys( P[:,k], x)
        Rdx = cy_retriangulate(x, bT, sqdelta, RT)
        cy_innerInt_retriangulate(x, bT, P, nP, dy, sqdelta, Rdx, RT, &innerLocal, innerNonloc)
        for b in range(3):
            for a in range(3):
                cy_termLocal[a][b] += aT.absDet() * psi[a][k] * psi[b][k] * dx[k] * innerLocal #innerLocal
                cy_termNonloc[a][b] += aT.absDet() * psi[a][k] * dx[k] * innerNonloc[b] #innerNonloc

cdef cy_innerInt_retriangulate(double [:] x,
                                    cy_clsTriangle T,
                                    double [:,:] P, int nP, double [:] dy,
                                    double sqdelta,
                                    int Rdx, double[:,:] RT,
                                    double * innerLocal,
                                    double [:] innerNonloc):
    cdef:
        int nRT = 0, i=0, k=0, rTdx=0, b=0
        double psi_rp=0, ker=0, rTdet=0
        double [:] p
        double ry[2]
        double rp[2]

    innerLocal[0] = 0
    for b in range(3):
        innerNonloc[b] = 0
    if Rdx == 0:
        return

    for rTdx in range(Rdx):
        for i in range(nP):
            cy_toPhys(RT[rTdx*3:rTdx*3+3, :], P[:, i], ry)
            T.cy_toRef(ry, rp)
            ker = c_kernelPhys(&x[0], &ry[0], sqdelta)
            rTdet = cy_absDet(RT[rTdx*3:rTdx*3+3, :])
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
        raise ValueError("in cy_evalPsi. Invalid psi index.")

cdef int cy_retriangulate(double [:] x_center, cy_clsTriangle T, double sqdelta, double [:,:] out_RE):
        """ Retriangulates a given triangle.

        :param x_center: nd.array, real, shape (2,). Center of normball, e.g. pyhsical quadtrature point.
        :param T: clsTriangle. Triangle to be retriangulated.
        :return: list of clsTriangle.
        """
        # C Variables and Arrays.
        cdef:
            int i=0, k=0, edgdx0=0, edgdx1=0
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
        # Memory Views.
        Rdx=0
        for i in range(9):
            for k in range(2):
                c_R[i][k] = 0.0

        for k in range(3):
            edgdx0 = k
            edgdx1 = (k+1) % 3

            for i in range(2):
                c_p[i] = T.E[edgdx0][i]
                c_q[i] = T.E[edgdx1][i]
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
                    out_RE[3*k + 0, i] = c_R[0][i]
                    out_RE[3*k + 1, i] = c_R[k+1][i]
                    out_RE[3*k + 2, i] = c_R[k+2][i]
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

cdef void cy_toPhys(double [:,:] E, double [:] p, double [:] out_x):
    cdef:
        int i=0
    for i in range(2):
        out_x[i] = (E[1][i] - E[0][i])*p[0] + (E[2][i] - E[0][i])*p[1] + E[0][i]

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

cdef void cy_solve2x2(double [:,:] A, double [:] b, double [:] x):
    cdef:
        int i=0, dx0 = 0, dx1 = 1
        double l=0, u=0

    # Column Pivot Strategy
    if abs(A[0,0]) < abs(A[1,0]):
        dx0 = 1
        dx1 = 0

    # Check invertibility
    if A[dx0,0] == 0:
        raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Decomposition
    l = A[dx1,0]/A[dx0,0]
    u = A[dx1,1] - l*A[dx0,1]

    # Check invertibility
    if u == 0:
        raise LinAlgError("in cy_solve2x2. Matrix not invertible.")

    # LU Solve
    x[1] = (b[dx1] - l*b[dx0])/u
    x[0] = (b[dx0] - A[dx0,1]*x[1])/A[dx0,0]
    return

cdef void cy_toRef(double [:,:] E, double [:] phys_x, double [:] ref_p):
    cdef:
        double M[2][2]
        double b[2]
        int i=0, j=0

    for i in range(2):
        M[i][0] = E[1][i] - E[0][i]
        M[i][1] = E[2][i] - E[0][i]
        b[i] = phys_x[i] - E[0][i]

    cy_solve2x2(M, b, ref_p)
    return

cdef double cy_absDet(double [:,:] E):
    cdef:
        double M[2][2]
    for i in range(2):
        M[i][0] = E[1][i] - E[0][i]
        M[i][1] = E[2][i] - E[0][i]
    return abs(M[0][0]*M[1][1] - M[0][1]*M[1][0])

cdef void baryCenter(double [:,:] E, double [:] bary):
    cdef:
        int i
    bary[0] = 0
    bary[1] = 0
    for i in range(3):
        bary[0] += E[i, 0]
        bary[1] += E[i, 1]
    bary[0] = bary[0]/3
    bary[1] = bary[1]/3


####################
cdef class cy_clsTriangle:
    cdef:
        double E[3][2]
        double M[2][2]
        double b[2]
        double baryCenter_[2]
        bint is_baryCenter_

    cdef public:
        double absDet_

    def __init__(self, E0, E1, E2):
        self.absDet_ = 0
        self.is_baryCenter_ = 0
        for i in range(2):
            self.E[0][i] = E0[i]
            self.E[1][i] = E1[i]
            self.E[2][i] = E2[i]

    cdef double absDet(self):
        cdef int i=0
        if self.absDet_!=0:
            return self.absDet_
        else:
            for i in range(2):
                self.M[i][0] = self.E[1][i] - self.E[0][i]
                self.M[i][1] = self.E[2][i] - self.E[0][i]
            self.absDet_ = abs(self.M[0][0]*self.M[1][1] - self.M[0][1]*self.M[1][0])
            return self.absDet_

    cdef void cy_toPhys(self, double [:] p, double [:] out_x):
        cdef:
            int i=0
        for i in range(2):
            out_x[i] = (self.E[1][i] - self.E[0][i])*p[0] + (self.E[2][i] - self.E[0][i])*p[1] + self.E[0][i]
        return

    cdef void cy_toRef(self, double [:] phys_x, double [:] ref_p):
        cdef:
            double b[2]
            int i=0, j=0

        for i in range(2):
            self.M[i][0] = self.E[1][i] - self.E[0][i]
            self.M[i][1] = self.E[2][i] - self.E[0][i]
            self.b[i] = phys_x[i] - self.E[0][i]

        cy_solve2x2(self.M, self.b, ref_p)
        return

    cdef void baryCenter(self, double [:] bary):
        cdef:
            int i
        if self.is_baryCenter_:
            for i in range(2):
                bary[i] = self.baryCenter_[i]
            return

        for i in range(2):
            bary[i] = (self.E[0][i] + self.E[1][i] + self.E[2][i])/3.
            self.baryCenter_[i] = bary[i]
        self.is_baryCenter_ = 1
        return

cdef void inNbhd(cy_clsTriangle aT, cy_clsTriangle bT, double sqdelta, int [:] M):
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
    bT.baryCenter(bary)
    for i in range(3):
        M[i] = (c_vecdist_sql2(&(aT.E[i])[0], &bary[0], 2) <= sqdelta)

    return

cdef void cy_vec_tozero(double [:] vec, int len):
    cdef int i=0
    for i in range(len):
        vec[i]  = 0.


cdef void cy_doublevec_tozero(double [:] vec, int len):
    cdef int i=0
    for i in range(len):
        vec[i]  = 0.

cdef void cy_intvec_tozero(int [:] vec, int len):
    cdef int i=0
    for i in range(len):
        vec[i]  = 0

cdef void cy_doublearr_tozero(double [:,:] arr, int rows, int cols):
    cdef int i=0, j=0
    for i in range(rows):
        for j in range(cols):
            arr[i, j]  = 0.

cdef int cy_intvec_any(int [:] vec, int len):
    cdef int i=0
    for i in range(len):
            if vec[i] != 0:
                return 1
    return 0

cdef int cy_intvec_all(int [:] vec, int len):
    cdef int i=0
    for i in range(len):
            if vec[i] == 0:
                return 0
    return 1