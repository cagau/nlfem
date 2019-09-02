#-*- coding:utf-8 -*-
#cython: language_level=3
#cython: boundscheck=False, wraparound=False, cdivision=True
# Setting this compiler directive will given a minor increase of speed.

# Cython Imports
cimport cython
cimport numpy as np
from cython.parallel import prange, threadid
# Python Imports
import numpy as np
from numpy.linalg import LinAlgError
import time
# C Imports
from libcpp.queue cimport queue
from libc.math cimport sqrt, pow, pi, cos
from libc.stdlib cimport malloc, free, abort

ctypedef np.int32_t npint32_t

def assemble(
        # Mesh information ------------------------------------
        int K, int K_Omega,
        int J, int J_Omega, # Number of Triangles and number of Triangles in Omega
        int L, int L_Omega, # Number of vertices (in case of CG = K and K_Omega)
        # Map Triangle index (Tdx) -> index of Vertices in Verts (Vdx = Triangle[Tdx] array of int, shape (3,))
        npint32_t [:,:] c_Triangles,
        # Map Vertex Index (Vdx) -> Coordinate of some Vertex i of some Triangle (E[i])
        double [:,:] c_Verts,
        # Cython interface of quadrature points
        double [:,:] P,
        # Weights for quadrature rule
        double [:] dx,
        double [:] dy,
        double delta
    ):

    ## Data Matrix ----------------------------------------
    # Allocate Matrix compute_A and right side f
    py_Ad = np.zeros((K_Omega, K))
    py_fd = np.zeros(K_Omega)

    # Define Basis-----------------------------------------
    cdef:
        int i=0
        # Number of integration points
        int nP = P.shape[0] # Does not differ in inner and outer integral!

    psi0 = np.zeros(nP)
    psi1 = np.zeros(nP)
    psi2 = np.zeros(nP)

    for i in range(nP):
        psi0[i] = (1 - P[i, 0] - P[i, 1])
        psi1[i] = P[i, 0]
        psi2[i] = P[i, 1]

    cdef:
        # Triangles -------------------------------------------------
        # Loop index of Triangles
        int aTdx=0

        # Cython interface of C-aligned arrays of solution and right side
        double[:] fd = py_fd
        double[:,:] Ad = py_Ad
        # Cython interface of Ansatzfunctions
        double[:,:] psi = np.ascontiguousarray(np.array([psi0, psi1, psi2]))

        # List of neighbours of each triangle, Neighbours[Tdx] returns row with Neighbour indices
        npint32_t[:,:] Neighbours = np.zeros((J, 4), dtype=np.int32)

        # Squared interaction horizon
        double sqdelta = pow(delta,2)

    # Setup adjaciency graph of the mesh --------------------------
    neigs = []
    for aTdx in range(J):
        neigs = set_neighbour(J, &c_Triangles[0,0], &c_Triangles[aTdx,0])
        n = len(neigs)
        for i in range(n):
            Neighbours[aTdx, i] = neigs[i]


    start = time.time()

    # Loop over triangles ----------------------------------------------------------------------------------------------
    for aTdx in prange(J_Omega, nogil=True):
        par_assemble(aTdx, Ad, fd, c_Triangles, c_Verts, J , J_Omega, L, L_Omega, nP, P, dx, dy, psi, sqdelta, Neighbours)

    total_time = time.time() - start
    print("\nTime needed", "{:1.2e}".format(total_time), " Sec")

    return py_Ad*2, py_fd

cdef void par_assemble(int aTdx,
                    double [:,:] Ad,
                    double [:] fd,
                    npint32_t [:,:] c_Triangles,
                    double [:,:] c_Verts,
                    # Number of Triangles and number of Triangles in Omega
                    int J, int J_Omega,
                    # Number of vertices (in case of CG = K and K_Omega)
                    int L, int L_Omega,
                    int nP, double [:,:] P,
                    double [:] dx,
                    double [:] dy,
                    double [:,:] psi,
                    double sqdelta,
                    npint32_t [:,:] Neighbours
                   ) nogil:

     ## BFS ------------------------------------------------
    # Allocate Graph of Neighbours
    # Further definitions
    cdef int *visited = <int *> malloc(J*sizeof(int))
    cdef npint32_t *Mis_interact = <npint32_t *> malloc(3*sizeof(npint32_t))

    cdef:
        # General Loop Indices ---------------------------------------
        int i=0, k=0, j=0, h=0, bTdx
       # Breadth First Search --------------------------------------
        # Loop index of current outer triangle in BFS
        int sTdx=0
        # Queue for Breadth first search
        queue[int] c_queue
        # List of visited triangles
        #np.ndarray[int, ndim=1, mode="c"] visited = py_visited
        # Matrix telling whether some vertex of Triangle a interactions with the baryCenter of Triangle b
        #np.ndarray[npint32_t, ndim=1, mode="c"] Mis_interact = py_Mis_interact

        npint32_t * NTdx
        # Determinant of Triangle a and b.
        double aTdet, bTdet, id=0
        # Vector containing the coordinates of the vertices of a Triangle
        double aTE[2*3]
        double bTE[2*3]
        # Integration information ------------------------------------
        # Loop index of basis functions
        int a=0, b=0, aAdxj =0
        # (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
        npint32_t * aAdx
        npint32_t * bAdx

        # Buffers for integration solutions
        double termf[3]
        double termLocal[3*3]
        double termNonloc[3*3]

    # Get index of ansatz functions in matrix compute_A.-------------------
    # Continuous Galerkin
    aAdx = &c_Triangles[aTdx,0]
    # Discontinuous Galerkin
    # - Not implemented -

    # Prepare Triangle information aTE and aTdet ------------------
    # Copy coordinates of Triange a to aTE.
    # this is done fore convenience only, actually those are unnecessary copies!
    for j in range(2):
        aTE[2*0+j] = c_Verts[c_Triangles[aTdx,0], j]
        aTE[2*1+j] = c_Verts[c_Triangles[aTdx,1], j]
        aTE[2*2+j] = c_Verts[c_Triangles[aTdx,2], j]
    # compute Determinant
    aTdet = absDet(&aTE[0])

    # Assembly of right side ---------------------------------------
    # We unnecessarily integrate over vertices which might lie on the boundary of Omega for convenience here.
    doubleVec_tozero(&termf[0], 3) # Initialize Buffer
    f(&aTE[0], aTdet, &P[0,0], nP, &dx[0], &psi[0,0], &termf[0]) # Integrate and fill buffer

    # Add content of buffer to the right side.
    for a in range(3):
        # Assembly happens in the interior of Omega only, so we throw away some values
        if c_Triangles[aTdx, a] < L_Omega:
            aAdxj = aAdx[a]
            fd[aAdxj] += termf[a]
    # Of course some uneccessary computation happens but only for some verticies of thos triangles which lie
    # on the boundary. This saves us from the pain to carry the information (a) into the integrator f.

    # BFS -------------------------------------------------------------
    # Intialize search queue with current outer triangle
    c_queue.push(aTdx)
    # Initialize vector of visited triangles with 0
    intVec_tozero(&visited[0], J)

    # Check whether BFS is over.
    while not c_queue.empty():
        # Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
        sTdx = c_queue.front()
        c_queue.pop()
        # Get all the neighbours of sTdx.
        NTdx =  &Neighbours[sTdx,0]
        # Run through the list of neighbours. (4 at max)
        for j in range(4):
            # The next valid neighbour is our candidate for the inner Triangle b.
            bTdx = NTdx[j]

            # Check how many neighbours sTdx has. It can be 4 at max. (Itself, and the three others)
            # In order to be able to store the list as contiguous array we fill upp the empty spots with J
            # i.e. the total number of Triangles, which cannot be an index.
            if bTdx < J:

                # Prepare Triangle information bTE and bTdet ------------------
                # Copy coordinates of Triange b to bTE.
                # again this is done fore convenience only, actually those are unnecessary copies!
                for j in range(2):
                    bTE[2*0+j] = c_Verts[c_Triangles[bTdx,0], j]
                    bTE[2*1+j] = c_Verts[c_Triangles[bTdx,1], j]
                    bTE[2*2+j] = c_Verts[c_Triangles[bTdx,2], j]
                bTdet = absDet(&bTE[0])

                # Check wheter bTdx is already visited.
                if visited[bTdx]==0:

                    # Check whether the verts of aT interact with (bary Center of) bT
                    inNbhd(&aTE[0], &bTE[0], sqdelta, &Mis_interact[0])
                    # If any of the verts interact we enter the retriangulation and integration
                    if npint32Vec_any(&Mis_interact[0], 3):
                        # Retriangulation and integration ------------------------

                        # Get (pointer to) intex of basis function (in Continuous Galerkin)
                        bAdx = &c_Triangles[bTdx,0]

                        # Assembly of matrix ---------------------------------------
                        doubleVec_tozero(&termLocal[0], 3 * 3) # Initialize Buffer
                        doubleVec_tozero(&termNonloc[0], 3 * 3) # Initialize Buffer
                        # Compute integrals and write to buffer
                        compute_A(&aTE[0], aTdet, &bTE[0], bTdet,
                                  &P[0,0], nP, &dx[0], &dy[0], &psi[0,0], sqdelta,
                                  intVec_all(&Mis_interact[0], 3), termLocal, termNonloc)
                        # Copy buffer into matrix. Again solutions which lie on the boundary are ignored
                        for a in range(3):
                            if c_Triangles[aTdx, a] < L_Omega:
                                aAdxj = aAdx[a]
                                for b in range(3):
                                    Ad[aAdxj, aAdx[b]] += termLocal[3*a+b]
                                    Ad[aAdxj, bAdx[b]] -= termNonloc[3*a+b]

                        # If bT interacts it will be a candidate for our BFS, so it is added to the queue
                        if doubleVec_any(termNonloc, 3 * 3) or doubleVec_any(termLocal, 3 * 3):
                            c_queue.push(bTdx)
                            # In order to further speed up the integration we only check whether the integral
                            # (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                            # However, this works only if we can guarantee that interacting triangles do actually
                            # also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere for example.
                            # This works for constant kernels, or franctional kernels
                            # The effect of this more precise criterea depends on delta and meshsize.
                # Mark bTdx as visited
                visited[bTdx] = 1
    # Be careful. With gil, will allow to throw python exceptions, but it leads to a breakdown of performance
    # by a factor of x3, as the code has to jump between gil and nogil!
    id = threadid()
    with gil:
        print("aTdx: ", aTdx, "id:", id, "\t Progress: ", round(aTdx / J_Omega * 100), "%", end="\r", flush=True)
    free(visited)
    free(Mis_interact)

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
            double * termf) nogil:
    cdef:
        int i,a
        double x[2]

    for a in range(3):
        for i in range(nP):
            toPhys(aTE, &P[2 * i], &x[0])
            termf[a] += psi[7*a + i] * fPhys(&x[0]) * aTdet * dx[i]

cdef void compute_A(double * aTE, double aTdet, double * bTE, double bTdet,
                    double * P,
                    int nP,
                    double * dx,
                    double * dy,
                    double * psi,
                    double sqdelta, bint is_allInteract,
                    double * cy_termLocal, double * cy_termNonloc) nogil:
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
        outerInt_full(&aTE[0], aTdet, &bTE[0], bTdet, P, nP, dx, dy, psi, sqdelta, cy_termLocal, cy_termNonloc)

cdef void outerInt_full(double * aTE, double aTdet,
                        double * bTE, double bTdet,
                        double * P,
                        int nP,
                        double * dx,
                        double * dy,
                        double * psi,
                        double sqdelta,
                        double * cy_termLocal,
                        double * cy_termNonloc) nogil:
    cdef:
        int i=0, k=0, Rdx=0, a=0, b=0
        double x[2]
        double innerLocal=0
        double innerNonloc[3]
        double RT[9*3*2]

    for k in range(nP):
        toPhys(&aTE[0], &P[2 * k], &x[0])
        Rdx = retriangulate(&x[0], bTE, sqdelta, &RT[0])
        innerInt_retriangulate(x, bTE, P, nP, dy, sqdelta, Rdx, &RT[0], &innerLocal, &innerNonloc[0])
        for b in range(3):
            for a in range(3):
                cy_termLocal[3*a+b] += aTdet * psi[7*a+k] * psi[7*b+k] * dx[k] * innerLocal #innerLocal
                cy_termNonloc[3*a+b] += aTdet * psi[7*a+k] * dx[k] * innerNonloc[b] #innerNonloc

cdef void innerInt_retriangulate(double * x,
                                 double * T,
                                 double * P,
                                 int nP, double * dy, double sqdelta, int Rdx,
                                 double * RT, double * innerLocal, double * innerNonloc) nogil:
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
            #toPhys(RT[rTdx*3:rTdx*3+3, :], &P[2*i], &ry[0])
            toPhys(&RT[2 * 3 * rTdx], &P[2 * i], &ry[0])
            toRef(T, ry, rp)
            ker = c_kernelPhys(&x[0], &ry[0], sqdelta)
            rTdet = absDet(&RT[2 * 3 * rTdx])
            innerLocal[0] += (ker * dy[i]) * rTdet # Local Term
            for b in range(3):
                psi_rp = evalBasisfunction(rp, b)
                innerNonloc[b] += (psi_rp * ker * dy[i]) * rTdet # Nonlocal Term

cdef double evalBasisfunction(double * p, int psidx) nogil:
    if psidx == 0:
        return 1 - p[0] - p[1]
    elif psidx == 1:
        return p[0]
    elif psidx == 2:
        return p[1]
    else:
        #with gil:
        #    raise ValueError("in evalBasisfunction. Invalid psi index.")
        abort() # Cython likes to ignore Python Excetions, I hope this line helps.

cdef int retriangulate(double * x_center, double * TE, double sqdelta, double * out_RT) nogil:
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
            v = pow(vec_dot(&c_a[0], &c_b[0], 2), 2) - (vec_dot(&c_a[0], &c_a[0], 2) - sqdelta) * vec_dot(&c_b[0], &c_b[0], 2)

            if v >= 0:
                term1 = -vec_dot(&c_a[0], &c_b[0], 2) / vec_dot(&c_b[0], &c_b[0], 2)
                term2 = sqrt(v) / vec_dot(&c_b[0], &c_b[0], 2)
                lam1 = term1 + term2
                lam2 = term1 - term2
                for i in range(2):
                    c_y1[i] = lam1*(c_p[i]-c_q[i]) + c_q[i]
                    c_y2[i] = lam2*(c_p[i]-c_q[i]) + c_q[i]

                if vec_sqL2dist(&c_p[0], &x_center[0], 2) <= sqdelta:
                    for i in range(2):
                        c_R[Rdx][i] = c_p[i]
                    Rdx += 1
                if 0 <= lam1 <= 1:
                    for i in range(2):
                        c_R[Rdx][i] = c_y1[i]
                    Rdx += 1
                if (0 <= lam2 <= 1) and (scal_sqL2dist(lam1, lam2) >= 1e-9):
                    for i in range(2):
                        c_R[Rdx][i] = c_y2[i]
                    Rdx += 1
            else:
                if vec_sqL2dist(c_p, &x_center[0], 2)  <= sqdelta:
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
cdef double fPhys(double * x) nogil:
        """ Right side of the equation.

        :param x: nd.array, real, shape (2,). Physical point in the 2D plane
        :return: real
        """
        # f = 1
        return 1.0

cdef void toPhys(double * E, double * p, double * out_x) nogil:
    cdef:
        int i=0
    for i in range(2):
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i]

cdef double c_kernelPhys(double * x, double * y, double sqdelta) nogil:
    # Nonconstant Kernel does not yet work
    # pow(1-vec_sqL2dist(x,y, 2)/sqdelta, 2)
    return 4 / (pi * pow(sqdelta, 2))

cdef double vec_sqL2dist(double * x, double * y, int length) nogil:
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

cdef double scal_sqL2dist(double x, double y) nogil:
    """
    Computes squared l2 distance
    
    :param x: double array
    :param y: double array
    :return: double
    """
    return pow((x-y), 2)

cdef double vec_dot(double * x, double * y, int length) nogil:
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

cdef void solve2x2(double * A, double * b, double * x) nogil:
    cdef:
        int i=0, dx0 = 0, dx1 = 1
        double l=0, u=0

    # Column Pivot Strategy
    if absolute(A[0]) < absolute(A[2]):
        dx0 = 1
        dx1 = 0

    # Check invertibility
    if A[2*dx0] == 0:
        #with gil:
        #    raise LinAlgError("in solve2x2. Matrix not invertible.")
        abort() # Cython likes to ignore Python Excetions, I hope this line helps.

    # LU Decomposition
    l = A[2*dx1]/A[2*dx0]
    u = A[2*dx1+1] - l*A[2*dx0+1]

    # Check invertibility
    if u == 0:
        #with gil:
        #    raise LinAlgError("in solve2x2. Matrix not invertible.")
        abort() # Cython likes to ignore Python Excetions, I hope this line helps.

    # LU Solve
    x[1] = (b[dx1] - l*b[dx0])/u
    x[0] = (b[dx0] - A[2*dx0+1]*x[1])/A[2*dx0]
    return

cdef void toRef(double * E, double * phys_x, double * ref_p) nogil:
    cdef:
        double M[2*2]
        double b[2]
        int i=0, j=0

    for i in range(2):
        M[2*i] = E[2*1+i] - E[2*0+i]
        M[2*i+1] = E[2*2+i] - E[2*0+i]
        b[i] = phys_x[i] - E[2*0+i]

    solve2x2(&M[0], &b[0], &ref_p[0])
    return

cdef double absDet(double * E) nogil:
    cdef:
        double out
        double M[2][2]
        int i=0
    for i in range(2):
        M[i][0] = E[2*1+i] - E[2*0+i]
        M[i][1] = E[2*2+i] - E[2*0+i]
    out =  absolute(M[0][0]*M[1][1] - M[0][1]*M[1][0])
    return out

cdef double absolute(double value) nogil:
    if value < 0 :
        return -value
    else:
        return value

cdef void baryCenter(double * E, double * bary) nogil:
    cdef:
        int i
    bary[0] = 0
    bary[1] = 0
    for i in range(3):
        bary[0] += E[2*i+0]
        bary[1] += E[2*i+1]
    bary[0] = bary[0]/3
    bary[1] = bary[1]/3

cdef void inNbhd(double * aTE, double * bTE, double sqdelta, npint32_t * M) nogil:
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
        M[i] = (vec_sqL2dist(&aTE[2 * i], &bary[0], 2) <= sqdelta)

    return

cdef void doubleVec_tozero(double * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
        vec[i]  = 0.

cdef void intVec_tozero(int * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
        vec[i]  = 0

cdef int npint32Vec_any(npint32_t * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
            if vec[i] != 0:
                return 1
    return 0

cdef int doubleVec_any(double * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
            if vec[i] != 0:
                return 1
    return 0

cdef int intVec_all(npint32_t * vec, int len) nogil:
    cdef int i=0
    for i in range(len):
            if vec[i] == 0:
                return 0
    return 1