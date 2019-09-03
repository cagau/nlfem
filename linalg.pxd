# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

cdef extern from "clinalg.cpp":
    int doubleVec_any(double * vec, int len) nogil
    void doubleVec_tozero(double * vec, int len) nogil
    void intVec_tozero(int * vec, int len) nogil
    void baryCenter(double * E, double * bary) nogil
    double absolute(double value) nogil
    double absDet(double * E) nogil
    void toRef(double * E, double * phys_x, double * ref_p) nogil
    void solve2x2(double * A, double * b, double * x) nogil
    double vec_dot(double * x, double * y, int length) nogil
    double scal_sqL2dist(double x, double y) nogil
    double vec_sqL2dist(double * x, double * y, int length) nogil
    void toPhys(double * E, double * p, double * out_x) nogil
    int retriangulate(double * x_center, double * TE, double sqdelta, double * out_RT) nogil
    double evalBasisfunction(double * p, int psidx) nogil
    void innerInt_retriangulate(double * x, double * T, double * P, int nP, double * dy, double sqdelta,
                                int Rdx, double * RT, double * innerLocal, double * innerNonloc) nogil
    double c_kernelPhys(double * x, double * y, double sqdelta) nogil
    void outerInt_full(double * aTE, double aTdet, double * bTE, double bTdet, double * P, int nP, double * dx,
                        double * dy, double * psi, double sqdelta, double * cy_termLocal, double * cy_termNonloc) nogil
    int longVec_all(long * vec, int len) nogil
    int longVec_any(long * vec, int len) nogil
    void inNbhd(double * aTE, double * bTE, double sqdelta, long * M) nogil
    double fPhys(double * x) nogil
    void compute_A(double * aTE, double aTdet, double * bTE, double bTdet, double * P, int nP, double * dx,
                   double * dy, double * psi, double sqdelta, bint is_allInteract,
                   double * cy_termLocal, double * cy_termNonloc) nogil
    void compute_f(double * aTE,
                    double aTdet,
                    double * P,
                    int nP,
                    double * dx,
                    double * psi,
                    double * termf) nogil
    void par_assemble( double * Ad,
                    int K,
                    double * fd,
                    long * c_Triangles,
                    double * c_Verts,
                    # Number of Triangles and number of Triangles in Omega
                    int J, int J_Omega,
                    # Number of vertices (in case of CG = K and K_Omega)
                    int L, int L_Omega,
                    int nP, double * P,
                    double * dx,
                    double * dy,
                    double * psi,
                    double sqdelta,
                    long * Neighbours
                   ) nogil