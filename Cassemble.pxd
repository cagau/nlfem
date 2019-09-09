# In this file all declarations of the C functions happen, in order to make them visible to cimport.
# If this .pxd is compiled with setup_linalg.py it results in a standalone package which can be cimported.
# In order import (to python) see linalg.pyx

cdef extern from "Cassemble.cpp":
    # Model ---------------------------------------------------------------------------------------------------------------
    double model_f(double *) nogil
    double model_kernel(double *, double *, double) nogil
    void model_basisFunction(double *, int, double *) nogil
    void inNbhd(double *, double *, double, long *) nogil
    double model_basisFunction(double * , int ) nogil
    
    # Integration ---------------------------------------------------------------------------------------------------------
    void innerInt_bary(double *, double *, double *, int, double *, double, int, double *, double *, double *, double *) nogil
    void innerInt_retriangulate(double *, double *, double *, int , double *, double, int, double *, double *, double *) nogil
    void outerInt_full(double *, double, double * , double , double *, int, double *, double *, double *, double, double *, double *) nogil
    int retriangulate(double *, double *, double, double *) nogil
    
    # Compute A and f -----------------------------------------------------------------------------------------------------
    void compute_f(double *, double, double *, int, double *, double *, double *) nogil
    void compute_A(double *, double, double *, double, double *, int, double *, double *, double *, double, bool, double *, double *) nogil

    # Assembly algorithm with BFS -----------------------------------------------------------------------------------------
    void par_assemble(double *, int, double *, long *, double *,int, int, int, int, int, double *, double *, double *, double, long *) nogil
    void par_evaluateA(double *, double *, int, long *, double *,int, int, int, int, int, double *, double *, double *, double, long *) nogil
    void par_assemblef(double *, long *, double *, int, int, int, double *, double *) nogil

    # Math functions ------------------------------------------------------------------------------------------------------
    void solve2x2(double *, double *, double *) nogil        # Solve 2x2 System with LU
    
    # Matrix operations (via * only) --------------------------------
    # Double
    double absDet(double *) nogil                            # Compute determinant
    void baryCenter(double *, double *) nogil                # Bary Center
    void toRef(double *, double *, double *) nogil           # Pull point to Reference Element (performs 2x2 Solve)
    void toPhys(double *, double *, double *) nogil          # Push point to Physical Element
    
    # Vector operations ---------------------------------------------
    # Double
    double vec_sqL2dist(double *, double *, int) nogil       # L2 Distance
    double vec_dot(double * x, double * y, int len) nogil    # Scalar Product
    int doubleVec_any(double *, int) nogil                   # Any
    void doubleVec_tozero(double *, int) nogil               # Reset to zero
    
    # Long
    int longVec_all(long *, int) nogil                       # All
    int longVec_any(long *, int) nogil                       # Any
    
    # Int
    void intVec_tozero(int *, int) nogil                     # Reset to zero
    
    # Scalar ----------------------------------------------------------
    double absolute(double) nogil                            # Get absolute value
    double scal_sqL2dist(double x, double y) nogil           # L2 Distance