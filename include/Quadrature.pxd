cdef extern from "Quadrature.h":
    cdef cppclass Quadrature:
        Quadrature(const long dim_,
                   const double * Px_,
                   const double * dx_,
                   const long nPx_,
                   const double * Py_,
                   const double * dy_,
                   const long nPy_,
                   const double * Pg_,
                   const double * dg_,
                   const long tensorGaussDegree_) except +

