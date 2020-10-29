cdef extern from "Mesh.h":
    cdef cppclass Mesh:
            Mesh(const long * ptrElements_,
                 const long * ptrElementLabels_,
                 const double * ptrVerts_,
                 const long nE_,
                 const long nE_Omega_,
                 const int nV_,
                 const int nV_Omega_,
                 const long * ptrNeighborIndices_,
                 const long * ptrNeighborIndexPtr_,
                 const long dim_,
                 const double maxDiameter_
                 ) except +
