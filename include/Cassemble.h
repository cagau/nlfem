#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(const double * x_center, const double * TE, const double sqdelta, double * out_reTriangle_list, const int is_placePointOnCap);

struct QuadratureStruct{
    double * Px;
    double * Py;
    double * dx;
    double * dy;
    int nPx;
    int nPy;
};
typedef QuadratureStruct QuadratureType;

struct MeshStruct{
    int K_Omega;
    int K;
    long * ptrTriangles;
    double * ptrVerts;
    // Number of Triangles and number of Triangles in Omega
    int J;
    int J_Omega;
    // Number of vertices (in case of CG = K and K_Omega)
    int L;
    int L_Omega;
    double sqdelta;
    long * ptrNeighbours;
    int is_DiscontinuousGalerkin;
    int is_NeumannBoundary;
    int dim;
    int dVertex;
};
typedef MeshStruct MeshType;



// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(  const MeshType & mesh,
                    const QuadratureType & quadRule,
                    double * Ad,
                    double * fd
);
void par_assembleMass(double *, long *, double *, int, int, int, double *, double *);
void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *);
double compute_area(double *, double, long, double *, double, long, double *, int, double *, double);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *);

//[DEBUG]
//void relativePosition(double *, double *, double *, double *, double *);
//void order(double *, int, double *);

#endif /* Cassemble.h */