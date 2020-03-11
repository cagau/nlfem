#ifndef CASSEMBLE_H
#define CASSEMBLE_H
#include <armadillo>
#include "MeshBuilder.h"

struct ElementStruct{
    arma::vec matE;
    double * E;
    int dim;
    long label;
    double absDet;
    int signDet;
};
typedef ElementStruct ElementType;

// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(const double * x_center, const double * TE, const MeshType & mesh, double * out_reTriangle_list, const int is_placePointOnCap);

void par_assemble( double * ptrAd, double * fd, MeshType & mesh, QuadratureType & quadRule,
        const int is_DiscontinuousGalerkin, const int is_NeumannBoundary);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(  double * ptrAd,
                    const int K_Omega,
                    const int K,
                    double * fd,
                    const long * ptrTriangles,
                    const long * ptrLabelTriangles,
                    const double * ptrVerts,
                     // Number of Triangles and number of Triangles in Omega
                    const int J, const int J_Omega,
                    // Number of vertices (in case of CG = K and K_Omega)
                    const int L, const int L_Omega,
                    const double * Px, const int nPx, const double * dx,
                    const double * Py, const int nPy, const double * dy,
                    const double sqdelta,
                    const long * ptrNeighbours,
                    const int is_DiscontinuousGalerkin,
                    const int is_NeumannBoundary,
                    const int dim
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