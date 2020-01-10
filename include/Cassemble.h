#ifndef CASSEMBLE_H
#define CASSEMBLE_H
#include <armadillo>

struct ElementStruct{
    arma::vec matE;
    double * E;
    int dim;
    long label;
    double absDet;
    int signDet;
};
typedef ElementStruct ElementType;

struct MeshStruct{
    const int K_Omega;
    const int K;
    const long * ptrTriangles;
    const long * ptrLabelTriangles;
    const double * ptrVerts;
    // Number of Triangles and number of Triangles in Omega
    const int J;
    const int J_Omega;
    // Number of vertices (in case of CG = K and K_Omega)
    const int L;
    const int L_Omega;
    const double sqdelta;
    const long * ptrNeighbours;
    const int is_DiscontinuousGalerkin;
    const int is_NeumannBoundary;

    const int dim;
    const int dVertex;

    const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->L)};
    const arma::Mat<long> Neighbours{arma::Mat<long>(this->ptrNeighbours, this->dVertex, this->J)};
    const arma::Mat<long> Triangles{arma::Mat<long>(this->ptrTriangles, this->dVertex, this->J)};
    const arma::Col<long> LabelTriangles{arma::Col<long>(this->ptrLabelTriangles, this->J)};
};
typedef MeshStruct MeshType;
//typedef int (*const interactionMethodType)(const double * x_center, const ElementType & T,
//                                       const MeshType & mesh, double * out_reTriangle_list);

struct QuadratureStruct{
    const double * Px;
    const double * Py;
    const double * dx;
    const double * dy;
    const int nPx;
    const int nPy;
    const int dim;
    //const interactionMethodType interactionMethod;
    arma::Mat<double> psix{arma::Mat<double>(this->dim +1, this->nPx)};
    arma::Mat<double> psiy{arma::Mat<double>(this->dim +1, this->nPy)};
};
typedef QuadratureStruct QuadratureType;
// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(const double * x_center, const double * TE, const MeshType & mesh, double * out_reTriangle_list, const int is_placePointOnCap);

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