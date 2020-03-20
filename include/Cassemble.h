#ifndef CASSEMBLE_H
#define CASSEMBLE_H
#include <armadillo>
#include "MeshTypes.h"
#include "cstring"
using namespace std;

// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(const double * x_center, const double * TE, const MeshType & mesh, double * re_Triangle_list, int is_placePointOnCap);

void par_assemble( double * ptrAd, double * fd, MeshType & mesh, QuadratureType & quadRule, ConfigurationType & conf);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(double *ptrAd, int K_Omega, int K, double *fd, const long *ptrTriangles,
                  const long *ptrLabelTriangles, const double *ptrVerts, int J, int J_Omega, int L,
                  int L_Omega, const double *Px, int nPx, const double *dx, const double *Py, int nPy,
                  const double *dy, double sqdelta, const long *ptrNeighbours, int is_DiscontinuousGalerkin,
                  int is_NeumannBoundary, string str_model_kernel, string str_model_f,
                  string str_integration_method, int is_PlacePointOnCap, int dim);
void par_assembleMass(double * Ad, long * Triangles, double * Verts, int K_Omega, int J_Omega, int nP, double * P, double * dx);
void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *);
double compute_area(double *, double, long, double *, double, long, double *, int, double *, double);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void
par_evaluateMass(double *vd, double *ud, long *Triangles, long *TriangleLabels, double *Verts, int K_Omega, int J,
                 int nP, double *P, double *dx);

//[DEBUG]
//void relativePosition(double *, double *, double *, double *, double *);
//void order(double *, int, double *);

#endif /* Cassemble.h */