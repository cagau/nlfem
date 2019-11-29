#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(const double * x_center, const double * TE, const double sqdelta, double * out_reTriangle_list, const int is_placePointOnCap);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(  double * Ad,
                    const int K_Omega,
                    const int K,
                    double * fd,
                    const long * Triangles,
                    const double * Verts,
                    // Number of Triangles and number of Triangles in Omega
                    const int J, const int J_Omega,
                    // Number of vertices (in case of CG = K and K_Omega)
                    const int L, const int L_Omega,
                    const double * Px, const int nPx, const double * dx,
                    const double * Py, const int nPy, const double * dy,
                    const double sqdelta,
                    const long * Neighbours,
                    const int is_DiscontinuousGalerkin,
                    const int is_NeumannBoundary
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