#ifndef CASSEMBLE2D_H
#define CASSEMBLE2D_H

const double pi = 3.141592653589793;
// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble2D(  double * Ad,
                    int K,
                    double * fd,
                    const long * Triangles,
                    const long * ptrLabelTriangles,
                    const double * Verts,
                    // Number of Triangles and number of Triangles in Omega
                    int J, int J_Omega,
                    // Number of vertices (in case of CG = K and K_Omega)
                    int L, int L_Omega,
                    double * Px, int nPx, double * dx,
                    double * Py, int nPy, double * dy,
                    double sqdelta,
                    const long * Neighbours,
                    int is_DiscontinuousGalerkin,
                    int is_NeumannBoundary);

void par_assembleMass2D(double *, long *, double *, int, int, int, double *, double *);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass2D(double * vd, double * ud, const long * Triangles, const double * Verts, int K_Omega, int J_Omega, int nP, double * P, double * dx);

#endif /* Cassemble2D.h */