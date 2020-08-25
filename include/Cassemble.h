#ifndef CASSEMBLE_H
#define CASSEMBLE_H
#include <armadillo>
#include "MeshTypes.h"
#include "cstring"
using namespace std;

////// Retriangulation Routine ---------------------------------------------------------------------------------------------
int method_retriangulate(const double * xCenter, const double * TE,
                         double sqdelta, double * reTriangleList,
                         int isPlacePointOnCap);
void toRef(const double * E, const double * phys_x, double * ref_p);     // Pull point to Reference Element (performs 2x2 Solve)
void toPhys(const double * E, const double * p, double * out_x);         // Push point to Physical Element
void toPhys(const double * E, const double * p, int dim, double * out_x);
void solve2x2(const double * A, const double * b, double * x);

////// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
/*!
 * @brief Parallel assembly of nonlocal operator using a finite element approach.
 *
 * @details Any 2-dimensional array is handed over by a pointer only. The expected shape of the input arrays is given
 * in the parameter list where the underlying data is expected to be in C-contiguous
 * (i.e. row-major) order. We denote the dimension of the domain by d.
 *
 * @param compute   string "system", "forcing", "systemforcing". This string determines whether the stiffnes matrix,
 * load or both should be computed.
 * @param path_spAd Save path for sparse matrix (armadillo binary)
 * @param path_fd   Save path for forcing (armadillo binary)
 * @param K_Omega   Number of rows of the stiffness matrix A. Example: If you want to solve a scalar equation using
 * continuous Galerkin basis functions then K_Omega is equal to the number of basisfunctions (nV_Omega) which are not part of the
 * Dirichlet boundary. If your problem is not scalar as for example in peridynamics then K_Omega = nV_Omega*outdim.
 * @param K Number of Columns of the stiffness matrix A
 * @param ptrTriangles <B>(nE, d+1)</B> Pointer to elements. Label 1 for elements in Omega, Label 2 for elements in OmegaI.
 * @param ptrLabelTriangles <B>(nE,)</B> Pointer to element labels
 * @param ptrVerts <B>(L, d)</B> Pointer to vertices
 * @param nE         Number of elements
 * @param nE_Omega   Number of elements in Omega
 * @param nV         Number of vertices
 * @param nV_Omega   Number of vertices in Omega
 * @param Px        <B>(nPx, d)</B> Pointer to quadrature points for the outer integral
 * @param nPx       Number of quadrature points in the outer integral
 * @param dx        <B>(nPx,)</B> Pointer to quadrature weights of the outer integral
 * @param Py        Number of quadrature points in the inner integral
 * @param nPy       <B>(nPy, d)</B> Pointer to quadrature points for the inner integral
 * @param dy        <B>(nPx,)</B> Pointer to quadrature weights of the inner integral
 * @param sqdelta   Squared delta
 * @param ptrNeighbours <B>(L, d+1)</B> Adjaciency graph of triangulation.
 * @param nNeighbours   number of columns in Neighbours List.
 * @param is_DiscontinuousGalerkin Switch for discontinuous Galerkin
 * @param is_NeumannBoundary Switch of Neumann Boundary Conditions
 * @param str_model_kernel  Name of kernel
 * @param str_model_f   Name of right hand side
 * @param str_integration_method Name of integration method
 * @param is_PlacePointOnCap Switch for withcaps parameter in retriangulation
 * @param dim       Dimension of the domain
 * @param outdim    Dimension of the solution space (e.g. 1 for scalar problems, dim for linear elasticity)
 * @param ptrZeta   Pointer to overlap counter of decomposed Mesh (optional)
 * @param nZeta     Number of rows of Zeta
 * @param Pg        <B>(nPg, dim^2)</B> Quadrature points for tensor Gauss quadrature (optional, needed for singular kernels).
 * @param nPg       <B>(tensorGaussDegree^dim,)</B> Number of quadrature points.
 * @param dg        <B>(nPg,)</B> Weights for tensor Gauss quadrature.
 */
void par_assemble(string compute, string path_spAd, string path_fd, int K_Omega, int K,
                  const long *ptrTriangles, const long *ptrLabelTriangles, const double *ptrVerts, int nE,
                  int nE_Omega, int nV, int nV_Omega, const double *Px, int nPx, const double *dx,
                  const double *Py, int nPy, const double *dy, double sqdelta, const long *ptrNeighbours,
                  int nNeighbours,
                  int is_DiscontinuousGalerkin, int is_NeumannBoundary, string str_model_kernel,
                  string str_model_f, string str_integration_method, int is_PlacePointOnCap,
                  int dim, int outdim, const long * ptrZeta = nullptr, long nZeta = 0,
                  const double * Pg = nullptr, int degree = 0, const double * dg = nullptr);

void par_system(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf);
void par_forcing(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
/**
 * @brief Evaluate the mass matrix v = Mu.
 *
 * @param vd        Pointer to the first entry of the output vector.
 * @param ud        Pointer to the first entry of the input vector.
 * @param Elements  List of elements of a finite element triangulation (CSR-format, row major order).
 * @param ElementLabels List of element Labels.
 * @param Verts     List of vertices (row major order).
 * @param K_Omega   Number of rows and columns in M. Example: If you use continuous Galerkin basis functions and
 * want to solve a scalar problem K_Omega = J.
 * @param J         Number of elements in the triangulation.
 * @param nP        Number of quadrature points in the outer integral
 * @param P         <B>(nPx, d)</B> Pointer to quadrature points.
 * @param dx        <B>(nPx,)</B> Pointer to quadrature weights.
 * @param dim       Dimension of the domain Omega (2 or 3).
 */
void par_evaluateMass(double *vd, double *ud, long *Elements, long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
                 double *P, double *dx, int dim);
//[DEBUG]


#endif /* Cassemble.h */