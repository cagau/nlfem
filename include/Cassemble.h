#ifndef CASSEMBLE_H
#define CASSEMBLE_H
#include <armadillo>
#include "MeshTypes.h"
#include "cstring"
using namespace std;

////// Retriangulation Routine ---------------------------------------------------------------------------------------------
int method_retriangulate(const double * x_center, const double * TE,
                         double sqdelta, double * re_Triangle_list,
                         int is_placePointOnCap);
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
 * @param compute   string "system", "forcing", "systemforcing"
 * @param path_spAd Save path for sparse matrix (armadillo binary)
 * @param path_fd   Save path for forcing (armadillo binary)
 * @param K_Omega Number of rows of the stiffness matrix A
 * @param K Number of Columns of the stiffness matrix A
 * @param ptrTriangles <B>(J, d+1)</B> Pointer to elements. Label 1 for elements in Omega, Label 2 for elements in OmegaI.
 * @param ptrLabelTriangles <B>(J,)</B> Pointer to element labels
 * @param ptrVerts <B>(L, d)</B> Pointer to vertices
 * @param J         Number of elements
 * @param J_Omega   Number of elements in Omega
 * @param L         Number of vertices
 * @param L_Omega   Number of vertices in Omega
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
 * @param dim Dimension of the domain
 * @param outdim Dimension of the solution space (e.g. 1 for scalar problems, dim for linear elasticity)
 * @param ptrZeta Pointer to overlap counter of decomposed Mesh (optional)
 * @param nZeta Number of rows of Zeta
 * @param Pg        <B>(nPg, dim^2)</B> Quadrature points for tensor Gauss quadrature (optional, needed for singular kernels).
 * @param nPg       <B>(tensorGaussDegree^dim,)</B> Number of quadrature points.
 * @param dg        <B>(nPg,)</B> Weights for tensor Gauss quadrature.
 */
void par_assemble(const string compute, const string path_spAd, const string path_fd, const int K_Omega, const int K,
                  const long *ptrTriangles, const long *ptrLabelTriangles, const double *ptrVerts, const int J,
                  const int J_Omega, const int L, const int L_Omega, const double *Px, const int nPx, const double *dx,
                  const double *Py, const int nPy, const double *dy, const double sqdelta, const long *ptrNeighbours,
                  const int nNeighbours,
                  const int is_DiscontinuousGalerkin, const int is_NeumannBoundary, const string str_model_kernel,
                  const string str_model_f, const string str_integration_method, const int is_PlacePointOnCap,
                  const int dim, const int outdim, const long * ptrZeta = nullptr, const long nZeta = 0,
                  const double * Pg = nullptr, const int degree = 0, const double * dg = nullptr);
void par_system(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf);
void par_forcing(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *vd, double *ud, long *Elements, long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
                 double *P, double *dx, const int dim);
//[DEBUG]


#endif /* Cassemble.h */