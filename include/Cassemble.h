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

////// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
/*!
 * @brief Parallel assembly of nonlocal operator using a finite element approach.
 *
 * @details Any 2-dimensional array is handed over by a pointer only. The expected shape of the input arrays is given
 * in the parameter list where the underlying data is expected to be in C-contiguous
 * (i.e. row-major) order. We denote the dimension of the domain by d.
 *
 * @param path_spAd <B>(K_Omega, K)</B> Pointer to empty double array to store the resulting stiffness matrix
 * @param K_Omega Number of rows of Ad
 * @param K Number of Columns of Ad
 * @param fd <B>(K_Omega,)</B> Pointer to empty double array to store resulting right hand side
 * @param ptrTriangles <B>(J, d+1)</B> Pointer to elements
 * @param ptrLabelTriangles <B>(J,)</B> Pointer to element labels
 * @param ptrVerts <B>(L, d)</B> Pointer to vertices
 * @param J         Number of elements
 * @param J_Omega   Number of elements in Omega
 * @param L         Number of vertices
 * @param L_        Omega Number of vertices in Omega
 * @param Px        <B>(nPx, d)</B> Pointer to quadrature points for the outer integral
 * @param nPx
 * @param dx        <B>(nPx,)</B> Pointer to quadrature weights of the outer integral
 * @param Py
 * @param nPy       <B>(nPy, d)</B> Pointer to quadrature points for the inner integral
 * @param dy        <B>(nPx,)</B> Pointer to quadrature weights of the inner integral
 * @param sqdelta   Squared delta
 * @param ptrNeighbours <B>(L, d+1)</B> Adjaciency graph of triangulation.
 * @param is_DiscontinuousGalerkin Switch for discontinuous Gakerkin
 * @param is_NeumannBoundary Switch of Neumann Boundary Conditions
 * @param str_model_kernel  Name of kernel
 * @param str_model_f   Name of right hand side
 * @param str_integration_method Name of integration method
 * @param is_PlacePointOnCap Switch for withcaps parameter in retriangulation
 * @param dim Dimension of the domain
 */
void par_assemble(const string path_spAd, const int K_Omega, const int K, double *fd, const long *ptrTriangles,
                  const long *ptrLabelTriangles, const double *ptrVerts, const int J, const int J_Omega, const int L,
                  const int L_Omega, const double *Px, const int nPx, const double *dx, const double *Py, const int nPy,
                  const double *dy, const double sqdelta, const long *ptrNeighbours, const int is_DiscontinuousGalerkin,
                  const int is_NeumannBoundary, const string str_model_kernel, const string str_model_f,
                  const string str_integration_method, const int is_PlacePointOnCap, const int dim);

void par_assemble(double *fd, MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *vd, double *ud, long *Elements, long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
                 double *P, double *dx, const int dim);
void constructAdjaciencyGraph(const int dim, const int nE, const long * elements, long * neighbours);
//[DEBUG]


#endif /* Cassemble.h */