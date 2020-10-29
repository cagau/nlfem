//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_MESH_H
#define NONLOCAL_ASSEMBLY_MESH_H

#include "armadillo"
#include "iostream"
using namespace std;

class Mesh {
    /**
     * @brief Mesh class.
     *
     * Each element, and each vertex are equipped with a label. Negaive labels are
     * interpreted as elements and vertices belonging to the non-local Dirichlet boundary.
     */
public:
        const long * ptrElements;
        const long * ptrElementLabels;
        const double * ptrVerts;
        // Number of Triangles and number of Triangles in Omega
        const long nE; // Required ?? -> Depending on the format.
        const long nE_Omega; // Required??
        // Number of vertices (in case of CG = K and K_Omega)
        const long nV; // Required??
        const long nV_Omega; // Required??

        const long * ptrNeighborIndices;
        const long * ptrNeighborIndexPtr;

        const long dim;
        const long dVertex;

        // Optional Argument Mesh Diameter
        const double maxDiameter; // Should be set to 0 if unused.

        const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->nV)};
        const arma::Mat<long> Elements{arma::Mat<long>(this->ptrElements, this->dVertex, this->nE)};
        const arma::Col<long> LabelElements{arma::Col<long>(this->ptrElementLabels, this->nE)};
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
             const double maxDiameter_=0
             ):
             ptrElements(ptrElements_),
             ptrElementLabels(ptrElementLabels_),
             ptrVerts(ptrVerts_),
             nE(nE_),
             nE_Omega(nE_Omega_),
             nV(nV_),
             nV_Omega(nV_Omega_),
             ptrNeighborIndices(ptrNeighborIndices_),
             ptrNeighborIndexPtr(ptrNeighborIndexPtr_),
             dim(dim_),
             dVertex(dim_+1),
             maxDiameter(maxDiameter_)
             { };
};


#endif //NONLOCAL_ASSEMBLY_MESH_H
