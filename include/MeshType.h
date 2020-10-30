//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_MESHTYPE_H
#define NONLOCAL_ASSEMBLY_MESHTYPE_H

#include "armadillo"
#include "iostream"
using namespace std;

class MeshType {
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
        // Number of Elements and number of Elements in Omega
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

        // Deprecated Arguments
        long outdim=0;
        long nZeta = 0;
        long * ptrZeta = nullptr;
        bool is_DiscontinuousGalerkin = false;
        const long nNeighbours;
        const arma::Mat<long> Neighbours{arma::Mat<long>(this->ptrNeighborIndices, this->nNeighbours, this->nE)};
        arma::Mat<long> ZetaIndicator;
        long K, K_Omega;
        double sqdelta=0;
        double delta=0;

        const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->nV)};
        const arma::Mat<long> Elements{arma::Mat<long>(this->ptrElements, this->dVertex, this->nE)};
        const arma::Col<long> LabelElements{arma::Col<long>(this->ptrElementLabels, this->nE)};
        MeshType(const long * ptrElements_,
                 const long * ptrElementLabels_,
                 const double * ptrVerts_,
                 const long nE_,
                 const long nE_Omega_,
                 const int nV_,
                 const int nV_Omega_,
                 const long * ptrNeighborIndices_,
                 const long * ptrNeighborIndexPtr_,
                 const long dim_,
                 const double maxDiameter_,
                 const long outdim_,
                 const long nNeighbours_
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
             maxDiameter(maxDiameter_),
             outdim(outdim_),
             nNeighbours(nNeighbours_)
             {
                cout << "Mesh setting sqdelta to 0.001" << endl;
                K = nV;
                K_Omega = nV_Omega;
                sqdelta = 0.01;
                delta = 0.1;
             };
};


#endif //NONLOCAL_ASSEMBLY_MESHTYPE_H
