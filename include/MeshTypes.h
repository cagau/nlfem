//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_MESHTYPES_H
#define NONLOCAL_ASSEMBLY_MESHTYPES_H
#include "armadillo"
#include "cstring"
#include "MeshType.h"
#include "mathhelpers.h"

using namespace std;

class ElementType {
/*!
 * Class containing all necessary data of one finite element.
 *
 *  Template of Triangle Point data
 *
 *   2D Case, a, b, c are the vertices of a triangle
 *

 *
 *   T.E is of ordered the following way:
 *   | 0 | 1 | 2 | 3 | 4 | 5 |
 *   | -- | -- | -- | -- | -- | -- |
 *   | a1 | a2 | b1 | b2 | c1 | c2 |
 *
 *   Hence, if one wants to put T.E into cloumn major order matrix it would be of shape
 *   *M(mesh.dim, mesh.dVerts)* =
 *
 *    | 0   | 1   | 2   |
 *    | --- | --- | --- |
 *    | a1  | b1  | c1  |
 *    | a2  | b2  | c2  |
 */
    public:
        const long dim = 0;
        const MeshType & mesh;

        double * E = nullptr;
        long label = -1;
        double absDetValue = 0.;
        int signDetValue = 0.;
        long Tdx=0;
        arma::vec matE;

        ElementType(const MeshType & mesh_);
        void setData(const int Tdx_);
        void setData(const long * Vdx_new);
};

struct ConfigurationStruct {
    const string path_spAd;
    const string path_fd;
    const string model_kernel;
    const string model_f;
    const string integration_method;
    const bool is_placePointOnCap;
    bool is_singularKernel=false;
};
//typedef ConfigurationStruct ConfigurationType;

struct MeshStruct{
    const int K_Omega;
    const int K;
    const long * ptrTriangles;
    const long * ptrLabelTriangles;
    const double * ptrVerts;
    // Number of Elements and number of Elements in Omega
    const int nE;
    const int nE_Omega;
    // Number of vertices (in case of CG = K and K_Omega)
    const int nV;
    const int nV_Omega;
    const double delta;
    const double sqdelta;
    const long * ptrNeighbours;
    const int nNeighbours;

    const int is_DiscontinuousGalerkin;
    const int is_NeumannBoundary;

    const int dim;
    const int outdim;
    const int dVertex;

    // Weights for Domain decomposition (optional)
    const long * ptrZeta = nullptr;
    const long nZeta; // Should be set to 0

    // Optional Argument Mesh Diameter
    const double maxDiameter; // Should be set to 0 if unused.

    const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->nV)};
    const arma::Mat<long> Neighbours{arma::Mat<long>(this->ptrNeighbours, this->nNeighbours, this->nE)};
    const arma::Mat<long> Triangles{arma::Mat<long>(this->ptrTriangles, this->dVertex, this->nE)};
    // Label of Elements inside Omega = 1
    // Label of Elements in OmegaI = 2
    const arma::Col<long> LabelTriangles{arma::Col<long>(this->ptrLabelTriangles, this->nE)};
    // Zeta is an optional parameter. In case we get a Zeta matrix,
    // the memory is already allocated we only need a wrapper.
    const arma::Mat<long> ZetaIndicator{arma::Mat<long>(this-> ptrZeta, this-> nZeta, this-> nE)};
};
//typedef MeshStruct MeshType;

struct QuadratureStruct{
    const double * Px;
    const double * Py;
    const double * dx;
    const double * dy;

    const int nPx;
    const int nPy;
    const int dim;

    const double * Pg;
    const double * dg;
    const int tensorGaussDegree;
    const int nPg = pow(tensorGaussDegree, dim * 2);

    //const interactionMethodType interactionMethod;
    arma::Mat<double> psix{arma::Mat<double>(this->dim +1, this->nPx)};
    arma::Mat<double> psiy{arma::Mat<double>(this->dim +1, this->nPy)};
};
//typedef QuadratureStruct QuadratureType;

struct entryStruct{
    unsigned long dx;
    double value;

    bool operator<(const entryStruct &other) const{
        return this->dx < other.dx;
    }
    bool operator>(const entryStruct &other) const{
        return this->dx > other.dx;
    }
    bool operator==(const entryStruct &other) const{
        return this->dx == other.dx;
    }
};
typedef entryStruct entryType;

#endif //NONLOCAL_ASSEMBLY_MESHTYPES_H
