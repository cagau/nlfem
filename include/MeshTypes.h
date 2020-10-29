//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_MESHTYPES_H
#define NONLOCAL_ASSEMBLY_MESHTYPES_H
#include "armadillo"
#include "cstring"
using namespace std;

struct ElementStruct
/*!
 * Struct containing all necessary data of one finite element.
 *
 *  Template of Triangle Point data
 *
 *   2D Case, a, b, c are the vertices of a triangle
 *
*/
{
    /*!
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
    arma::vec matE;
    double * E;
    int dim;
    long label;
    double absDet;
    int signDet;
    long Tdx=0;
};
typedef ElementStruct ElementType;

class ElementClass {
    public:
        double * E = nullptr;
        int dim = 0;
        long label = -1;
        double absDet = 0.;
        int signDet = 0.;
        long Tdx=0;
        arma::vec matE;

        ElementClass(){
        };
        ElementClass(int dim_):
                dim(dim_){
            matE = arma::vec(this->dim*(dim+1), arma::fill::zeros);
            E = matE.memptr();
        };
        ~ElementClass () {};
};
int getElement(ElementClass &element);

struct ConfigurationStruct {
    const string path_spAd;
    const string path_fd;
    const string model_kernel;
    const string model_f;
    const string integration_method;
    const bool is_placePointOnCap;
    bool is_singularKernel=false;
};
typedef ConfigurationStruct ConfigurationType;

struct MeshStruct{
    const int K_Omega;
    const int K;
    const long * ptrTriangles;
    const long * ptrLabelTriangles;
    const double * ptrVerts;
    // Number of Triangles and number of Triangles in Omega
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
    // Label of Triangles inside Omega = 1
    // Label of Triangles in OmegaI = 2
    const arma::Col<long> LabelTriangles{arma::Col<long>(this->ptrLabelTriangles, this->nE)};
    // Zeta is an optional parameter. In case we get a Zeta matrix,
    // the memory is already allocated we only need a wrapper.
    const arma::Mat<long> ZetaIndicator{arma::Mat<long>(this-> ptrZeta, this-> nZeta, this-> nE)};
};
typedef MeshStruct MeshType;

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
typedef QuadratureStruct QuadratureType;

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
