//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_MESHTYPES_H
#define NONLOCAL_ASSEMBLY_MESHTYPES_H
#include "armadillo"
#include "cstring"
using namespace std;

struct ElementStruct{
    arma::vec matE;
    double * E;
    int dim;
    long label;
    double absDet;
    int signDet;
};
typedef ElementStruct ElementType;

struct ConfigurationStruct{
    const string model_kernel;
    const string  model_f;
    const string  integration_method;
    const bool is_placePointOnCap;
};
typedef ConfigurationStruct ConfigurationType;

struct MeshStruct{
    const int K_Omega;
    const int K;
    const long * ptrTriangles;
    const long * ptrLabelTriangles;
    const double * ptrVerts;
    // Number of Triangles and number of Triangles in Omega
    const int J;
    const int J_Omega;
    // Number of vertices (in case of CG = K and K_Omega)
    const int L;
    const int L_Omega;
    const double sqdelta;
    const long * ptrNeighbours;
    const int is_DiscontinuousGalerkin;
    const int is_NeumannBoundary;

    const int dim;
    const int dVertex;

    const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->L)};
    const arma::Mat<long> Neighbours{arma::Mat<long>(this->ptrNeighbours, this->dVertex, this->J)};
    const arma::Mat<long> Triangles{arma::Mat<long>(this->ptrTriangles, this->dVertex, this->J)};
    const arma::Col<long> LabelTriangles{arma::Col<long>(this->ptrLabelTriangles, this->J)};
};
typedef MeshStruct MeshType;
//typedef int (*const interactionMethodType)(const double * x_center, const ElementType & T,
//                                       const MeshType & mesh, double * out_reTriangle_list);

struct QuadratureStruct{
    const double * Px;
    const double * Py;
    const double * dx;
    const double * dy;
    const int nPx;
    const int nPy;
    const int dim;
    //const interactionMethodType interactionMethod;
    arma::Mat<double> psix{arma::Mat<double>(this->dim +1, this->nPx)};
    arma::Mat<double> psiy{arma::Mat<double>(this->dim +1, this->nPy)};
};
typedef QuadratureStruct QuadratureType;

#endif //NONLOCAL_ASSEMBLY_MESHTYPES_H
