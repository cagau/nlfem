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
    const int J;
    const int J_Omega;
    // Number of vertices (in case of CG = K and K_Omega)
    const int L;
    const int L_Omega;
    const double sqdelta;
    const long * ptrNeighbours;
    const int nNeighbours;

    const int is_DiscontinuousGalerkin;
    const int is_NeumannBoundary;

    const int dim;
    const int outdim;
    const int dVertex;

    // Weights for Domain decomposition
    const long * ptrZeta = nullptr;
    const long nZeta;

    // Zeta is an optional parameter. In case we get a Zeta matrix,
    // the memory is already allocated we only need a wrapper.
    // Caution: map[k]. If k does not match the key of any element in the container,
    // the function inserts a new element with that key and
    // returns a reference to its mapped value.
    // >> This eats up memory unnecessarily if you want to read only!
    // Note: Armadillo uvec.find() is much slower.
    map<long, const long *> Zeta;
    const arma::Mat<double> Verts{arma::Mat<double>(this->ptrVerts, this->dim, this->L)};
    const arma::Mat<long> Neighbours{arma::Mat<long>(this->ptrNeighbours, this->nNeighbours, this->J)};
    const arma::Mat<long> Triangles{arma::Mat<long>(this->ptrTriangles, this->dVertex, this->J)};
    // Label of Triangles inside Omega = 1
    // Label of Triangles in OmegaI = 2
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

    const double * Pg;
    const double * dg;
    const int tensorGaussDegree;
    const int nPg = pow(tensorGaussDegree, dim * 2);

    //const interactionMethodType interactionMethod;
    arma::Mat<double> psix{arma::Mat<double>(this->dim +1, this->nPx)};
    arma::Mat<double> psiy{arma::Mat<double>(this->dim +1, this->nPy)};
};
typedef QuadratureStruct QuadratureType;

//class sp_index {
//public:
    //int i;
    //int j;
    // Needed for (ordered) std::map
    /*bool operator<(const sp_index &other) const {
        if (i < other.i) return true;
        if (other.i < i) return false;
        return (j < other.j);
    }*/
    // Needed for std::unoredered_map
    /*bool operator==(const sp_index &other) const {
        return (i == other.i) && (j == other.j);
    }*/
//};
// Hashing function for unordered map
// https://en.cppreference.com/w/cpp/utility/hash
// https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key
/*namespace std {
    template <>
    struct hash<sp_index>
    {
        std::size_t operator()(const sp_index& dx) const
        {
            using std::size_t;
            using std::hash;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:

            return ((hash<int>()(dx.i) ^ (hash<int>()(dx.j) << 1)) >> 1);
        }
    };

}

struct sp_reduce {
    int threadID;
    int start;
    int end;
};
typedef sp_reduce sp_reduceType;
*/
#endif //NONLOCAL_ASSEMBLY_MESHTYPES_H
