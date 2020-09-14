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

class sparseMatrix{
public:
    entryStruct * indexValuePairs = nullptr;
    entryStruct * indexValuePairs_source = nullptr;
    unsigned long reserved_total = 0;
    const unsigned long reserved_buffer;
    unsigned long size_guess;
    const MeshType mesh;

    entryStruct * A = nullptr;
    entryStruct * buffer_A = nullptr;
    unsigned long n_entries=0, n_buffer=0;

    sparseMatrix(unsigned long chunkSize, MeshType & mesh) :
    reserved_buffer(2),//10 * static_cast<unsigned long>(pow(mesh.dVertex*mesh.outdim,2))),
    size_guess(chunkSize),
    mesh(mesh)
    {
        unsigned long estimatedNNZ = size_guess *
                                     static_cast<unsigned long>(pow(2*ceil(mesh.delta / mesh.maxDiameter + 1)*
                                                                    mesh.outdim, mesh.dim));
        //estimatedNNZ = size_guess;
        reserved_total = reserved_buffer + estimatedNNZ;
        cout << "Now reserved " << reserved_total << endl;
        indexValuePairs = static_cast<entryType *>(malloc(sizeof(entryType) * reserved_total));
        A = indexValuePairs;
        buffer_A = indexValuePairs+n_entries;

        cout << "Allocated!" << endl;
    }

    ~sparseMatrix() = default;
    // Putting this yields munmapchunk error. Hence I guess indexValuePairs is freed automatically.
    // free(indexValuePairs);

    int append(entryStruct & entry);
    int mergeBuffer();

    static unsigned long reduce(entryStruct * mat, unsigned long length);
};

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
