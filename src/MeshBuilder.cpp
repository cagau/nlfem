//
// Created by klar on 19.12.19.
//

#include "MeshBuilder.h"
#include "iostream"
#include "armadillo"
using namespace std;

static int factorial(const int n){
    int i=0, fac=1;
    if (n==0){
        return fac;
    }
    for (i=1; i<=n; i++){
        fac *= i;
    }
    return fac;
}

class Grid{
public:
    Grid(int dim, int N, double delta):N(N), delta(delta), dim(dim){
        cout << "Hello Class" << endl;

    }

private:
    const int N;
    const double delta;
    const int dim;
    const arma::vec baseGrid = arma::linspace(-delta,1+delta, N);
    arma::Col<long> baseLine = arma::Col<long> {0, 1};
    arma::Mat<long> baseTriangle =
            arma::Mat<long> {
                    { 0,        0 },
                    { 1,        1 + N },
                    { 1 + N,    N }
                };
    // The algorithm to find the tetrahedons in a cube is the same as in the case of the retriangulations.
    arma::Mat<long> baseTetrahedon =
            arma::Mat<long> {
                    { 0,            0,            0,           0,           0,           0 },
                    { 1,            1+ N,         1 + N + N*N, 1,           1 + N*N,     1 + N + N*N },
                    { 1 + N,        1 + N + N*N,  N,           1 + N*N,     1 + N + N*N, N*N },
                    { 1 + N + N*N,  N,            N*N + N,     1 + N + N*N, N*N,         N*N + N }
                };

    arma::vec meshGrid(const int i) const;
    arma::vec meshGrid(const int i, const int j) const;
    arma::vec meshGrid(const int i, const int j, const int k) const;
    arma::Col<long> vertexIndex(int Vdx) const;

public:
    const int nV = pow(N, dim); // Number of Verticies
    const int nE = pow((N-1), dim) * factorial(dim); // Number of Elements

    arma::vec Vertex(const int Vdx) const;
    arma::Col<long> Line(const int i) const;
    arma::Col<long> Triangle(const int i) const;
    arma::Col<long> Tetrahedon(const int i) const;
};

arma::Col<long> Grid::vertexIndex(int Vdx) const {
    arma::Col<long> vIndex(3);
    // Think in Column Major order.
    // i.e. The indices jump first in Row then Col then Tube.
    // 3D Case - extract Tubes
    vIndex(2) = Vdx / (int) pow(N,2);
    Vdx = Vdx % (int) pow(N,2);
    // 2D Case - extract Cols
    vIndex(1) = Vdx / N;
    Vdx = Vdx % N;
    // 1D Case - extract Row
    vIndex(0) = Vdx;

    return vIndex;
}
arma::vec Grid::meshGrid(const int i) const {
    arma::vec vertexi(1);
    vertexi(0) = baseGrid(i);
    return vertexi;
}
arma::vec Grid::meshGrid(const int i, const int j)  const {
    arma::vec vertexij(2);
    vertexij(0) = baseGrid(i);
    vertexij(1) = baseGrid(j);
    return vertexij;
}
arma::vec Grid::meshGrid(const int i, const int j, const int k)  const {
    arma::vec vertexijk(3);
    vertexijk(0) = baseGrid(i);
    vertexijk(1) = baseGrid(j);
    vertexijk(2) = baseGrid(k);
    return vertexijk;
}

arma::vec Grid::Vertex(int Vdx) const {
    arma::Col<long> vIndex(3);

    if(Vdx > nV){
        cout << "Error in Grid::Vertex, Vertex does not exist." << endl;
        abort();
    }
    vIndex = vertexIndex(Vdx);
    if (dim==1){
        return meshGrid(vIndex(0));
    }
    if (dim==2){
        return meshGrid(vIndex(0), vIndex(1));
    }
    if (dim==3){
        return meshGrid(vIndex(0), vIndex(1), vIndex(2));
    }
    else {
        cout << "Error in Grid::Vertex, Wrong dimension." << endl;
        abort();
    }
}

arma::Col<long> Grid::Line(const int Ldx) const {
    if (Ldx >= nE){
        cout << "Error in Grid::Line, Element does not exist." << endl;
        abort();
    }
    return baseLine + Ldx;
}

arma::Col<long> Grid::Triangle(int Tdx) const {
    arma::Col <long> triangleIndex(3);
    long cornerVdx=0;
    long trNumber; // 0, 1

    if (Tdx >= nE){
        cout << "Error in Grid::Triangle, Element does not exist." << endl;
        abort();
    }
    triangleIndex(2) = Tdx / (2*(N-1));
    Tdx = Tdx % (2*(N-1));
    triangleIndex(1) = Tdx / 2;
    Tdx = Tdx % 2;
    triangleIndex(0) = Tdx;

    // Corner Vertex = triangleIndex(1), triangleIndex(2)
    cornerVdx = triangleIndex(2)*N + triangleIndex(1);
    trNumber = triangleIndex(0);
    arma::Col<long> Vdx = baseTriangle.col(trNumber);
    Vdx += cornerVdx;
    return  Vdx;
}

arma::Col<long> Grid::Tetrahedon(int Tdx) const {
    arma::Col <long> tetrahedonIndex(4);
    long cornerVdx=0;
    long tetNumber; // 0 to 5

    if (Tdx >= nE){
        cout << "Error in Grid::Triangle, Element does not exist." << endl;
        abort();
    }
    tetrahedonIndex(3) = Tdx / (6*(N-1)*(N-1));
    Tdx = Tdx % (6*(N-1)*(N-1));
    tetrahedonIndex(2) = Tdx / (6*(N-1));
    Tdx = Tdx % (6*(N-1));
    tetrahedonIndex(1) = Tdx / 6;
    Tdx = Tdx % 6;
    tetrahedonIndex(0) = Tdx;

    // Corner Vertex = triangleIndex(1), triangleIndex(2)
    cornerVdx = tetrahedonIndex(3)*N*N + tetrahedonIndex(2)*N + tetrahedonIndex(1);
    tetNumber = tetrahedonIndex(0);
    arma::Col<long> Vdx = baseTetrahedon.col(tetNumber);
    Vdx += cornerVdx;
    return  Vdx;
}

int  main() {
    int i=0, j=0, k=0;
    const int N = 3;
    const double delta = .1;


    Grid D1(1, N, delta);
    Grid D2(2, N, delta);
    Grid D3(3, N, delta);

    // Test Lines
    cout << endl << "Lines" << endl;
    cout << D1.Line(1)(0) << ", " << D1.Line(1)(1) << endl;
    cout << endl << "Triangles" << endl;
    k  = pow(N-1, 2)*factorial(2);
    for (i=0; i<k; i++){
        cout << D2.Triangle(i)(0) << ", " << D2.Triangle(i)(1) << ", " << D2.Triangle(i)(2) << endl;
    }
    cout << endl << "Tetrahedon" << endl;
    k  = pow(N-1, 3)*factorial(3);
    for (i=0; i<k; i++){
        cout << D3.Tetrahedon(i)(0) << ", " << D3.Tetrahedon(i)(1) << ", "
             << D3.Tetrahedon(i)(2) << ", " << D3.Tetrahedon(i)(3) << endl;
    }

    // Test Grid
    cout << endl << "1D" << endl;
    for (i=0; i<N; i++){
        cout << D1.Vertex(i)(0) << endl;
    }
    cout << endl;

    cout << "2D" << endl;
    for (i=0; i<N*N; i++){
        cout << D2.Vertex(i)(0) << ", " << D2.Vertex(i)(1) << endl;
    }
    cout << endl;

    cout << "3D" << endl;
    for (i=0; i<N*N*N; i++){
        cout << D3.Vertex(i)(0) << ", " << D3.Vertex(i)(1) << ", " << D3.Vertex(i)(2) << endl;
    }
    cout << endl;

    return 0;
};