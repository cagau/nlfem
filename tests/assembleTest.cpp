//
// Created by klar on 18.12.19.
//

#include <Cassemble.h>
#include <MeshBuilder.h>
#include "armadillo"
#include "gtest/gtest.h"

using namespace std;

TEST(assembleTest, gridTest) {
    int i=0;
    const double delta = .1;

    Grid1D D1(3, delta);
    Grid2D D2(3, delta);
    Grid3D D3(3, delta);

    // Test Grid
    // 1D
    arma::vec d1 = {-0.1, 0, 0.5, 1, 1.1};
    arma::vec grid1 = {D1.Vertex(0)(0), D1.Vertex(1)(0), D1.Vertex(2)(0),
                       D1.Vertex(3)(0), D1.Vertex(4)(0)};

    // 2D
    arma::mat d2 = {{-0.1, -0.1, 2},
                    {0, -0.1, 2},
                    {0.5, -0.1, 2},
                    {1, -0.1, 2},
                    {1.1, -0.1, 2},
                    {-0.1, 0, 2},
                    {0, 0, 2},
                    {0.5, 0, 2},
                    {1, 0, 2},
                    {1.1, 0, 2},
                    {-0.1, 0.5, 2},
                    {0, 0.5, 2},
                    {0.5, 0.5, 1}, // The one point with label 1.
                    {1, 0.5, 2},
                    {1.1, 0.5, 2},
                    {-0.1, 1, 2},
                    {0, 1, 2},
                    {0.5, 1, 2},
                    {1, 1, 2},
                    {1.1, 1, 2},
                    {-0.1, 1.1, 2},
                    {0, 1.1, 2},
                    {0.5, 1.1, 2},
                    {1, 1.1, 2},
                    {1.1, 1.1, 2}};
    arma::mat grid2(D2.N * D2.N ,3);

    //3D
    arma::mat d3 = {{-0.1, -0.1, -0.1, 2},
                    {0, -0.1, -0.1, 2},
                    {0.5, -0.1, -0.1, 2},
                    {1, -0.1, -0.1, 2},
                    {1.1, -0.1, -0.1, 2},
                    {-0.1, 0, -0.1, 2},
                    {0, 0, -0.1, 2},
                    {0.5, 0, -0.1, 2},
                    {1, 0, -0.1, 2},
                    {1.1, 0, -0.1, 2},
                    {-0.1, 0.5, -0.1, 2},
                    {0, 0.5, -0.1, 2},
                    {0.5, 0.5, -0.1, 2},
                    {1, 0.5, -0.1, 2},
                    {1.1, 0.5, -0.1, 2},
                    {-0.1, 1, -0.1, 2},
                    {0, 1, -0.1, 2},
                    {0.5, 1, -0.1, 2},
                    {1, 1, -0.1, 2},
                    {1.1, 1, -0.1, 2},
                    {-0.1, 1.1, -0.1, 2},
                    {0, 1.1, -0.1, 2},
                    {0.5, 1.1, -0.1, 2},
                    {1, 1.1, -0.1, 2},
                    {1.1, 1.1, -0.1, 2},
                    {-0.1, -0.1, 0, 2},
                    {0, -0.1, 0, 2},
                    {0.5, -0.1, 0, 2},
                    {1, -0.1, 0, 2},
                    {1.1, -0.1, 0, 2},
                    {-0.1, 0, 0, 2},
                    {0, 0, 0, 2},
                    {0.5, 0, 0, 2},
                    {1, 0, 0, 2},
                    {1.1, 0, 0, 2},
                    {-0.1, 0.5, 0, 2},
                    {0, 0.5, 0, 2},
                    {0.5, 0.5, 0, 2},
                    {1, 0.5, 0, 2},
                    {1.1, 0.5, 0, 2},
                    {-0.1, 1, 0, 2},
                    {0, 1, 0, 2},
                    {0.5, 1, 0, 2},
                    {1, 1, 0, 2},
                    {1.1, 1, 0, 2},
                    {-0.1, 1.1, 0, 2},
                    {0, 1.1, 0, 2},
                    {0.5, 1.1, 0, 2},
                    {1, 1.1, 0, 2},
                    {1.1, 1.1, 0, 2},
                    {-0.1, -0.1, 0.5, 2},
                    {0, -0.1, 0.5, 2},
                    {0.5, -0.1, 0.5, 2},
                    {1, -0.1, 0.5, 2},
                    {1.1, -0.1, 0.5, 2},
                    {-0.1, 0, 0.5, 2},
                    {0, 0, 0.5, 2},
                    {0.5, 0, 0.5, 2},
                    {1, 0, 0.5, 2},
                    {1.1, 0, 0.5, 2},
                    {-0.1, 0.5, 0.5, 2},
                    {0, 0.5, 0.5, 2},
                    {0.5, 0.5, 0.5, 1}, // The one point with label 1
                    {1, 0.5, 0.5, 2},
                    {1.1, 0.5, 0.5, 2},
                    {-0.1, 1, 0.5, 2},
                    {0, 1, 0.5, 2},
                    {0.5, 1, 0.5, 2},
                    {1, 1, 0.5, 2},
                    {1.1, 1, 0.5, 2},
                    {-0.1, 1.1, 0.5, 2},
                    {0, 1.1, 0.5, 2},
                    {0.5, 1.1, 0.5, 2},
                    {1, 1.1, 0.5, 2},
                    {1.1, 1.1, 0.5, 2},
                    {-0.1, -0.1, 1, 2},
                    {0, -0.1, 1, 2},
                    {0.5, -0.1, 1, 2},
                    {1, -0.1, 1, 2},
                    {1.1, -0.1, 1, 2},
                    {-0.1, 0, 1, 2},
                    {0, 0, 1, 2},
                    {0.5, 0, 1, 2},
                    {1, 0, 1, 2},
                    {1.1, 0, 1, 2},
                    {-0.1, 0.5, 1, 2},
                    {0, 0.5, 1, 2},
                    {0.5, 0.5, 1, 2},
                    {1, 0.5, 1, 2},
                    {1.1, 0.5, 1, 2},
                    {-0.1, 1, 1, 2},
                    {0, 1, 1, 2},
                    {0.5, 1, 1, 2},
                    {1, 1, 1, 2},
                    {1.1, 1, 1, 2},
                    {-0.1, 1.1, 1, 2},
                    {0, 1.1, 1, 2},
                    {0.5, 1.1, 1, 2},
                    {1, 1.1, 1, 2},
                    {1.1, 1.1, 1, 2},
                    {-0.1, -0.1, 1.1, 2},
                    {0, -0.1, 1.1, 2},
                    {0.5, -0.1, 1.1, 2},
                    {1, -0.1, 1.1, 2},
                    {1.1, -0.1, 1.1, 2},
                    {-0.1, 0, 1.1, 2},
                    {0, 0, 1.1, 2},
                    {0.5, 0, 1.1, 2},
                    {1, 0, 1.1, 2},
                    {1.1, 0, 1.1, 2},
                    {-0.1, 0.5, 1.1, 2},
                    {0, 0.5, 1.1, 2},
                    {0.5, 0.5, 1.1, 2},
                    {1, 0.5, 1.1, 2},
                    {1.1, 0.5, 1.1, 2},
                    {-0.1, 1, 1.1, 2},
                    {0, 1, 1.1, 2},
                    {0.5, 1, 1.1, 2},
                    {1, 1, 1.1, 2},
                    {1.1, 1, 1.1, 2},
                    {-0.1, 1.1, 1.1, 2},
                    {0, 1.1, 1.1, 2},
                    {0.5, 1.1, 1.1, 2},
                    {1, 1.1, 1.1, 2},
                    {1.1, 1.1, 1.1, 2}};

    arma::mat grid3(D3.N*D3.N*D3.N,4);

    cout << "2D" << endl;
    for (i = 0; i < D2.N * D2.N ; i++) {
        grid2(i, 0) = D2.Vertex(i)(0); // Point
        grid2(i, 1) = D2.Vertex(i)(1);
        grid2(i, 2) = D2.Vertex(i)(2); // Label
        //cout << "{" << D2.Vertex(i)(0) << ", " << D2.Vertex(i)(1) << ", " << D2.Vertex(i)(2) << "}," << endl;
    }
    cout << endl;

    cout << "3D" << endl;
    for (i = 0; i < D3.N * D3.N * D3.N; i++) {
        grid3(i, 0) = D3.Vertex(i)(0); // Point
        grid3(i, 1) = D3.Vertex(i)(1);
        grid3(i, 2) = D3.Vertex(i)(2);
        grid3(i, 3) = D3.Vertex(i)(3); // Label
        //cout << "{" << D3.Vertex(i)(0) << ", " << D3.Vertex(i)(1) << ", "
        //            << D3.Vertex(i)(2) << ", " << D3.Vertex(i)(3) << "}," << endl;
    }
    cout << endl;


    EXPECT_TRUE(arma::norm(d1 - grid1) <  1.0e-12);
    EXPECT_TRUE(arma::norm(arma::vectorise(d2 - grid2)) <  1.0e-12);
    EXPECT_TRUE(arma::norm(arma::vectorise(d3 - grid3)) <  1.0e-10);
}

TEST(assembleTest, elementLabelTest){
    int i=0, k;
    const double delta = .25;
    const int N=2; //N = (int)(1/delta) + 2*2;
    int K = max(floor(delta*(N-1)) + 1, 2); // For K = 1 we get a single element equal to end
    arma::Col <long> coords;

    Grid1D D1(N, delta);
    Grid2D D2(N, delta);
    Grid3D D3(N, delta);

    cout << "Joined Grid" << endl;
    for(i=0; i<D2.N; i++) {
        cout << D2.baseGrid(i) << " " ;
    }
    cout << endl;

    cout << endl << "2D Vertex Labels" << endl;
    cout << "N " << N << endl;
    for(i=0; i<D2.N; i++){
        for(k=0; k<D2.N; k++){
            cout << D2.Vertex(D2.N*i + k)(2) << "   ";
        }
        cout << endl;
    }

    cout << endl << "2D Element Labels" << endl;
    for (i = 0; i < D2.nE; i++) {
        coords = D2.Element(i);
        cout << "L " << D2.ElementLabel(i) << endl;
        cout << "x1 " << D2.Vertex(coords(0))(0) << ", " << D2.Vertex(coords(0))(1) << endl;
        cout << "x2 " << D2.Vertex(coords(1))(0) << ", " << D2.Vertex(coords(1))(1) << endl;
        cout << "x3 " << D2.Vertex(coords(2))(0) << ", " << D2.Vertex(coords(2))(1) << endl;
        //cout << D2.Vertex(i)(0) << ", " << D2.Vertex(i)(1) << endl;
    }

}

TEST(assembleTest, meshTest) {
    int i=0, k=0, N;
    const double delta = .1;

    Grid1D D1(2, delta);
    Grid2D D2(2, delta);
    Grid3D D3(2, delta);

    N = D1.N; // N is the same for all grids, but computed from N_Omega und delta.

    arma::Mat<long> testTetrahedons = {
            {0, 1, 5, 21},
            {0, 5, 21, 4},
            {0, 21, 4, 20},
            {0, 1, 17, 21},
            {0, 17, 21, 16},
            {0, 21, 16, 20},
            {1, 2, 6, 22},
            {1, 6, 22, 5},
            {1, 22, 5, 21},
            {1, 2, 18, 22},
            {1, 18, 22, 17},
            {1, 22, 17, 21},
            {2, 3, 7, 23},
            {2, 7, 23, 6},
            {2, 23, 6, 22},
            {2, 3, 19, 23},
            {2, 19, 23, 18},
            {2, 23, 18, 22},
            {4, 5, 9, 25},
            {4, 9, 25, 8},
            {4, 25, 8, 24},
            {4, 5, 21, 25},
            {4, 21, 25, 20},
            {4, 25, 20, 24},
            {5, 6, 10, 26},
            {5, 10, 26, 9},
            {5, 26, 9, 25},
            {5, 6, 22, 26},
            {5, 22, 26, 21},
            {5, 26, 21, 25},
            {6, 7, 11, 27},
            {6, 11, 27, 10},
            {6, 27, 10, 26},
            {6, 7, 23, 27},
            {6, 23, 27, 22},
            {6, 27, 22, 26},
            {8, 9, 13, 29},
            {8, 13, 29, 12},
            {8, 29, 12, 28},
            {8, 9, 25, 29},
            {8, 25, 29, 24},
            {8, 29, 24, 28},
            {9, 10, 14, 30},
            {9, 14, 30, 13},
            {9, 30, 13, 29},
            {9, 10, 26, 30},
            {9, 26, 30, 25},
            {9, 30, 25, 29},
            {10, 11, 15, 31},
            {10, 15, 31, 14},
            {10, 31, 14, 30},
            {10, 11, 27, 31},
            {10, 27, 31, 26},
            {10, 31, 26, 30},
            {16, 17, 21, 37},
            {16, 21, 37, 20},
            {16, 37, 20, 36},
            {16, 17, 33, 37},
            {16, 33, 37, 32},
            {16, 37, 32, 36},
            {17, 18, 22, 38},
            {17, 22, 38, 21},
            {17, 38, 21, 37},
            {17, 18, 34, 38},
            {17, 34, 38, 33},
            {17, 38, 33, 37},
            {18, 19, 23, 39},
            {18, 23, 39, 22},
            {18, 39, 22, 38},
            {18, 19, 35, 39},
            {18, 35, 39, 34},
            {18, 39, 34, 38},
            {20, 21, 25, 41},
            {20, 25, 41, 24},
            {20, 41, 24, 40},
            {20, 21, 37, 41},
            {20, 37, 41, 36},
            {20, 41, 36, 40},
            {21, 22, 26, 42},
            {21, 26, 42, 25},
            {21, 42, 25, 41},
            {21, 22, 38, 42},
            {21, 38, 42, 37},
            {21, 42, 37, 41},
            {22, 23, 27, 43},
            {22, 27, 43, 26},
            {22, 43, 26, 42},
            {22, 23, 39, 43},
            {22, 39, 43, 38},
            {22, 43, 38, 42},
            {24, 25, 29, 45},
            {24, 29, 45, 28},
            {24, 45, 28, 44},
            {24, 25, 41, 45},
            {24, 41, 45, 40},
            {24, 45, 40, 44},
            {25, 26, 30, 46},
            {25, 30, 46, 29},
            {25, 46, 29, 45},
            {25, 26, 42, 46},
            {25, 42, 46, 41},
            {25, 46, 41, 45},
            {26, 27, 31, 47},
            {26, 31, 47, 30},
            {26, 47, 30, 46},
            {26, 27, 43, 47},
            {26, 43, 47, 42},
            {26, 47, 42, 46},
            {32, 33, 37, 53},
            {32, 37, 53, 36},
            {32, 53, 36, 52},
            {32, 33, 49, 53},
            {32, 49, 53, 48},
            {32, 53, 48, 52},
            {33, 34, 38, 54},
            {33, 38, 54, 37},
            {33, 54, 37, 53},
            {33, 34, 50, 54},
            {33, 50, 54, 49},
            {33, 54, 49, 53},
            {34, 35, 39, 55},
            {34, 39, 55, 38},
            {34, 55, 38, 54},
            {34, 35, 51, 55},
            {34, 51, 55, 50},
            {34, 55, 50, 54},
            {36, 37, 41, 57},
            {36, 41, 57, 40},
            {36, 57, 40, 56},
            {36, 37, 53, 57},
            {36, 53, 57, 52},
            {36, 57, 52, 56},
            {37, 38, 42, 58},
            {37, 42, 58, 41},
            {37, 58, 41, 57},
            {37, 38, 54, 58},
            {37, 54, 58, 53},
            {37, 58, 53, 57},
            {38, 39, 43, 59},
            {38, 43, 59, 42},
            {38, 59, 42, 58},
            {38, 39, 55, 59},
            {38, 55, 59, 54},
            {38, 59, 54, 58},
            {40, 41, 45, 61},
            {40, 45, 61, 44},
            {40, 61, 44, 60},
            {40, 41, 57, 61},
            {40, 57, 61, 56},
            {40, 61, 56, 60},
            {41, 42, 46, 62},
            {41, 46, 62, 45},
            {41, 62, 45, 61},
            {41, 42, 58, 62},
            {41, 58, 62, 57},
            {41, 62, 57, 61},
            {42, 43, 47, 63},
            {42, 47, 63, 46},
            {42, 63, 46, 62},
            {42, 43, 59, 63},
            {42, 59, 63, 58},
            {42, 63, 58, 62}
    };
    arma::Mat<long> Tetrahedons(6*(N-1)*(N-1)*(N-1), 4);

    // Test Lines
    cout << endl << "Lines" << endl;
    for (i=0; i<N-1; i++) {
        cout << D1.Element(i)(0) << ", " << D1.Element(i)(1) << endl;
    }
    // Test Triangles
    cout << endl << "Triangles" << endl;
    k  = (int) pow(N-1, 2)* (int) factorial(2);
    for (i=0; i<k; i++){
       // cout << "{" << D2.Element(i)(0) << ", " << D2.Element(i)(1) << ", " << D2.Element(i)(2) << "}," << endl;
    }

    // Test Tetrahedons
    cout << endl << "Tetrahedon" << endl;
    cout << "Don't forget to check compatibility of adverse faces of the cube!" << endl;
    cout << "NElements " << D3.nE << endl;
    k  = (int) pow(N-1, 3)* (int) factorial(3);
    for (i=0; i<k; i++){
        Tetrahedons(i, 0) = D3.Element(i)(0);
        Tetrahedons(i, 1) = D3.Element(i)(1);
        Tetrahedons(i, 2) = D3.Element(i)(2);
        Tetrahedons(i, 3) = D3.Element(i)(3);

        //cout << "{" << D3.Element(i)(0) << ", " << D3.Element(i)(1) << ", "
        //     << D3.Element(i)(2) << ", " << D3.Element(i)(3) << "}," << endl;
    }
    EXPECT_TRUE(arma::all(arma::vectorise(Tetrahedons == testTetrahedons)));

}

TEST(assembleTest, gridIndexTest) {
    int i=0, Vdx = 0;
    const double delta = .1;
    arma::Col<long> index;
    Grid2D D2(3, delta);
    Grid3D D3(3, delta);
    int N = D2.N;
    // Check that converting a Vdx (n) to Index (i,j,k) and back is mutual inverse.

    cout << "2D" << endl;
    Vdx  = 8;
    index = D2.toijk(Vdx);
    cout << "index: [" << index(0) << ", " << index(1) << "]" << endl;
    cout << "Vdx: " << D2.toVdx(index) << endl << endl;;
    EXPECT_EQ(D2.toVdx(D2.toijk(Vdx)), Vdx);
    EXPECT_TRUE(arma::all(arma::vectorise((D2.toijk(D2.toVdx(index)) == index))));

    cout << "3D" << endl;
    Vdx = 26;
    index = D3.toijk(Vdx);
    cout << "index: [" << index(0) << ", " << index(1) << ", " << index(2)  << "]" << endl;
    cout << "Vdx: " << D3.toVdx(index) << endl << endl;;
    EXPECT_EQ(D3.toVdx(D3.toijk(Vdx)), Vdx);
    EXPECT_TRUE(arma::all(arma::vectorise((D3.toijk(D3.toVdx(index)) == index))));

}

TEST(assembleTest, interpolatorTest){
    long i=0, j=0, k=0, n,N;
    const long n_Omega = 3, N_Omega = 2*n_Omega-1;
    const double delta = .5;
    double checkSum;
    Grid2D * ptrD2;

    Grid1D D1_coarse(n_Omega, delta);
    Grid2D D2_coarse(n_Omega, delta);
    Grid3D D3_coarse(n_Omega, delta);
    n = D1_coarse.N;

    D1_coarse.DataVdx.fill(arma::fill::ones);
    D2_coarse.DataVdx.fill(arma::fill::ones);
    D3_coarse.DataVdx.fill(arma::fill::ones);

    for (i=0; i<n; i++){
        D1_coarse.DataVdx(i) = i;
        for(j=0; j<n; j++){
            D2_coarse.DataVdx(i+n*j) = i+j*2;
            for (k=0; k<n; k++){
                D3_coarse.DataVdx(i+n*j + n*n*k) = i+j*2+k*3;
            }
        }
    }

    Grid1D D1 = D1_coarse.refine();
    ptrD2 = new Grid2D(& D2_coarse);
    //Grid3D D3 = D3_coarse.refine();
    N = D1.N;
    checkSum = 0;

    for (i=0; i<N; i++){
        checkSum += abs(D1.DataVdx(i) - ((double) i)/2.0);
        for(j=0; j<N; j++){
            checkSum += abs(ptrD2->DataVdx(i+N*j) - ((double)(i+j*2))/2.0);
            for (k=0; k<N; k++){
                //checkSum += abs(D3.DataVdx(i+N*j + N*N*k) - ((double)(i+j*2+k*3))/2.0);
            }
        }
    }
    EXPECT_DOUBLE_EQ(checkSum, 0.0);
}

TEST(assembleTest, neighbourTest){
    long Tdx=0;
    const long n = 3;
    const double delta = .1;
    Grid2D D2(n, delta);
    Grid3D D3(n, delta);

    arma::Mat<long> testNeighbours = { {1, 8, 5}, {2, 0, 8}, {3, 1, 7},
                                       {8, 2, 8}, {5, 8, 8}, {6, 4, 0},
                                       {7, 5, 8}, {8, 6, 2} };
    arma::Mat<long> testTetNeighbours = {{1, 3, 8, 48}, {0, 2, 15, 48}, {1, 5, 16, 48},
                                         {0, 4, 11, 48}, {3, 5, 24, 48}, {2, 4, 25, 48}};
    arma::Mat<long> outputNeighbours(D2.nE, 3);
    arma::Mat<long> outputTetNeighbours(6, 4);
    for (Tdx=0; Tdx<D2.nE; Tdx++) {
        outputNeighbours.row(Tdx) = D2.Neighbour(Tdx).t();
    }
    for (Tdx=0; Tdx<6; Tdx++){
        outputTetNeighbours.row(Tdx) = D3.Neighbour(Tdx).t();
    }
    EXPECT_TRUE(arma::all(arma::vectorise(outputNeighbours == testNeighbours)));
    EXPECT_TRUE(arma::all(arma::vectorise(outputTetNeighbours == testTetNeighbours)));
}