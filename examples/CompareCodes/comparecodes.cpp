//
// Created by klar on 11.03.20.
//

#include <MeshBuilder.h>
#include <Cassemble2D.h>
#include <Cassemble.h>

double uExact(arma::vec x){
    return pow(x(0), 2)*x(1) + pow(x(1),2);
}
int main() {
    cout << "Compare Codes 2D\n" << endl;
    int nPx, nPy;
    double delta = 0.1;
    double sqdelta = pow(delta, 2);
    int N_Omega = 11;
    arma::mat Px, Py;
    arma::vec dx, dy;
    arma::vec x(2);
    Px.load("conf/P16.txt");
    nPx = Px.n_rows;
    Px = Px.t();
    Py.load("conf/P1.txt");
    nPy = Py.n_rows;
    Py = Py.t();
    dx.load("conf/d16.txt");
    dy.load("conf/d1.txt");

    int k;
    bool compute2D = false;

    cout << "\nConstruct Grid and Adjaciency Graph..." << endl;
    Grid2D coarseGrid(N_Omega, delta);
    MeshType coarseMesh = coarseGrid.mesh(true);
    arma::mat Ad(coarseMesh.K, coarseMesh.K_Omega, arma::fill::zeros);
    arma::vec fd(coarseMesh.K_Omega, arma::fill::zeros);
    cout << "Start Assembly..." << endl;

    if (compute2D) {
        par_assemble2D(Ad.memptr(),
                       coarseMesh.K,
                       fd.memptr(),
                       coarseMesh.Triangles.memptr(),
                       coarseMesh.LabelTriangles.memptr(),
                       coarseMesh.Verts.memptr(),
                       coarseMesh.J, coarseMesh.J_Omega,
                       coarseMesh.L, coarseMesh.L_Omega, Px.memptr(), nPx, dx.memptr(), Py.memptr(), nPy, dy.memptr(),
                       sqdelta,
                       coarseMesh.Neighbours.memptr(),
                       false,
                       false);
    } else {
        par_assemble(Ad.memptr(),
                     coarseMesh.K_Omega,
                     coarseMesh.K,
                     fd.memptr(),
                     coarseMesh.Triangles.memptr(),
                     coarseMesh.LabelTriangles.memptr(),
                     coarseMesh.Verts.memptr(),
                     coarseMesh.J, coarseMesh.J_Omega,
                     coarseMesh.L, coarseMesh.L_Omega, Px.memptr(), nPx, dx.memptr(), Py.memptr(), nPy, dy.memptr(),
                     sqdelta,
                     coarseMesh.Neighbours.memptr(),
                     false,
                     false, 2);
    };


    return 0;
}