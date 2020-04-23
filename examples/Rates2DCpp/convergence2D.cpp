//
// Created by klar on 11.02.20.
//
#include <MeshBuilder.h>
#include <Cassemble2D.h>
double uExact(arma::vec x){
    return pow(x(0), 2)*x(1) + pow(x(1),2);
}
int main(){
    cout << "Convergence Test 2D\n" << endl;
    int nPx, nPy;
    double delta = 0.1;
    double err, err_=1;
    double sqdelta = pow(delta, 2);
    arma::Col<long> N_Omega={11,21,41};
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


    int nInterations = N_Omega.n_elem;
    int i,k;
    Grid2D *pGrid;

    for (i=0; i<nInterations; i++){
        cout << "\nConstruct Grid and Adjaciency Graph..." << endl;
        Grid2D coarseGrid(N_Omega(i), delta);
        MeshType coarseMesh = coarseGrid.mesh(true);
        arma::mat Ad(coarseMesh.K, coarseMesh.K_Omega, arma::fill::zeros);
        arma::vec fd(coarseMesh.K_Omega, arma::fill::zeros);
        cout << "Start Assembly..." << endl;
        par_assemble2D(Ad.memptr(),
                coarseMesh.K,
                fd.memptr(),
                coarseMesh.Triangles.memptr(),
                coarseMesh.LabelTriangles.memptr(),
                coarseMesh.Verts.memptr(),
                coarseMesh.J, coarseMesh.J_Omega,
                coarseMesh.L, coarseMesh.L_Omega,Px.memptr(), nPx, dx.memptr(), Py.memptr(), nPy, dy.memptr(), sqdelta,
                coarseMesh.Neighbours.memptr(),
                 false,
                 false);
        Ad = Ad.t();
        //Ad.save("data/Ad", arma::arma_ascii);
        arma::vec gd(coarseMesh.K - coarseMesh.K_Omega, arma::fill::zeros);
        for (k=0; k<coarseMesh.K - coarseMesh.K_Omega; k++){
            x = coarseMesh.Verts.col(k+coarseMesh.K_Omega);
            gd(k) = uExact(x);
            coarseGrid.DataVdx(coarseGrid.sortIndex(k+coarseMesh.K_Omega)) = gd(k);
        }
        fd -= Ad.cols(coarseMesh.K_Omega, coarseMesh.K-1)*gd;
        arma::vec ud(coarseMesh.K_Omega, arma::fill::zeros);
        ud = arma::solve(Ad.cols(0, coarseMesh.K_Omega-1), fd);
        for (k=0; k<coarseMesh.K_Omega; k++){
            coarseGrid.DataVdx(coarseGrid.sortIndex(k)) = ud(k);
        }
        pGrid = &coarseGrid;
        k=0;
        cout << pGrid -> N_Omega;
        while (k < (nInterations-i-1+2)){
            cout << " refine > ";
            pGrid = new Grid2D(pGrid); // Copy constructor performs refinement!
            k++;
        }
        cout << pGrid -> N_Omega << endl;
        MeshType fineMesh = pGrid->mesh(false);
        //coarseGrid.save("uCoarse"+to_string(N_Omega(i)));
        arma::vec diff(fineMesh.K_Omega, arma::fill::zeros);

        for (k=0; k<fineMesh.K_Omega; k++){
            x = fineMesh.Verts.col(k);
            pGrid -> DataVdx(pGrid->sortIndex(k)) -= uExact(x);
            diff(k) = pGrid -> DataVdx(pGrid->sortIndex(k));
        }
        //pGrid -> save("fineGrid");
        arma::vec MassDiff(fineMesh.K_Omega, arma::fill::zeros);

        //void par_evaluateMass2D(double * vd, double * ud, const long * Triangles, const double * Verts, int K_Omega, int J_Omega, int nP, double * P, double * dx);
        par_evaluateMass2D(
                MassDiff.memptr(),
                diff.memptr(),
                fineMesh.Triangles.memptr(),
                fineMesh.Verts.memptr(),
                fineMesh.K_Omega,
                fineMesh.J_Omega,
                nPx, Px.memptr(), dx.memptr()
                );
        cout << "h " << 1./(N_Omega(i)-1) << endl;
        err = sqrt(arma::dot(diff, MassDiff));
        cout << "L2_Distance " << err << endl;
        cout << "Rates " << log2(err_/err) << endl;
        err_ = err;

        pGrid = 0;

    }
    return 0;
}