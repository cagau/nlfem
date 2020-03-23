//
// Created by klar on 11.03.20.
//
#include "armadillo"
// #include <MeshBuilder.h>
// #include <Cassemble2D.h>
// #include "Cassemble.h"
#include "iostream"
#include "cstdio"

using namespace std;

double uExact(arma::vec x){
    return pow(x(0), 2)*x(1) + pow(x(1),2);
}

int safereadi(const char * name, fstream & f){
    string fline;
    int value;
    getline(f, fline);
    if (strcmp(name, fline.c_str()))
    {
        cout << "Error in read_configuration(): Variable name "<< name <<
             " and file "<< fline.c_str() <<" do not match." << endl;
        abort();
    } else {
        getline(f, fline);
        value = stoi(fline);
        cout << name << " " << value << endl;
        return value;
    }

}

double safereadd(const char * name, fstream & f){
    string fline;
    double value;
    getline(f, fline);
    if (strcmp(name, fline.c_str()))
    {
        cout << "Error in read_configuration(): Variable name "<< name <<
             " and file "<< fline.c_str() <<" do not match." << endl;
        abort();
    } else {
        getline(f, fline);
        value = atof(fline.c_str());
        cout << name << " " << value << endl;
        return value;
    }

}

string safereads(const char * name, fstream & f){
    string fline;
    getline(f, fline);
    if (strcmp(name, fline.c_str()))
    {
        cout << "Error in read_configuration(): Variable name "<< name <<
             " and file "<< fline.c_str() <<" do not match." << endl;
        abort();
    } else {
        getline(f, fline);
        cout << name << " " << fline << endl;
        return fline;
    }

}

int safereadb(const char * name, fstream & f){
    string fline;
    int value;
    getline(f, fline);
    if (strcmp(name, fline.c_str()))
    {
        cout << "Error in read_configuration(): Variable name "<< name <<
             " and file "<< fline.c_str() <<" do not match." << endl;
        abort();
    } else {
        getline(f, fline);
        value = !strcmp("True", fline.c_str());
        cout << name << " " << value << endl;
        return value;

    }

}

int read_configuration(string path){
    fstream f;
    string path_conf = path + "/conf";
    string path_mesh = path + "/mesh.conf";
    string path_verts = path + "/mesh.verts";
    string path_elemt = path + "/mesh.elemt";
    string path_neigh = path + "/mesh.neigh";

    int K_Omega, K,J, J_Omega, L, L_Omega, is_DiscontinuousGalerkin,
        is_NeumannBoundary, dim, dVertex, is_PlacePointOnCap;
    double sqdelta;
    string model_kernel, model_f, integration_method;


    // Read general Configuration
    f.open(path_conf.c_str(), ios::in);
    if (f.is_open()) {
        model_kernel = safereads("model_kernel", f);
        model_f = safereads("model_f", f);
        integration_method = safereads("integration_method", f);
        is_PlacePointOnCap = safereadb("is_PlacePointOnCap", f);
    }  else {
        printf("Error in read_configuration(): Could not open file %s/conf.", path);
        abort();
    }
    f.close();

    // Read Mesh Configuration
    f.open(path_mesh.c_str(), ios::in);
    if (f.is_open()){
        string fline;
        printf("Reading Configuration file ---------------------------------------------\n");
        printf("path %s/mesh.conf\n\n", path.c_str());
        K_Omega = safereadi("K_Omega", f);
        K = safereadi("K", f);
        J = safereadi("nE", f);
        J_Omega = safereadi("nE_Omega", f);
        L = safereadi("nV", f);
        L_Omega = safereadi("nV_Omega", f);
        sqdelta = safereadd("sqdelta", f);
        is_DiscontinuousGalerkin = safereadb("is_DiscontinuousGalerkin", f);
        is_NeumannBoundary = safereadb("is_NeumannBoundary", f);
        dim = safereadi("dim", f);
        dVertex = dim+1;
    } else {
        printf("Error in read_configuration(): Could not open file %s/mesh.conf.", path);
        abort();
    }
    f.close();

    // Read Mesh Data
    arma::mat elements;
    arma::mat vertices;
    arma::Mat<long> neighbours;
    elements.load(path_elemt.c_str(), arma::raw_binary);
    vertices.load(path_verts.c_str(), arma::raw_binary);
    neighbours.load(path_neigh.c_str(), arma::raw_binary);



    //MeshType mesh = {K_Omega, K, ptrTriangles, ptrLabelTriangles, ptrVerts, J, J_Omega,
    //                 L, L_Omega, sqdelta, ptrNeighbours, is_DiscontinuousGalerkin,
    //                 is_NeumannBoundary, dim, dim+1};



    return 0;

}

int main() {
    arma::mat A;
    if (A.load("examples/RatesScipy3D/results/A.bin", arma::raw_binary)){
        cout << A(0,0) << endl;
        cout << "Ok!" << endl;
    }

    arma::sp_mat B(3,3);
    //  sp_mat(locations, values, sort_locations = true)
    B(0,0) = 1.0;
    B(1,1) = 2.0;
    B(2,2)  =4.0;

    //B.save("examples/RatesScipy3D/results/B.bin");
    string path = "examples/RatesScipy3D/results";
    read_configuration(path);

    /*
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
                     false,
                     "constant",
                     "linear",
                     "retriangulate",
                     1,
                     2);
    };
    */

    return 0;
}