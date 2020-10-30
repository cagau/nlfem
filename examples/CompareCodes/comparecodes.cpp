//
// Created by klar on 11.03.20.
//
#include "armadillo"
#include "metis.h"
#include "MeshTypes.h"
#include "Cassemble.h"
#include "iostream"
#include "cstdio"
#include "map"
#include "omp.h"
#include "queue"
#include "list"
#include "vector"

using namespace std;

double uExact(arma::vec x){
    return pow(x(0), 2)*x(1) + pow(x(1),2);
}

int safereadi(const char * name, fstream & f){
    string fline;
    int value;
    getline(f, fline);
    if (strcmp(name, fline.c_str()) != 0)
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
    if (strcmp(name, fline.c_str()) != 0)
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
    if (strcmp(name, fline.c_str()) != 0)
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
    if (strcmp(name, fline.c_str()) != 0)
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

void baryCenter_tmp(const double * a, const double * b, const double * c, double * bary){
    bary[0] = (a[0] + b[0] + c[0])/3.;
    bary[1] = (a[1] + b[1] + c[1])/3.;
}
double vec_sqL2dist_tmp(const double * x, const double * y, const int len){
    double r=0;
    int i=0;
    for (i=0; i<len; i++){
        r += pow((x[i] - y[i]), 2);
    }
    return r;
}

int read_configuration(const string &path, idx_t nparts){
    fstream f;
    string path_conf = path + "/conf";
    string path_mesh = path + "/mesh.conf";
    string path_verts = path + "/mesh.verts";
    string path_elemt = path + "/mesh.elemt";
    string path_elelb = path + "/mesh.elelb";
    string path_neigh = path + "/mesh.neigh";
    string path_Px = path + "/quad.Px";
    string path_Py = path + "/quad.Py";
    string path_dx = path + "/quad.dx";
    string path_dy = path + "/quad.dy";
    string path_Ad = path + "/result.Ad";
    string path_fd = path + "/result.fd";
    string path_partition = path + "/result.partition";

    int K_Omega, K,J, J_Omega, L, L_Omega, is_DiscontinuousGalerkin,
        is_NeumannBoundary, dim, is_PlacePointOnCap, nNeighbours;
    double sqdelta;
    string str_model_kernel, str_model_f, str_integration_method;


    // Read general Configuration
    f.open(path_conf.c_str(), ios::in);
    if (f.is_open()) {
        printf(" \nReading ConfigurationType ---------------------------------------------------------------------\n");
        str_model_kernel = safereads("model_kernel", f);
        str_model_f = safereads("model_f", f);
        str_integration_method = safereads("integration_method", f);
        is_PlacePointOnCap = safereadb("is_PlacePointOnCap", f);
    }  else {
        cout << "Error in read_configuration(): Could not open file " << path << "/conf." << endl;
        abort();
    }
    f.close();

    // Read Mesh Configuration
    f.open(path_mesh.c_str(), ios::in);
    if (f.is_open()){
        string fline;
        printf(" \nReading MeshType ConfigurationType ----------------------------------------------------------------\n");
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
        nNeighbours = safereadi("nNeighbours", f);
        dim = safereadi("dim", f);
    } else {
        cout << "Error in read_configuration(): Could not open file " << path << "/mesh.conf." << endl;
        abort();
    }
    f.close();

    // Read Mesh Data
    printf( " \nReading Mesh Data ----------------------------------------------------------------------------\n");
    arma::Mat<long> elements;
    arma::Mat<long> elementLabels;
    arma::mat vertices;
    arma::Mat<long> neighbours;
    elements.load(path_elemt, arma::raw_binary);
    elementLabels.load(path_elelb, arma::raw_binary);
    vertices.load(path_verts, arma::raw_binary);
    neighbours.load(path_neigh, arma::raw_binary);

    arma::mat Px;
    arma::mat Py;
    arma::vec dx;
    arma::vec dy;
    Px.load(path_Px, arma::raw_binary);
    Py.load(path_Py, arma::raw_binary);
    dx.load(path_dx, arma::raw_binary);
    dy.load(path_dy, arma::raw_binary);

    MeshType mesh = {K_Omega, K, elements.memptr(), elementLabels.memptr(), vertices.memptr(), J, J_Omega,
                     L, L_Omega, sqrt(sqdelta), sqdelta, neighbours.memptr(), nNeighbours, is_DiscontinuousGalerkin,
                     is_NeumannBoundary, dim, 1,dim + 1, nullptr, 0, 0.01};

    QuadratureType quadRule = {Px.memptr(), Py.memptr(), dx.memptr(), dy.memptr(),
                               static_cast<int>(dx.n_elem), static_cast<int>(dy.n_elem), dim};

    ConfigurationType conf = {"sp_Ad", "fd", str_model_kernel, str_model_f,
                              str_integration_method, static_cast<bool>(is_PlacePointOnCap)};

    arma::vec fd(K_Omega);
    //idx_t options[METIS_NOPTIONS];
    //METIS_SetDefaultOptions(options);

    // METIS TEST ------------------------------------------------------------------------------------------------------
    idx_t nE = mesh.nE;
    idx_t nV = mesh.nV;
    idx_t ncommon = 1;
    //The partitions ntended to be quite spiky with ncommon=mesh.dim (recommended by METIS).
    //However, I assume in the nonlocal case this is not favored.
    idx_t epart[nE], npart[nV];
    idx_t objval = 0;
    idx_t eind[nE * mesh.dVertex];
    idx_t eptr[nE + 1];

    for (int k=0; k<nE; k++){
        for (int l=0; l<mesh.dVertex; l++){
            eind[mesh.dVertex*k + l] = elements[mesh.dVertex*k + l];
        }
        eptr[k] = k*mesh.dVertex;
    }
    eptr[nE] = nE*mesh.dVertex;
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
    options[METIS_OBJTYPE_VOL] = 1;

    // Partition mit METIS berechnen.
    int ret = METIS_PartMeshDual(&nE, &nV, eptr, eind,
                       nullptr, nullptr, &ncommon, &nparts, nullptr,
                       options, &objval, epart, npart);
    cout << ret << endl;
    arma::vec partition(mesh.nV);
    for (int k=0; k<mesh.nV; k++){
        partition[k] = static_cast<double>(npart[k]);
    }
    par_system(mesh, quadRule, conf);

    //Ad.save(path_Ad.c_str(), arma::raw_binary);
    // Vertex-zerlegung speichern
    partition.save(path_partition, arma::arma_binary);
    // [End] METIS TEST ------------------------------------------------------------------------------------------------

    idx_t numflag=0;
    idx_t *xadj;
    idx_t *adjncy;

    // Compute Adjacency Graph with METIS
    METIS_MeshToDual(&nE, &nV, eptr, eind, &ncommon, &numflag, &xadj, &adjncy); // ???
    arma::mat dualGraph(13, nE);

    // Breadth First Search to determine interaction graph

    // Breadth First Search --------------------------------------
    long maxSize=0;
    map<unsigned long, unsigned long> interaction;
    arma::Col<int> visited(mesh.nE, arma::fill::zeros);
    arma::Col<int> colored(mesh.nE, arma::fill::zeros);
    // Queue for Breadth first search
    queue<idx_t> queue;

    for (idx_t aTdx=0; aTdx<mesh.nE; aTdx++) {
        queue.push(aTdx);
        double aBary[2];
        baryCenter_tmp(&vertices[elements[3*aTdx]],
                       &vertices[elements[3*aTdx+1]],
                       &vertices[elements[3*aTdx+2]],
                       aBary);

        // Initialize vector of visited triangles with 0
        visited.zeros();
        // Check whether BFS is completed.
        while (!queue.empty()) {
            int sTdx = queue.front();
            queue.pop();
            // Get all the neighbours of sTdx.
            idx_t startNdx = xadj[sTdx];
            idx_t endNdx = xadj[sTdx+1];
            idx_t nTdx = 0;
            while (startNdx + nTdx < endNdx) {
                // The next valid neighbour is our candidate for the inner Triangle b.
                idx_t bTdx = adjncy[startNdx + nTdx];

                double bBary[2];
                baryCenter_tmp(&vertices[elements[3*bTdx]],
                               &vertices[elements[3*bTdx+1]],
                               &vertices[elements[3*bTdx+2]],
                               bBary);
                // Check whether bTdx is already visited.
                if (!visited[bTdx]) {
                    if (vec_sqL2dist_tmp(aBary, bBary, 2) < sqdelta  + mesh.maxDiameter){
                        interaction[aTdx*nE + bTdx] = 1;
                        queue.push(bTdx);
                    }
                }
                visited[bTdx] = 1;
                nTdx++;
            }
        }
    }


    arma::Mat<arma::uword> indices(2, interaction.size());
    arma::Col<double> values(interaction.size(), arma::fill::ones);
    int k = 0;
    for(auto & it : interaction) {
        unsigned long adx = it.first;
        indices(0, k) = adx % mesh.nE;
        indices(1, k) = adx / mesh.nE;
        k++;
    }
    arma::SpMat<double> interactionMatrix(indices, values, mesh.nE, mesh.nE);
    interactionMatrix.save("data/result.interaction");

    // Save Graph for Python
    dualGraph.fill(nE);

    for (idx_t i = 0; i < nE; i++){
        idx_t s = xadj[i];
        idx_t j = 0;
        while(s + j < xadj[i+1]){
            dualGraph(j, i) = adjncy[s + j];
            j++;
        }
    }
    dualGraph.save("data/result.dual", arma::arma_binary);

    METIS_Free(xadj);
    METIS_Free(adjncy);

    std::cout << "Check out OMP's nested parallelism" << endl;

    #pragma omp parallel num_threads(nparts) default(none)
    {
        printf("Level 1 thread num %d of %d.\n", omp_get_thread_num(), omp_get_num_threads());
        #pragma omp parallel default(none)
        {
            printf("Level 2 thread num %d of %d.\n", omp_get_thread_num(), omp_get_num_threads());
            #pragma omp barrier
            #pragma omp master
            {
            printf("Only INNER masters talking here. (After all of my slaves)\n");
            }
        }
        #pragma omp barrier
        #pragma omp master
        {
            printf("Only OUTER master talking here. (After all others)");
        }
    }

    return 0;
}



int main(int argc, char *argv[]) {
    if (argc < 2){
        cout << "ERROR: Please hand over nparts!" << endl;
        abort();
    }
    //cout << argv[1] << endl;
    idx_t nparts = *argv[1] - '0';
    cout << "Partition into " << nparts << " parts." << endl;
    string path = "data";
    read_configuration(path, nparts);
    return 0;
}