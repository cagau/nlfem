/**
    Contains assembly algorithm for nonlocal stiffnes matrix and forcing function.
    @file Cassemble.cpp
    @author Manuel Klar
    @version 0.1 25/08/20
*/

#include <iostream>
#include <cmath>
#include <queue>
#include <armadillo>
#include <map>

#include <Cassemble.h>

#include "integration.h"
#include "mathhelpers.h"
#include "model.h"
#include "checks.cpp"


using namespace std;
/**
 * This function looks up the configuration. It has to be updated whenever a new kernel,
 * forcing function or integration routine is added in order to make the option available.
 *
 * @param conf ConfigurationType
 * @param verbose
 */
void lookup_configuration(ConfigurationType & conf, int verbose=0){
    // Lookup right hand side ------------------------------------------------------------------------------------------
    if (verbose) cout << "Right hand side: " << conf.model_f << endl;
    //void (*model_f)(const double * x, double * forcing_out);
    map<string, void (*)(const double * x, double * forcing_out)> lookup_f = {
            {"linear", f_linear},
            {"linear3D", f_linear3D},
            {"constant", f_constant},
            {"linearField", fField_linear},
            {"constantRightField", fField_constantRight},
            {"constantDownField", fField_constantDown},
            {"constantBothField", fField_constantBoth}
    };
    map<string, void (*)(const double * x, double * forcing_out)>::iterator it_f;
    it_f = lookup_f.find(conf.model_f);
    if (it_f != lookup_f.end()){
        if (verbose) cout << "Forcing: " << conf.model_f << endl;
        model_f = lookup_f[conf.model_f];
    } else {
        if (verbose) cout << "No forcing function chosen." << endl;
        model_f = nullptr;
    }

    map<string, void (*)(const double * x, long labelx, const double * y, long labely,
                         const MeshType &mesh, double * kernel_val)> lookup_kernel = {
            {"constantTruncated", kernel_constantTruncated},
            {"constant", kernel_constant},
            {"constantLinf2D", kernel_constantLinf2D},
            {"labeled", kernel_labeled},
            {"constant3D", kernel_constant3D},
            {"constant1D", kernel_constant1D},
            {"parabola", kernel_parabola},
            {"linearPrototypeMicroelastic", kernel_linearPrototypeMicroelastic},
            {"linearPrototypeMicroelasticField", kernelField_linearPrototypeMicroelastic},
            {"constantField", kernelField_constant},
            {"fractional", kernel_fractional}
    };
    map<string, void (*)(const double * x, long labelx, const double * y, long labely,
                         const MeshType &mesh, double * kernel_val)>::iterator it_kernel;

    it_kernel = lookup_kernel.find(conf.model_kernel);
    if (it_kernel != lookup_kernel.end()){
        if (verbose) cout << "Kernel: " << conf.model_kernel << endl;
        model_kernel = lookup_kernel[conf.model_kernel];
    } else {
        if (verbose) cout << "No kernel chosen." << endl;
        model_kernel = nullptr;
    }

    map<string, bool> lookup_singularKernels = {
            {"linearPrototypeMicroelastic", true},
            {"linearPrototypeMicroelasticField", true},
            {"fractional", true}
    };
    conf.is_singularKernel = lookup_singularKernels[conf.model_kernel];


    // Lookup integration method  --------------------------------------------------------------------------------------
    map<string, int (*)(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule, const MeshType &mesh,
                        const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal, double *termNonloc,
                        double *termLocalPrime, double *termNonlocPrime)> lookup_integrate {
            {"baryCenter", integrate_baryCenter},
            {"subSetBall", integrate_subSuperSetBalls},
            {"averageBall", integrate_subSuperSetBalls},
            {"baryCenterRT", integrate_baryCenterRT},
            {"retriangulate", integrate_retriangulate},
            {"retriangulateLinfty", integrate_retriangulate},
            {"exactBall", integrate_exact},
            {"noTruncation", integrate_fullyContained},
            {"fractional", integrate_fractional},
            {"fractional_orig", integrate_fractional_orig},
            {"weakSingular", integrate_weakSingular},
            {"weakSingular_orig", integrate_weakSingular_orig}
    };
    map<string, int (*)(const ElementType &aT, const ElementType &bT, const QuadratureType &quadRule, const MeshType &mesh,
                        const ConfigurationType &conf, bool is_firstbfslayer, double *termLocal, double *termNonloc,
                        double *termLocalPrime, double *termNonlocPrime)>::iterator it_integrator;
    it_integrator = lookup_integrate.find(conf.integration_method_remote);
    if (it_integrator != lookup_integrate.end()){
        if (verbose) cout << "Integration Method Remote: " << conf.integration_method_remote << endl;
        integrate_remote = lookup_integrate[conf.integration_method_remote];
    } else {
        if (verbose) cout << "No integration routine for remote elements chosen." << endl;
        integrate_remote = nullptr;
    }

    it_integrator = lookup_integrate.find(conf.integration_method_close);
    if (it_integrator != lookup_integrate.end()){
        if (verbose) cout << "Integration Method Close: " << conf.integration_method_close << endl;
        integrate_close = lookup_integrate[conf.integration_method_close];
    } else {
        if (verbose) cout << "No integration routine for close elements chosen." << endl;
        integrate_close = nullptr;
    }

    if (conf.integration_method_remote == "retriangulateLinfty" || conf.integration_method_close == "retriangulateLinfty") {
        method = method_retriangulateInfty;
    } else {
        // Becomes effective only if integrate_retriangulate is actually used. Has no meaning otherwise.
        method = method_retriangulate;
    }


}
// Compute A and f -----------------------------------------------------------------------------------------------------
void compute_f(     const ElementType & aT,
                    const QuadratureType &quadRule,
                    const MeshType & mesh,
                    double * termf){
    int i,a;
    double x[mesh.dim];
    double forcing_value[mesh.outdim];

    for (a=0; a<mesh.dVertex*mesh.outdim; a++){
        for (i=0; i<quadRule.nPx; i++){
            toPhys(aT.E, &(quadRule.Px[mesh.dim * i]),  mesh.dim,&x[0]);
            model_f(&x[0], forcing_value);
            termf[a] += quadRule.psix(a/mesh.outdim, i) * forcing_value[a%mesh.outdim] * aT.absDet * quadRule.dx[i];
        }
    }
}

void par_evaluateMass(double *vd, const double *ud, long *Elements,
                      const long *ElementLabels, const double *Verts, const long * VertexLabels, int K_Omega, int J, int nP,
                      double *P, const double *dx, const int dim, const int outdim, const int is_DiscontinuousGalerkin) {
    const int dVerts = dim+1;
    double tmp_psi[dVerts];
    auto *psi = (double *) malloc((dVerts)*nP*sizeof(double));
    long aDGdx[dVerts]; // Index for discontinuous Galerkin

    for(int k=0; k<nP; k++){
        //model_basisFunction(const double * p, const MeshType & mesh, double *psi_vals){
       model_basisFunction(&P[dim*k], dim, &tmp_psi[0]);
       for (int kk=0; kk<dVerts; kk++) {
           psi[nP * kk + k] = tmp_psi[kk];
           //psi[nP * 1 + j] = tmp_psi[1];
           //psi[nP * 2 + j] = tmp_psi[2];
       }
    }

    #pragma omp parallel
    {
    //private(aAdx, a, b, aTE, aTdet, j)
        double aTdet;
        long * aAdx;

        double aTE[dim*(dVerts)];
        #pragma omp for
        for (int aTdx=0; aTdx < J; aTdx++){
            if (ElementLabels[aTdx] > 0) {
                // Get index of ansatz functions in matrix compute_A.-------------------

                // Discontinuous Galerkin
                if (is_DiscontinuousGalerkin) {
                    // Discontinuous Galerkin
                    //aDGdx[0] = (dVertex+1)*aTdx+1; aDGdx[1] = (dVertex+1)*aTdx+2; aDGdx[2] = (dVertex+1)*aTdx+3;
                    for (int j = 0; j < dVerts; j++) {
                        aDGdx[j] = dVerts * aTdx + j;
                    }
                    aAdx = aDGdx;
                } else {
                    // Continuous Galerkin
                    aAdx = &Elements[dVerts* aTdx];
                }

                // Prepare Triangle information aTE and aTdet ------------------
                // Copy coordinates of Triangel a to aTE.
                for (int jj=0; jj<dVerts; jj++){
                    for (int j = 0; j < dim; j++) {
                        aTE[dim * jj + j] = Verts[dim * Elements[dVerts * aTdx + jj] + j];
                    }
                    //aTE[2 * 0 + j] = Verts[2 * Elements[4 * aTdx + 1] + j];
                    //aTE[2 * 1 + j] = Verts[2 * Elements[4 * aTdx + 2] + j];
                    //aTE[2 * 2 + j] = Verts[2 * Elements[4 * aTdx + 3] + j];
                }
                // compute Determinant
                aTdet = absDet(&aTE[0], dim);

                for (int a = 0; a < dVerts; a++) {
                    if(is_DiscontinuousGalerkin || VertexLabels[ aAdx[a] ] > 0) {
                        for (int aOut = 0; aOut < outdim; aOut++) {
                            for (int b = 0; b < dVerts; b++) {
                                if(is_DiscontinuousGalerkin || VertexLabels[ aAdx[b] ] > 0) {
                                    for (int bOut = 0; bOut < outdim; bOut++) {
                                        for (int j = 0; j < nP; j++) {
                                            // Evaluation
#pragma omp atomic update
                                            vd[outdim*aAdx[a] + aOut] +=
                                                    psi[nP * a + j] * psi[nP * b + j] * aTdet * dx[j] * ud[outdim*aAdx[b] + bOut];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    } // Pragma Omp Parallel
    free(psi);
}

void estimateNNZperRow(const MeshType & mesh, const ConfigurationType & conf){
    const int sampleSize = 3;
    int indexList[sampleSize];
    int nE = mesh.nE;
    bool isDG = mesh.is_DiscontinuousGalerkin;

    ElementType aT, bT;
    aT.matE = arma::vec(mesh.dim*(mesh.dim+1));
    bT.matE = arma::vec(mesh.dim*(mesh.dim+1));
    
    arma::Col<int> rowCount(mesh.K, arma::fill::zeros);
    arma::Col<int> bvertexVisited(mesh.nV, arma::fill::zeros);
    arma::Col<int> avertexVisited(mesh.nV, arma::fill::zeros);

    // Traverse Triangles by Neighbours
    for (int & k : indexList){
        k = rand() % nE;
    }
    // Breadth First Search --------------------------------------
    // Every Thread owns its copy!
    arma::Col<int> visited(nE, arma::fill::zeros);

    // Queue for Breadth first search
    queue<int> queue;
    // List of visited neighbours
    const long *NTdx;
    //for (int aTdx=0; aTdx<mesh.nE; aTdx++)
    for (int & aTdx : indexList){
        initializeElement(aTdx, mesh, aT);
        // Intialize search queue with current outer triangle
        queue.push(aTdx);
        // Initialize vector of visited triangles with 0
        visited.zeros();

        while (!queue.empty()) {
            // Get and delete the next Triangle indexList of the queue. The first one will be the triangle aTdx itself.
            int sTdx = queue.front();
            queue.pop();
            // Get all the neighbours of sTdx.
            NTdx = &mesh.Neighbours(0, sTdx);

            // Run through the list of neighbours.
            for (int j = 0; j < mesh.nNeighbours; j++) {

                // The next valid neighbour is our candidate for the inner Triangle b.
                int bTdx = NTdx[j];
                // Check how many neighbours sTdx has.
                // In order to be able to store the list as contiguous array we fill
                // up the empty spots with the number nE
                // i.e. the total number of Triangles (which cannot be an indexList).
                if (bTdx < mesh.nE) {
                    // Check whether bTdx is already visited.
                    if (!visited(bTdx)){
                        initializeElement(bTdx, mesh, bT);

                        double abary[mesh.dim], bbary[mesh.dim];
                        baryCenter(mesh.dim, aT.E, abary);
                        baryCenter(mesh.dim, bT.E, bbary);
                        double dist = sqrt(vec_sqL2dist(abary, bbary, mesh.dim));
                        if (dist < mesh.delta + mesh.maxDiameter){
                            queue.push(bTdx);

                            if (isDG) {
                                rowCount(3*aTdx) += 3;
                                rowCount(3*aTdx+1) += 3;
                                rowCount(3*aTdx+2) += 3;
                            } else {
                                for (int i = 0; i < mesh.dVertex; i++) {
                                    int bVdx = mesh.Triangles(i, bTdx);
                                    if (!bvertexVisited(bVdx)) {
                                        for (int k = 0; k < mesh.dVertex; k++) {
                                            int aVdx = mesh.Triangles(k, aTdx);
                                            if (!avertexVisited(aVdx)) {
                                                //Vdx = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx + k+1];
                                                rowCount(aVdx) += 1;
                                            }
                                        } // End for (int k = 0; k < mesh.dVertex; k++)
                                        bvertexVisited(bVdx) = 1;
                                    } // End if (!bvertexVisited(bVdx))
                                } // End for (int i = 0; i < mesh.dVertex; i++)
                            } // End if (isDG)

                        } // End  if (dist < mesh.delta + mesh.maxDiameter)
                    } // End visited[bTdx] = 1;
                    visited[bTdx] = 1;
                } // if (bTdx < mesh.nE)
            } // End for (int j = 0; j < mesh.nNeighbours; j++)
        } // End while (!queue.empty())
        for (int k = 0; k < mesh.dVertex; k++) {
            int aVdx = mesh.Triangles(k, aTdx);
            avertexVisited(aVdx) = 1;
        }
    } // End for (int aTdx=0; aTdx<mesh.nE; aTdx++)

    printf("C++ Estimated Row NNZ %i\n", arma::max(rowCount));
} // End estimateNNZperRow

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(const string compute, const string path_spAd, const string path_fd, const int K_Omega, const int K,
                  const long *ptrTriangles, const long *ptrLabelTriangles, const double *ptrVerts, const long * ptrLabelVerts, const int nE,
                  const int nE_Omega, const int nV, const int nV_Omega, const double *Px, const int nPx, const double *dx,
                  const double *Py, const int nPy, const double *dy, const double sqdelta, const long *ptrNeighbours,
                  const int nNeighbours,
                  const int is_DiscontinuousGalerkin, const int is_NeumannBoundary, const string str_model_kernel,
                  const string str_model_f, const string str_integration_method_remote,
                  const string str_integration_method_close,
                  const int is_PlacePointOnCap,
                  const int dim, const int outdim, const long * ptrZeta, const long nZeta,
                  const double * Pg, const int degree, const double * dg, double maxDiameter, double fractional_s,
                  int is_fullConnectedComponentSearch, int verbose) {

    MeshType mesh = {K_Omega, K, ptrTriangles, ptrLabelTriangles, ptrVerts, ptrLabelVerts, nE, nE_Omega,
                     nV, nV_Omega, sqrt(sqdelta), sqdelta, ptrNeighbours, nNeighbours, is_DiscontinuousGalerkin,
                     is_NeumannBoundary, dim, outdim, dim+1, ptrZeta, nZeta, maxDiameter, fractional_s};

    ConfigurationType conf = {path_spAd, path_fd, str_model_kernel, str_model_f,
                              str_integration_method_remote, str_integration_method_close,
                              static_cast<bool>(is_PlacePointOnCap),
                              false,
                              static_cast<bool>(is_fullConnectedComponentSearch),
                              verbose};

    QuadratureType quadRule = {Px, Py, dx, dy, nPx, nPy, dim, Pg, dg, degree};
    initializeQuadrule(quadRule, mesh);

    if (compute=="system") {
        map<unsigned long, double> Ad;
        chk_Mesh(mesh, verbose);
        chk_Conf(mesh, conf, quadRule);

        estimateNNZperRow(mesh, conf);

        par_system(Ad, mesh, quadRule, conf);
        if (verbose) cout << "K_Omega " << mesh.K_Omega << endl;
        if (verbose) cout << "K " << mesh.K << endl;

        int nnz_total = static_cast<int>(Ad.size());
        arma::vec values_all(nnz_total);
        arma::umat indices_all(2, nnz_total);
        if (verbose) cout << "Total NNZ " << nnz_total << endl;

        int k = 0;
        for (auto &it : Ad) {
            unsigned long adx = it.first;
            double value = it.second;
            values_all(k) = value;
            // column major format of transposed matrix Ad
            indices_all(0, k) = adx % mesh.K;
            indices_all(1, k) = adx / mesh.K;
            //printf("Index a %llu, b %llu, k %i\n", indices_all(0, k), indices_all(1, k), k);
            k++;
        }
        arma::sp_mat sp_Ad(true, indices_all, values_all, mesh.K, mesh.K);

        sp_Ad.save(conf.path_spAd);
    }

    if (compute=="forcing") {
        par_forcing(mesh, quadRule, conf);
    }
}

template <typename T_Matrix>
void par_system(T_Matrix &Ad, MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {

    const int verbose = conf.verbose;

    if (verbose) printf("Function: par_system\n");
    if (verbose) printf("Ansatz Space: %s\n", mesh.is_DiscontinuousGalerkin ? "DG" : "CG");
    if (verbose) printf("Mesh dimension: %i\n", mesh.dim);
    if (verbose) printf("Output dimension: %i\n", mesh.outdim);
    if (verbose) printf("Recieved Zeta for DD: %s\n", (mesh.ptrZeta) ? "true" : "false");
    lookup_configuration(conf, verbose);
    if (verbose) printf("Quadrule outer: %i\n", quadRule.nPx);
    if (verbose) printf("Quadrule inner: %i\n", quadRule.nPy);
    if (verbose) printf("Full Graph Search: %i\n", conf.is_fullConnectedComponentSearch);

    for(int h=0; h<quadRule.nPx; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Px[mesh.dim*h], mesh.dim, & quadRule.psix[mesh.dVertex * h]);
    }
    for(int h=0; h<quadRule.nPy; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Py[mesh.dim*h], mesh.dim,& quadRule.psiy[mesh.dVertex * h]);
    }
    chk_BasisFunction(quadRule);

    #pragma omp parallel shared(mesh, quadRule, conf,  Ad)
    {
    //map<unsigned long, double> Ad;
    unsigned long Adx;

    // Breadth First Search --------------------------------------
    arma::Col<int> visited(mesh.nE, arma::fill::zeros);

    // Queue for Breadth first search
    queue<int> queue;
    // List of visited triangles
    const long *NTdx;

    // Vector containing the coordinates of the vertices of a Triangle
    ElementType aT, bT;
    aT.matE = arma::vec(mesh.dim*(mesh.dim+1));
    bT.matE = arma::vec(mesh.dim*(mesh.dim+1));

    // Integration information ------------------------------------
    // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
    const long * aAdx;
    const long * bAdx;
    long aDGdx[mesh.dVertex]; // Index for discontinuous Galerkin
    long bDGdx[mesh.dVertex];

    // Buffers for integration solutions
    double termLocal[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];
    double termNonloc[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];
    double termLocalPrime[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];
    double termNonlocPrime[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];

    #pragma omp for
    for (int aTdx=0; aTdx<mesh.nE; aTdx++) {
        if (mesh.LabelTriangles[aTdx]) {

            //[DEBUG]
            /*
            if (aTdx==7){
                cout << endl << "Total Local Term" << endl ;
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalLocal[0], DEBUG_termTotalLocal[1], DEBUG_termTotalLocal[2]);
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalLocal[3], DEBUG_termTotalLocal[4], DEBUG_termTotalLocal[5]);
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalLocal[6], DEBUG_termTotalLocal[7], DEBUG_termTotalLocal[8]);

                cout << endl << "Total Nonlocal Term" << endl ;
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalNonloc[0], DEBUG_termTotalNonloc[1], DEBUG_termTotalNonloc[2]);
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalNonloc[3], DEBUG_termTotalNonloc[4], DEBUG_termTotalNonloc[5]);
                printf ("[%17.16e, %17.16e, %17.16e] \n", DEBUG_termTotalNonloc[6], DEBUG_termTotalNonloc[7], DEBUG_termTotalNonloc[8]);

                abort();
            }
            */
            //[DEBUG]

            // Get index of Ansatz functions in matrix compute_A.-------------------
            if (mesh.is_DiscontinuousGalerkin) {
                // Discontinuous Galerkin
                for (int j = 0; j < mesh.dVertex; j++) {
                    aDGdx[j] = mesh.dVertex * aTdx + j;
                }
                aAdx = aDGdx;
            } else {
                // Continuous Galerkin
                aAdx = &mesh.Triangles(0, aTdx);
            }
            // Prepare Triangle information aTE and aTdet ------------------
            initializeElement(aTdx, mesh, aT);

            // BFS -------------------------------------------------------------
            // Intialize search queue with current outer triangle
            queue.push(aTdx);
            // Initialize vector of visited triangles with 0
            visited.zeros();

            // Tells that we are in the first layer of the BFS
            bool is_firstbfslayer = conf.is_singularKernel;
            // Check whether BFS is completed.
            while (!queue.empty()) {
                // Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
                int sTdx = queue.front();
                queue.pop();
                // Get all the neighbours of sTdx.
                NTdx = &mesh.Neighbours(0, sTdx);
                // Run through the list of neighbours.
                for (int j = 0; j < mesh.nNeighbours; j++) {
                    // The next valid neighbour is our candidate for the inner Triangle b.
                    int bTdx = NTdx[j];

                    // Check how many neighbours sTdx has.
                    // In order to be able to store the list as contiguous array we fill
                    // up the empty spots with the number nE
                    // i.e. the total number of Triangles (which cannot be an index).
                    if (bTdx < mesh.nE) {
                        // Check whether bTdx is already visited.
                        if (visited[bTdx] == 0) {
                            // Check whether bTdx is part of the discretization
                            // otherwise it is just appended to the queue
                            if (mesh.LabelTriangles[bTdx]) {
                                // Prepare Triangle information bTE and bTdet ------------------
                                initializeElement(bTdx, mesh, bT);
                                // Retriangulation and integration -----------------------------
                                if (mesh.is_DiscontinuousGalerkin) {
                                    // Discontinuous Galerkin
                                    for (int jj = 0; jj < mesh.dVertex; jj++) {
                                        bDGdx[jj] = mesh.dVertex * bTdx + jj;
                                    }
                                    bAdx = bDGdx;
                                } else {
                                    // Get (pointer to) index of basis function (in Continuous Galerkin)
                                    bAdx = &mesh.Triangles(0, bTdx);
                                }

                                // Domain decomposition. If Zeta is empty, the weight is set to 1.
                                double weight = 1.;
                                if (mesh.ptrZeta) {
                                    double zeta = arma::dot(mesh.ZetaIndicator.col(aTdx), mesh.ZetaIndicator.col(bTdx));
                                    weight = 1. / zeta;
                                }

                                // Assembly of matrix ---------------------------------------
                                doubleVec_tozero(termLocal, mesh.dVertex * mesh.dVertex * mesh.outdim *
                                                            mesh.outdim); // Initialize Buffer
                                doubleVec_tozero(termNonloc, mesh.dVertex * mesh.dVertex * mesh.outdim *
                                                             mesh.outdim); // Initialize Buffer
                                doubleVec_tozero(termLocalPrime, mesh.dVertex * mesh.dVertex * mesh.outdim *
                                                                 mesh.outdim); // Initialize Buffer
                                doubleVec_tozero(termNonlocPrime, mesh.dVertex * mesh.dVertex * mesh.outdim *
                                                                  mesh.outdim); // Initialize Buffer

                                // Compute integrals and write to buffer
                                int doesInteract = integrate(aT, bT, quadRule, mesh, conf, is_firstbfslayer,
                                                             termLocal, termNonloc, termLocalPrime, termNonlocPrime);
                                if (doesInteract) {

                                    // If bT interacts it will be a candidate for our BFS, so it is added to the queue

                                    //[DEBUG]
                                    /*
                                    if (is_firstbfslayer){//aTdx == 0 && bTdx == 0){

                                    printf("aTdx %i\ndet %17.16e, label %li \n", aTdx, aT.absDet, aT.label);
                                    printf ("aTE\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n", aT.E[0],aT.E[1],aT.E[2],aT.E[3],aT.E[4],aT.E[5]);
                                    printf("bTdx %i\ndet %17.16e, label %li \n", bTdx, bT.absDet, bT.label);
                                    printf ("bTE\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n", bT.E[0],bT.E[1],bT.E[2],bT.E[3],bT.E[4],bT.E[5]);

                                    cout << endl << "Local Term" << endl ;
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[0], termLocal[1], termLocal[2]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[3], termLocal[4], termLocal[5]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[6], termLocal[7], termLocal[8]);

                                    cout << endl << "Nonlocal Term" << endl ;
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[0], termNonloc[1], termNonloc[2]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[3], termNonloc[4], termNonloc[5]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[6], termNonloc[7], termNonloc[8]);

                                    cout << endl << "Local Term Prime" << endl ;
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocalPrime[0], termLocalPrime[1], termLocalPrime[2]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocalPrime[3], termLocalPrime[4], termLocalPrime[5]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termLocalPrime[6], termLocalPrime[7], termLocalPrime[8]);

                                    cout << endl << "Nonlocal Term Prime" << endl ;
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonlocPrime[0], termNonlocPrime[1], termNonlocPrime[2]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonlocPrime[3], termNonlocPrime[4], termNonlocPrime[5]);
                                    printf ("[%17.16e, %17.16e, %17.16e] \n", termNonlocPrime[6], termNonlocPrime[7], termNonlocPrime[8]);
                                    //if (aTdx == 10){
                                    //    abort();
                                    //}

                                    }
                                    if ((aTdx == 1) && !is_firstbfslayer) abort();
                                    */
                                    //[End DEBUG]

                                    queue.push(bTdx);
                                    // We only check whether the integral
                                    // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                                    // However, this works only if we can guarantee that interacting triangles do actually
                                    // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere on its support for example.
                                    // The effect (in speedup) of this more precise criterea depends on delta and meshsize.

                                    // Copy buffer into matrix. Again solutions which lie on the boundary are ignored (in Continuous Galerkin)
                                    for (int a = 0; a < mesh.dVertex * mesh.outdim; a++) {
                                        for (int b = 0; b < mesh.dVertex * mesh.outdim; b++) {
                                            //printf("Local: a %i, b %i, ker (%i, %i) \nAdx %lu \n", a/mesh.outdim, b/mesh.outdim, a%mesh.outdim, b%mesh.outdim, Adx);
                                            // TERM LOCAL & TERM NON-LOCAL ORDER
                                            // Note: This order is not trivially obtained from innerNonloc, as b switches in between.
                                            // However it mimics the matrix which results from the multiplication.
                                            //                                      Kernel switches back here. v
                                            // [(a 0, b 0, ker (0,0)), (a 0, b 0, ker (0,1)), (a 0, b 1, ker (0,0)), (a 0, b 1, ker (0,1)), (a 0, b 2, ker (0,0)), (a 0, b 2, ker (0,1)),
                                            //  (a 0, b 0, ker (1,0)), (a 0, b 0, ker (1,1)), (a 0, b 0, ker (1,0)), (a 0, b 1, ker (1,1)), (a 0, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
                                            //  (a 1, b 0, ker (0,0)), (a 1, b 0, ker (0,1)), (a 1, b 1, ker (0,0)), (a 1, b 1, ker (0,1)), (a 1, b 2, ker (0,0)), (a 1, b 2, ker (0,1)),
                                            //  (a 1, b 0, ker (1,0)), (a 1, b 0, ker (1,1)), (a 1, b 0, ker (1,0)), (a 1, b 1, ker (1,1)), (a 1, b 2, ker (1,0)), (a 0, b 2, ker (1,1)),
                                            //  (a 2, b 0, ker (0,0)), (a 2, b 0, ker (0,1)), (a 2, b 1, ker (0,0)), (a 2, b 1, ker (0,1)), (a 2, b 2, ker (0,0)), (a 2, b 2, ker (0,1)),
                                            //  (a 2, b 0, ker (1,0)), (a 2, b 0, ker (1,1)), (a 2, b 0, ker (1,0)), (a 2, b 1, ker (1,1)), (a 2, b 2, ker (1,0)), (a 2, b 2, ker (1,1))]

                                            if (mesh.is_DiscontinuousGalerkin ||
                                                (mesh.LabelVerts[aAdx[a / mesh.outdim]] > 0)) {
                                                Adx = (mesh.outdim * aAdx[a / mesh.outdim] + a % mesh.outdim) * mesh.K +
                                                      mesh.outdim * aAdx[b / mesh.outdim] + b % mesh.outdim;
#pragma omp critical
                                                {
                                                    Ad[Adx] += termLocal[mesh.dVertex * mesh.outdim * a + b] * weight;
                                                }

                                                Adx = (mesh.outdim * aAdx[a / mesh.outdim] + a % mesh.outdim) * mesh.K +
                                                      mesh.outdim * bAdx[b / mesh.outdim] + b % mesh.outdim;
#pragma omp critical
                                                {
                                                    Ad[Adx] += -termNonloc[mesh.dVertex * mesh.outdim * a + b] * weight;
                                                }
                                            }
                                            if (mesh.is_DiscontinuousGalerkin ||
                                                (mesh.LabelVerts[bAdx[b / mesh.outdim]] > 0)) {
                                                Adx = (mesh.outdim * bAdx[b / mesh.outdim] + b % mesh.outdim) * mesh.K +
                                                      mesh.outdim * bAdx[a / mesh.outdim] + a % mesh.outdim;
#pragma omp critical
                                                {
                                                    Ad[Adx] +=
                                                            termLocalPrime[mesh.dVertex * mesh.outdim * a + b] * weight;
                                                }

                                                Adx = (mesh.outdim * bAdx[b / mesh.outdim] + b % mesh.outdim) * mesh.K +
                                                      mesh.outdim * aAdx[a / mesh.outdim] + a % mesh.outdim;
#pragma omp critical
                                                {
                                                    Ad[Adx] += -termNonlocPrime[mesh.dVertex * mesh.outdim * a + b] *
                                                               weight;
                                                }
                                            }
                                        }
                                    }
                                }
                            }// End if(mesh.LabelTriangles[bTdx])
                            else {
                                queue.push(bTdx);
                            }
                        }// End if BFS (visited[bTdx] == 0)
                        // Mark bTdx as visited
                        visited[bTdx] = 1;
                    }// End if BFS (bTdx < mesh.nE)
                }//End for loop BFS (j = 0; j < mesh.nNeighbours; j++)
                is_firstbfslayer = false;
            }//End while loop BFS (!queue.empty())
       }// End if LabelTriangles != 0
    }// End parallel for

    }// End pragma omp parallel

}// End function par_system

void par_forcing(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {
    arma::vec fd(mesh.K, arma::fill::zeros);
    const int verbose = conf.verbose;
    if (verbose) printf("Function: par_forcing\n");
    //printf("Mesh dimension: %i\n", mesh.dim);
    //printf("Recieved Zeta for DD: %s\n", (mesh.ptrZeta) ? "true" : "false");
    lookup_configuration(conf);
    //printf("Quadrule outer: %i\n", quadRule.nPx);
    //printf("Quadrule inner: %i\n", quadRule.nPy);

    for (int h = 0; h < quadRule.nPx; h++) {
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(&quadRule.Px[mesh.dim * h], mesh.dim, &quadRule.psix[mesh.dVertex * h]);
    }

    #pragma omp parallel
    {
        // General Loop Indices ---------------------------------------
        // Vector containing the coordinates of the vertices of a Triangle
        ElementType aT;
        aT.matE = arma::vec(mesh.dim * (mesh.dim + 1));
        // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
        const long *aAdx;
        long aDGdx[mesh.dVertex]; // Index for discontinuous Galerkin
        // Buffers for integration solutions
        double termf[mesh.dVertex*mesh.outdim];
        #pragma omp for
        for (int aTdx = 0; aTdx < mesh.nE; aTdx++) {
            if (mesh.LabelTriangles[aTdx] > 0) {
                // Get index of ansatz functions in matrix compute_A.-------------------
                if (mesh.is_DiscontinuousGalerkin) {
                    // Discontinuous Galerkin
                    for (int j = 0; j < mesh.dVertex; j++) {
                        aDGdx[j] = mesh.dVertex * aTdx + j;
                    }
                    aAdx = aDGdx;
                } else {
                    // Continuous Galerkin
                    aAdx = &mesh.Triangles(0, aTdx);
                }
                // Prepare Triangle information aTE and aTdet ------------------
                initializeElement(aTdx, mesh, aT);
                // Assembly of right side ---------------------------------------
                // We unnecessarily integrate over vertices which might lie on the boundary of Omega for convenience here.
                doubleVec_tozero(termf, mesh.dVertex*mesh.outdim); // Initialize Buffer

                compute_f(aT, quadRule, mesh, termf); // Integrate and fill buffer
                // Add content of buffer to the right side.

                // Domain decomposition. If Zeta is empty, the weight is set to 1.
                double weight = 1.;
                if(mesh.ptrZeta){
                    double zeta = arma::dot(mesh.ZetaIndicator.col(aTdx), mesh.ZetaIndicator.col(aTdx));
                    weight = 1./zeta;
                }

                for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                    if (mesh.is_DiscontinuousGalerkin || (mesh.LabelVerts[aAdx[a/mesh.outdim]] > 0)) {
                        #pragma omp atomic update
                        fd[mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim] += termf[a]*weight;
                    }
                }// end for rhs

            }// end outer if (mesh.LabelTriangles[aTdx] > 0)
        }// end outer for loop (aTdx=0; aTdx<mesh.nE; aTdx++)
    }// end pragma omp parallel
    fd.save(conf.path_fd);
}// end par_righthandside

// [DEBUG] _____________________________________________________________________________________________________________
/*
double compute_area(double * aTE, double aTdet, long labela, double * bTE, double bTdet, long labelb, double * P, int nP, double * dx, double sqdelta){
    double areaTerm=0.0;
    int rTdx, Rdx;
    double * x;
    double physical_quad[2];
    double reTriangle_list[9*3*2];
    const MeshType mesh = {K_Omega, K, ptrTriangles, ptrVerts, nE, nE_Omega,
                                        L, L_Omega, sqdelta, ptrNeighbours, is_DiscontinuousGalerkin,
                                        is_NeumannBoundary, dim, dim+1};;
    x = &P[0];
    toPhys(aTE, x, physical_quad);
    Rdx = retriangulate(physical_quad, bTE, mesh, reTriangle_list, true);
    for (rTdx=0; rTdx < Rdx; rTdx++){
        areaTerm += absDet(&reTriangle_list[2*3*rTdx])/2;
    }
    return areaTerm;
}
*/
