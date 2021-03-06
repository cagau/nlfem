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
 * @param conf
 */
void lookup_configuration(ConfigurationType & conf){
    // Lookup right hand side ------------------------------------------------------------------------------------------
    cout << "Right hand side: " << conf.model_f << endl;
    if (conf.model_f == "linear") {
        model_f = f_linear;
    } else if (conf.model_f == "linear3D") {
        model_f = f_linear3D;
    } else if (conf.model_f == "constant"){
        model_f = f_constant;
    } else if (conf.model_f == "linearField"){
        model_f = fField_linear;
    } else if (conf.model_f == "constantRightField"){
        model_f = fField_constantRight;
    } else if (conf.model_f == "constantDownField"){
        model_f = fField_constantDown;
    } else if (conf.model_f == "constantBothField"){
        model_f = fField_constantBoth;
    } else {
        cout << "No right hand side chosen" << endl;
        model_kernel = nullptr;
        //cout << "Error in Cassemble lookup_configuration. Right hand side: " << conf.model_f << " is not implemented." << endl;
        //abort();
    }

    // Lookup kernel ---------------------------------------------------------------------------------------------------
    // Alternatively to if we can lookup the kernel in a map. As in given below for the constant kernel case.
    // https://stackoverflow.com/questions/17762232/use-string-to-class-lookup-table-in-c-to-instantiate-classes
    //map<string, void (*)(const double *, long, const double *, long, double, double *)> lookupKernelName {
    //        {"constant", kernel_constant}
    //};

    cout << "Kernel: " << conf.model_kernel << endl;
    if (conf.model_kernel == "constant"){
        model_kernel = kernel_constant;
        //model_kernel = lookupKernelName[conf.model_kernel];
    } else if (conf.model_kernel == "constantTruncated") {
        model_kernel = kernel_constantTruncated;
    } else if (conf.model_kernel == "labeled") {
        model_kernel = kernel_labeled;
    } else if (conf.model_kernel == "constant3D") {
        model_kernel = kernel_constant3D;
    } else if (conf.model_kernel == "constant1D") {
        model_kernel = kernel_constant1D;
    } else if (conf.model_kernel == "parabola") {
        model_kernel = kernel_parabola;
        conf.is_singularKernel = true;
    } else if (conf.model_kernel == "linearPrototypeMicroelastic") {
        model_kernel = kernel_linearPrototypeMicroelastic;
        conf.is_singularKernel = true;
    } else if (conf.model_kernel == "linearPrototypeMicroelasticField") {
        model_kernel = kernelField_linearPrototypeMicroelastic;
        conf.is_singularKernel = true;
    } else if (conf.model_kernel == "constantField") {
        model_kernel = kernelField_constant;
    }
    else {
        cout << "No kernel chosen" << endl;
        model_kernel = nullptr;
        //cout << "Error in Cassemble lookup_configuration. Kernel " << conf.model_kernel << " is not implemented." << endl;
        //abort();
    }

    // Lookup integration method  --------------------------------------------------------------------------------------
    cout << "Integration Method: " << conf.integration_method << endl;
    if ( conf.model_kernel == "constantField"){ // Test Case
        if (conf.integration_method == "baryCenter") {
            integrate = integrate_linearPrototypeMicroelastic_baryCenter;
        } else if (conf.integration_method == "baryCenterRT") {
            integrate = integrate_linearPrototypeMicroelastic_baryCenterRT;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
        } else if (conf.integration_method == "retriangulate") {
            integrate = integrate_linearPrototypeMicroelastic_retriangulate;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
        } else {
            cout << "No integration method chosen" << endl;
            model_kernel = nullptr;
            //cout << "Error in Cassemble lookup_configuration. Integration method " << conf.integration_method <<
            //     " is not implemented." << endl;
            //abort();
        }
    } else if ( conf.model_kernel == "linearPrototypeMicroelastic" ||
                conf.model_kernel == "linearPrototypeMicroelasticField") {
        if (conf.integration_method == "baryCenter") {
            integrate = integrate_linearPrototypeMicroelastic_baryCenter;
        } else if (conf.integration_method == "baryCenterRT") {
            integrate = integrate_linearPrototypeMicroelastic_baryCenterRT;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
        } else if (conf.integration_method == "retriangulate") {
            integrate = integrate_linearPrototypeMicroelastic_retriangulate;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
        } else {
            cout << "No integration method chosen" << endl;
            model_kernel = nullptr;
            // cout << "Error in par:assemble. Integration method " << conf.integration_method <<
            //     " is not implemented." << endl;
            // abort();
        }
    }  else {
        if (conf.integration_method == "baryCenter") {
            integrate = integrate_baryCenter;
        } else if (conf.integration_method == "subSetBall") {
            integrate = integrate_subSuperSetBalls;

        } else if (conf.integration_method == "superSetBall") {
            integrate = integrate_subSuperSetBalls;

        } else if (conf.integration_method == "averageBall") {
            integrate = integrate_subSuperSetBalls;

        } else if (conf.integration_method == "baryCenterRT") {
            integrate = integrate_baryCenterRT;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");

        } else if (conf.integration_method == "retriangulate") {
            integrate = integrate_retriangulate;
            printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");

        } else if (conf.integration_method == "noTruncation") {
            integrate = integrate_fullyContained;

        } else if (conf.integration_method == "tensorgauss") {
            integrate = integrate_tensorgauss;
            conf.is_singularKernel = true; // Test Case
        }
        else {
            cout << "No integration method chosen" << endl;
            model_kernel = nullptr;
            //cout << "Error in Cassemble lookup_configuration. Integration method " << conf.integration_method <<
            //     " is not implemented." << endl;
            //abort();
        }
    }
}

void initializeTriangle( const int Tdx, const MeshType & mesh, ElementType & T){
    /*
    Copy coordinates of Triange b to bTE.

     Tempalte of Triangle Point data.
     2D Case, a, b, c are the vertices of a triangle
     T.E -> | a1 | a2 | b1 | b2 | c1 | c2 |
     Hence, if one wants to put T.E into col major order matrix it would be of shape\
                                 | a1 | b1 | c1 |
     M(mesh.dim, mesh.dVerts) =  | a2 | b2 | c2 |
    */

    int j, k, Vdx;
    //T.matE = arma::vec(dim*(dim+1));
    for (k=0; k<mesh.dVertex; k++) {
        //Vdx = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx + k+1];
        Vdx = mesh.Triangles(k, Tdx);
        for (j=0; j<mesh.dim; j++){
            T.matE[mesh.dim * k + j] = mesh.Verts(j, Vdx);
            //printf ("%3.2f ", T.matE[mesh.dim * k + j]);
            //T.matE[mesh.dim * k + j] = mesh.ptrVerts[ mesh.dim*Vdx + j];
        }
        //printf("\n");
    }
    // Initialize Struct
    T.E = T.matE.memptr();
    T.absDet = absDet(T.E, mesh.dim);
    T.signDet = static_cast<int>(signDet(T.E, mesh));
    T.label = mesh.LabelTriangles(Tdx);

    //T.label = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx];
    //T.dim = dim;
    T.dim = mesh.dim;
    T.Tdx = Tdx;
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

void par_evaluateMass(double *vd, double *ud, long *Elements,
                      long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
                      double *P, double *dx, const int dim, const int outdim, const bool is_DiscontinuousGalerkin) {
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
                        aTE[dim * jj + j] = Verts[dim *aAdx[jj] + j];
                    }
                    //aTE[2 * 0 + j] = Verts[2 * Elements[4 * aTdx + 1] + j];
                    //aTE[2 * 1 + j] = Verts[2 * Elements[4 * aTdx + 2] + j];
                    //aTE[2 * 2 + j] = Verts[2 * Elements[4 * aTdx + 3] + j];
                }
                // compute Determinant
                aTdet = absDet(&aTE[0], dim);

                for (int a = 0; a < dVerts; a++) {
                    if (aAdx[a] < K_Omega/outdim) {
                        for (int aOut = 0; aOut < outdim; aOut++) {
                            for (int b = 0; b < dVerts; b++) {
                                if (aAdx[b] < K_Omega/outdim) {
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

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(const string compute, const string path_spAd, const string path_fd, const int K_Omega, const int K,
                  const long *ptrTriangles, const long *ptrLabelTriangles, const double *ptrVerts, const long * ptrLabelVerts, const int nE,
                  const int nE_Omega, const int nV, const int nV_Omega, const double *Px, const int nPx, const double *dx,
                  const double *Py, const int nPy, const double *dy, const double sqdelta, const long *ptrNeighbours,
                  const int nNeighbours,
                  const int is_DiscontinuousGalerkin, const int is_NeumannBoundary, const string str_model_kernel,
                  const string str_model_f, const string str_integration_method, const int is_PlacePointOnCap,
                  const int dim, const int outdim, const long * ptrZeta, const long nZeta,
                  const double * Pg, const int degree, const double * dg, double maxDiameter) {

    MeshType mesh = {K_Omega, K, ptrTriangles, ptrLabelTriangles, ptrVerts, ptrLabelVerts, nE, nE_Omega,
                     nV, nV_Omega, sqrt(sqdelta), sqdelta, ptrNeighbours, nNeighbours, is_DiscontinuousGalerkin,
                     is_NeumannBoundary, dim, outdim, dim+1, ptrZeta, nZeta, maxDiameter};

    QuadratureType quadRule = {Px, Py, dx, dy, nPx, nPy, dim, Pg, dg, degree};
    chk_QuadratureRule(quadRule);

    ConfigurationType conf = {path_spAd, path_fd, str_model_kernel, str_model_f, str_integration_method, static_cast<bool>(is_PlacePointOnCap)};

    if (compute=="system") {
        map<unsigned long, double> Ad;

        chk_Mesh(mesh);
        chk_Conf(mesh, conf, quadRule);

        par_system(Ad, mesh, quadRule, conf);

        cout << "K_Omega " << mesh.K_Omega << endl;
        cout << "K " << mesh.K << endl;
        int nnz_total = static_cast<int>(Ad.size());
        arma::vec values_all(nnz_total);
        arma::umat indices_all(2, nnz_total);
        cout << "Total NNZ " << nnz_total << endl;

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

void par_system(map<unsigned long, double> &Ad, MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {
    printf("Function: par_system\n");
    printf("Ansatz Space: %s\n", mesh.is_DiscontinuousGalerkin ? "DG" : "CG");
    printf("Mesh dimension: %i\n", mesh.dim);
    printf("Output dimension: %i\n", mesh.outdim);
    printf("Recieved Zeta for DD: %s\n", (mesh.ptrZeta) ? "true" : "false");
    lookup_configuration(conf);
    printf("Quadrule outer: %i\n", quadRule.nPx);
    printf("Quadrule inner: %i\n", quadRule.nPy);

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

    #pragma omp for
    for (int aTdx=0; aTdx<mesh.nE; aTdx++) {
        if (mesh.LabelTriangles[aTdx] > 0) {

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
            initializeTriangle(aTdx, mesh, aT);

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
                            // Prepare Triangle information bTE and bTdet ------------------
                            initializeTriangle(bTdx, mesh, bT);
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
                            // Assembly of matrix ---------------------------------------
                            doubleVec_tozero(termLocal, mesh.dVertex * mesh.dVertex*mesh.outdim*mesh.outdim); // Initialize Buffer
                            doubleVec_tozero(termNonloc, mesh.dVertex * mesh.dVertex*mesh.outdim*mesh.outdim); // Initialize Buffer
                            // Compute integrals and write to buffer
                            integrate(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);

                            // If bT interacts it will be a candidate for our BFS, so it is added to the queue

                            //[DEBUG]
                            /*
                            if (aTdx == 0 && bTdx == 45){

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
                            //if (aTdx == 10){
                            //    abort();
                            //}
                            abort();
                            }
                            */
                            //[End DEBUG]

                            // Domain decomposition. If Zeta is empty, the weight is set to 1.
                            double weight = 1.;
                            if (mesh.ptrZeta){
                                double zeta = arma::dot(mesh.ZetaIndicator.col(aTdx), mesh.ZetaIndicator.col(bTdx));
                                weight = 1./zeta;
                            }

                            if (doubleVec_any(termNonloc, mesh.dVertex * mesh.dVertex) ||
                                doubleVec_any(termLocal, mesh.dVertex * mesh.dVertex)) {
                                queue.push(bTdx);
                                // We only check whether the integral
                                // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                                // However, this works only if we can guarantee that interacting triangles do actually
                                // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere on its support for example.
                                // The effect (in speedup) of this more precise criterea depends on delta and meshsize.

                                // Copy buffer into matrix. Again solutions which lie on the boundary are ignored (in Continuous Galerkin)
                                for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                                    if (mesh.is_DiscontinuousGalerkin || (mesh.LabelVerts[aAdx[a/mesh.outdim]] > 0)) {
                                        for (int b = 0; b < mesh.dVertex*mesh.outdim; b++) {
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

                                            Adx =  (mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim) * mesh.K +
                                                    mesh.outdim*aAdx[b/mesh.outdim] + b%mesh.outdim;

                                            #pragma omp critical
                                            {
                                                Ad[Adx] += termLocal[mesh.dVertex * mesh.outdim * a + b] * weight;
                                            }

                                            Adx =   (mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim) * mesh.K +
                                                     mesh.outdim*bAdx[b/mesh.outdim] + b%mesh.outdim;
                                            #pragma omp critical
                                            {
                                                Ad[Adx] += -termNonloc[mesh.dVertex * mesh.outdim * a + b] * weight;
                                            }
                                        }
                                    }
                                }
                            }

                        }// End if BFS (visited[bTdx] == 0)
                        // Mark bTdx as visited
                        visited[bTdx] = 1;
                    }// End if BFS (bTdx < mesh.nE)
                }//End for loop BFS (j = 0; j < mesh.nNeighbours; j++)
                is_firstbfslayer = false;
            }//End while loop BFS (!queue.empty())
        }// End if LabelTriangles > 0
    }// End parallel for

    }// End pragma omp parallel

}// End function par_system

void par_forcing(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {
    arma::vec fd(mesh.K, arma::fill::zeros);

    printf("Function: par_forcing\n");
    printf("Mesh dimension: %i\n", mesh.dim);
    printf("Recieved Zeta for DD: %s\n", (mesh.ptrZeta) ? "true" : "false");
    lookup_configuration(conf);
    printf("Quadrule outer: %i\n", quadRule.nPx);
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
                initializeTriangle(aTdx, mesh, aT);
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
