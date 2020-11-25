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
        cout << "Error in Cassemble lookup_configuration. Right hand side: " << conf.model_f << " is not implemented." << endl;
        abort();
    }

    // Lookup kernel ---------------------------------------------------------------------------------------------------
    cout << "Kernel: " << conf.model_kernel << endl;
    if (conf.model_kernel == "constant"){
        model_kernel = kernel_constant;
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
        cout << "Error in Cassemble lookup_configuration. Kernel " << conf.model_kernel << " is not implemented." << endl;
        abort();
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
            cout << "Error in Cassemble lookup_configuration. Integration method " << conf.integration_method <<
                 " is not implemented." << endl;
            abort();
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
            cout << "Error in par:assemble. Integration method " << conf.integration_method <<
                 " is not implemented." << endl;
            abort();
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
            cout << "Error in Cassemble lookup_configuration. Integration method " << conf.integration_method <<
                 " is not implemented." << endl;
            abort();
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

void par_assembleMass(double * Ad, long * Triangles, double * Verts, int K_Omega, int J_Omega, int nP, double * P, double * dx){
    int aTdx=0, a=0, b=0, j=0;
    double aTE[2*3];
    double aTdet;
    double tmp_psi[3];
    long * aAdx;

    double *psi = (double *) malloc(3*nP*sizeof(double));

    for(j=0; j<nP; j++){
       model_basisFunction(&P[2*j], &tmp_psi[0]);
       psi[nP*0+j] = tmp_psi[0];
       psi[nP*1+j] = tmp_psi[1];
       psi[nP*2+j] = tmp_psi[2];
    }

    #pragma omp parallel for private(aAdx, a, b, aTE, aTdet, j)
    for (aTdx=0; aTdx < J_Omega; aTdx++){
        // Get index of ansatz functions in matrix compute_A.-------------------
        // Continuous Galerkin
        aAdx = &Triangles[4*aTdx+1];
        // Discontinuous Galerkin
        // - Not implemented -

        // Prepare Triangle information aTE and aTdet ------------------
        // Copy coordinates of Triange a to aTE.
        // this is done fore convenience only, actually those are unnecessary copies!
        for (j=0; j<2; j++){
            aTE[2*0+j] = Verts[2*Triangles[4*aTdx+1] + j];
            aTE[2*1+j] = Verts[2*Triangles[4*aTdx+2] + j];
            aTE[2*2+j] = Verts[2*Triangles[4*aTdx+3] + j];
        }
        // compute Determinant
        aTdet = absDet(&aTE[0]);

        for (a=0; a<3; a++){
            if (aAdx[a] < K_Omega){
                for (b=0; b<3; b++){
                    if (aAdx[b] < K_Omega){
                        for (j=0; j<nP; j++){
                            // Assembly
                            #pragma omp atomic update
                            Ad[aAdx[a]*K_Omega + aAdx[b]] += psi[nP*a+j]*psi[nP*b+j]*aTdet*dx[j];
                            // Evaluation
                            //vd[aAdx[a]] += psi[nP*a+j]*psi[nP*b+j]*aTdet*dx[j]*ud[aAdx[b]];
                        }
                    }
                }
            }
        }
    }

}

void
par_evaluateMass(double *vd, double *ud, long *Elements, long *ElementLabels, double *Verts, int K_Omega, int J, int nP,
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
                  const long *ptrTriangles, const long *ptrLabelTriangles, const double *ptrVerts, const int nE,
                  const int nE_Omega, const int nV, const int nV_Omega, const double *Px, const int nPx, const double *dx,
                  const double *Py, const int nPy, const double *dy, const double sqdelta, const long *ptrNeighbours,
                  const int nNeighbours,
                  const int is_DiscontinuousGalerkin, const int is_NeumannBoundary, const string str_model_kernel,
                  const string str_model_f, const string str_integration_method, const int is_PlacePointOnCap,
                  const int dim, const int outdim, const long * ptrZeta, const long nZeta,
                  const double * Pg, const int degree, const double * dg, double maxDiameter) {
    //const long * ptrZeta;
    //cout << "nZeta is" << nZeta << endl;

    // [1]
    // Mesh will contain K_Omega = outdim*nV_Omega in CG case,
    // K_Omega = outdim*3*nE_Omega in DG [X] Not implemented!
    MeshType mesh = {K_Omega, K, ptrTriangles, ptrLabelTriangles, ptrVerts, nE, nE_Omega,
                     nV, nV_Omega, sqrt(sqdelta), sqdelta, ptrNeighbours, nNeighbours, is_DiscontinuousGalerkin,
                     is_NeumannBoundary, dim, outdim, dim+1, ptrZeta, nZeta, maxDiameter};
    // [2]
    // Above should be checked
    chk_Mesh(mesh);

    QuadratureType quadRule = {Px, Py, dx, dy, nPx, nPy, dim, Pg, dg, degree};
    chk_QuadratureRule(quadRule);
    ConfigurationType conf = {path_spAd, path_fd, str_model_kernel, str_model_f, str_integration_method, static_cast<bool>(is_PlacePointOnCap)};
    // [3]
    // kernel has to match outdim. Can we, or do we want to check this?
    chk_Conf(mesh, conf, quadRule);

    if (compute=="system") {
        par_system(mesh, quadRule, conf);
    }
    if (compute=="forcing") {
        par_forcing(mesh, quadRule, conf);
    }
}

void par_system(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {

    printf("Function: par_system\n");
    printf("Ansatz Space: %s\n", mesh.is_DiscontinuousGalerkin ? "DG" : "CG");
    printf("Mesh dimension: %i\n", mesh.dim);
    printf("Output dimension: %i\n", mesh.outdim);
    printf("Recieved Zeta for DD: %s\n", (mesh.ptrZeta) ? "true" : "false");
    lookup_configuration(conf);
    printf("Quadrule outer: %i\n", quadRule.nPx);
    printf("Quadrule inner: %i\n", quadRule.nPy);

    //const int dVertex = dim + 1;
    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!
    //arma::Mat<double> Ad(ptrAd, mesh.K, mesh.K_Omega, false, true);
    //Ad.zeros();
    //arma::sp_mat sp_Ad(mesh.K, mesh.K_Omega);
    //arma::vec values_all;
    //arma::umat indices_all(2,0);
    //int nnz_total=0;

    for(int h=0; h<quadRule.nPx; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Px[mesh.dim*h], mesh.dim, & quadRule.psix[mesh.dVertex * h]);
    }
    for(int h=0; h<quadRule.nPy; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Py[mesh.dim*h], mesh.dim,& quadRule.psiy[mesh.dVertex * h]);
    }
    chk_BasisFunction(quadRule);
    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!
    // Contains one row more than number of verticies as label information is contained here
    //const arma::Mat<long> Triangles(mesh.ptrTriangles, mesh.dVertex+1, mesh.nE);
    //mesh.Triangles = arma::Mat<long>(mesh.ptrTriangles, mesh.dVertex+1, mesh.nE);
    // Contains number of direct neighbours of an element + 1 (itself).
    //const arma::Mat<long> Neighbours(mesh.ptrNeighbours, mesh.dVertex, mesh.nE);
    //const arma::Mat<double> Verts(mesh.ptrVerts, mesh.dim, mesh.L);

    // default(none)  does not work well with g++.
    //const auto chunkSize = mesh.nE / omp_get_num_procs() ;
    //printf("Chunk Size %i\n", chunkSize);
    map<unsigned long, double> Ad;

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
    //double aTE[3*2];
    //double bTE[3*2];
    // Integration information ------------------------------------
    // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
    const long * aAdx;
    const long * bAdx;
    long aDGdx[mesh.dVertex]; // Index for discontinuous Galerkin
    long bDGdx[mesh.dVertex];

    // Buffers for integration solutions
    //double termf[mesh.dVertex];
    double termLocal[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];
    double termNonloc[mesh.dVertex*mesh.dVertex*mesh.outdim*mesh.outdim];
    //[DEBUG]
    /*
    double DEBUG_termTotalLocal[3*3];
    double DEBUG_termTotalNonloc[3*3];
    */
    //[End DEBUG]
    //long debugTdx = 570;

    #pragma omp for
    for (int aTdx=0; aTdx<mesh.nE; aTdx++) {
        //if (aTdx == debugTdx){
        //    cout << "aTdx " << aTdx << endl;
        //}
        if (mesh.LabelTriangles[aTdx] == 1) {
            // It would be nice, if in future there is no dependency on the element ordering...
            //cout <<  aTdx << endl;
            //cout << "L " << mesh.LabelTriangles[aTdx] << endl;
            //if (mesh.LabelTriangles[aTdx] == 1) {

            //[DEBUG]
            /*
            //if (false){
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
            //[End DEBUG]

            //[DEBUG]
            /*
            doubleVec_tozero(DEBUG_termTotalLocal, 9);
            doubleVec_tozero(DEBUG_termTotalNonloc, 9);
            */
            //[End DEBUG]

            // Get index of ansatz functions in matrix compute_A.-------------------
            if (mesh.is_DiscontinuousGalerkin) {
                // Discontinuous Galerkin
                //aDGdx[0] = (dVertex+1)*aTdx+1; aDGdx[1] = (dVertex+1)*aTdx+2; aDGdx[2] = (dVertex+1)*aTdx+3;
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
            //doubleVec_tozero(termf, mesh.dVertex); // Initialize Buffer
            //compute_f(aT, quadRule, mesh, termf); // Integrate and fill buffer
            // Add content of buffer to the right side.
            //for (a = 0; a < mesh.dVertex; a++) {
                // Assembly happens in the interior of Omega only, so we throw away some values
            //    if (mesh.is_DiscontinuousGalerkin || (aAdx[a] < mesh.L_Omega)) {
            //        #pragma omp atomic update
            //        fd[aAdx[a]] += termf[a];
            //    }
            //}

            // Of course some uneccessary computation happens but only for some verticies of thos triangles which lie
            // on the boundary. This saves us from the pain to carry the information (a) into the integrator compute_f.

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
                // 3 at max in 2D, 4 in 3D.
                for (int j = 0; j < mesh.nNeighbours; j++) {
                    // The next valid neighbour is our candidate for the inner Triangle b.
                    int bTdx = NTdx[j];

                    // Check how many neighbours sTdx has. It can be 3 at max.
                    // In order to be able to store the list as contiguous array we fill up the empty spots with the number nE
                    // i.e. the total number of Triangles (which cannot be an index).
                    if (bTdx < mesh.nE) {

                        // Check whether bTdx is already visited.
                        if (visited[bTdx] == 0) {
                            // Prepare Triangle information bTE and bTdet ------------------
                            initializeTriangle(bTdx, mesh, bT);
                            //cout << aTdx << ", " << bTdx << endl;

                            // Retriangulation and integration ------------------------
                            if (mesh.is_DiscontinuousGalerkin) {
                                // Discontinuous Galerkin
                                for (int jj = 0; jj < mesh.dVertex; jj++) {
                                    bDGdx[jj] = mesh.dVertex * bTdx + jj;
                                }
                                bAdx = bDGdx;
                            } else {
                                // Get (pointer to) index of basis function (in Continuous Galerkin)
                                bAdx = &mesh.Triangles(0, bTdx);
                                // The first entry (index 0) of each row in triangles contains the Label of each point!
                                // Hence, in order to get an pointer to the three Triangle idices, which we need here
                                // we choose &Triangles[4*aTdx+1];
                            }
                            // Assembly of matrix ---------------------------------------
                            doubleVec_tozero(termLocal, mesh.dVertex * mesh.dVertex*mesh.outdim*mesh.outdim); // Initialize Buffer
                            doubleVec_tozero(termNonloc, mesh.dVertex * mesh.dVertex*mesh.outdim*mesh.outdim); // Initialize Buffer
                            // Compute integrals and write to buffer
                            integrate(aT, bT, quadRule, mesh, conf, is_firstbfslayer, termLocal, termNonloc);
                            // [DEBUG]
                            //doubleVec_add(termLocal, DEBUG_termTotalLocal, DEBUG_termTotalLocal, 9);
                            //doubleVec_add(termNonloc, DEBUG_termTotalNonloc, DEBUG_termTotalNonloc, 9);
                            // [End DEBUG]

                            // If bT interacts it will be a candidate for our BFS, so it is added to the queue

                            //[DEBUG]
                            /*
                            if (aTdx == 0 && bTdx == 45){
                            //if (true){

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
                            //if (aTdx == debugTdx){
                            //    cout << bTdx << ", ";
                            //}
                            if (doubleVec_any(termNonloc, mesh.dVertex * mesh.dVertex) ||
                                doubleVec_any(termLocal, mesh.dVertex * mesh.dVertex)) {
                                queue.push(bTdx);
                                // In order to speed up the integration we only check whether the integral
                                // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                                // However, this works only if we can guarantee that interacting triangles do actually
                                // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere on its support for example.
                                // The effect (in speedup) of this more precise criterea depends on delta and meshsize.

                                // Copy buffer into matrix. Again solutions which lie on the boundary are ignored (in Continuous Galerkin)
                                //printf("aTdx %i \nbTdx %i\n", aTdx, bTdx);
                                for (int a = 0; a < mesh.dVertex*mesh.outdim; a++) {
                                    // [x 3]
                                    // for (int a = 0; a < mesh.dVertex*mesh.outdim; a++){ ...

                                    if (mesh.is_DiscontinuousGalerkin || (aAdx[a/mesh.outdim] < mesh.nV_Omega)) {
                                        for (int b = 0; b < mesh.dVertex*mesh.outdim; b++) {
                                            // [x 4]
                                            // for (int b = 0; b < mesh.dVertex*mesh.outdim; b++){ ...

                                            // INDEX CHECK
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

                                            // [x 5]
                                            Adx =  (mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim) * mesh.K +
                                                    mesh.outdim*aAdx[b/mesh.outdim] + b%mesh.outdim;
                                            //Adx = aAdx[a]*mesh.K + aAdx[b];

                                            // [x 6]
                                            // termLocal and termNonloc come with larger dimension already..
                                            // Ad[Adx] += termLocal[mesh.dVertex * a + b] * weight;
#pragma omp critical
                                            {
                                                Ad[Adx] += termLocal[mesh.dVertex * mesh.outdim * a + b] * weight;
                                            }

                                            // [6]
                                            Adx =   (mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim) * mesh.K +
                                                     mesh.outdim*bAdx[b/mesh.outdim] + b%mesh.outdim;
                                            //Adx = aAdx[a]*mesh.K + bAdx[b];
#pragma omp critical
                                            {
                                                Ad[Adx] += -termNonloc[mesh.dVertex * mesh.outdim * a + b] * weight;
                                            }

                                            //if (aTdx == debugTdx){
                                            //    cout << termLocal[mesh.dVertex * a + b] << ", ";
                                            //    cout << -termNonloc[mesh.dVertex * a + b] << ", ";
                                            //}

                                            // Caution: -----------------------------------------------
                                            // If k does not match the key of any element in the container,
                                            // the []-method inserts a new element with that key and
                                            // returns a reference to its mapped value.
                                            // >> This unnecessarily eats up memory, if you want to read only. Use find().
                                        }
                                    }
                                }
                            }// End if (doubleVec_any(termNonloc, ...)
                            //if (aTdx == debugTdx){
                            //    cout << endl;
                            //}

                        }// End if BFS (visited[bTdx] == 0)
                        // Mark bTdx as visited
                        visited[bTdx] = 1;
                    }// End if BFS (bTdx < mesh.nE)
                }//End for loop BFS (j = 0; j < mesh.nNeighbours; j++)
                is_firstbfslayer = false;
            }//End while loop BFS (!queue.empty())
            //}// End if Label of (aTdx == 1)
        }// End if LabelTriangles == 1
    }// End parallel for

    int nnz_start = 0;

    }// End pragma omp parallel

    cout << "K_Omega " << mesh.K_Omega << endl;
    cout << "K " << mesh.K << endl;
    int nnz_total = static_cast<int>(Ad.size());
    arma::vec values_all(nnz_total);
    arma::umat indices_all(2, nnz_total);
    cout << "Total nnz" << nnz_total << endl;


    //cout << "Thread "<< omp_get_thread_num() << ", start  " << nnz_start << endl;
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

    //cout << "K_Omega " << mesh.K_Omega << endl;
    //cout << "K " << mesh.K << endl;
    //cout << "NNZ " << nnz_total << endl;
    //cout << arma::max(indices_all.row(1)) << endl;
    arma::sp_mat sp_Ad(true, indices_all, values_all, mesh.K, mesh.K_Omega);
    sp_Ad.save(conf.path_spAd);
    //cout << "Data saved." << endl;
}// End function par_system

void par_forcing(MeshType &mesh, QuadratureType &quadRule, ConfigurationType &conf) {
    arma::vec fd(mesh.K_Omega, arma::fill::zeros);

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
            if (mesh.LabelTriangles[aTdx] == 1) {
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
                    if (mesh.is_DiscontinuousGalerkin || (aAdx[a/mesh.outdim] < mesh.nV_Omega)) {
                        #pragma omp atomic update
                        fd[mesh.outdim*aAdx[a/mesh.outdim] + a%mesh.outdim] += termf[a]*weight;
                    }
                }// end for rhs

            }// end outer if (mesh.LabelTriangles[aTdx] == 1)
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
