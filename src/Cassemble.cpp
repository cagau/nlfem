#include <Cassemble.h>
#include <cmath>
#include <queue>
#include <iostream>
#include <integration.cpp>
#include <mathhelpers.cpp>
#include <model.cpp>
#include <armadillo>
using namespace std;

void lookup_configuration(ConfigurationType & conf){

    // Lookup right hand side ------------------------------------------------------------------------------------------
    cout << "Right hand side: " << conf.model_f << endl;
    if (conf.model_f == "linear") {
        model_f = f_linear;
    } else if (conf.model_f == "linear3D") {
        model_f = f_linear3D;
    } else if (conf.model_f == "constant"){
        model_f = f_constant;
    } else {
        cout << "Error in par:assemble. Right hand side: " << conf.model_f << " is not implemented." << endl;
        abort();
    }

    // Lookup kernel ---------------------------------------------------------------------------------------------------
    cout << "Kernel: " << conf.model_kernel << endl;
    if (conf.model_kernel == "constant"){
        model_kernel = kernel_constant;
    } else if (conf.model_kernel == "labeled") {
        model_kernel = kernel_labeled;
    } else if (conf.model_kernel == "constant3D") {
        model_kernel = kernel_constant3D;
    } else {
        cout << "Error in par:assemble. Kernel " << conf.model_kernel << " is not implemented." << endl;
        abort();
    }

    // Lookup integration method  --------------------------------------------------------------------------------------
    cout << "Integration Method: " << conf.integration_method << endl;
    if (conf.integration_method == "baryCenter"){
        integrate = integrate_baryCenter;
    } else if (conf.integration_method == "baryCenterRT") {
        integrate = integrate_baryCenterRT;
        printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
    }  else if (conf.integration_method == "retriangulate") {
        integrate = integrate_retriangulate;
        printf("With caps: %s\n", conf.is_placePointOnCap ? "true" : "false");
    } else {
        cout << "Error in par:assemble. Integration method " << conf.integration_method <<
             " is not implemented." << endl;
        abort();
    }
}

void initializeTriangle( const int Tdx, const MeshType & mesh, ElementType & T){
    // Copy coordinates of Triange b to bTE.

    // Tempalte of Triangle Point data.
    // 2D Case, a, b, c are the vertices of a triangle
    // T.E -> | a1 | a2 | b1 | b2 | c1 | c2 |
    // Hence, if one wants to put T.E into col major order matrix it would be of shape\
    //                             | a1 | b1 | c1 |
    // M(mesh.dim, mesh.dVerts) =  | a2 | b2 | c2 |
    //

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
}

void constructAdjaciencyGraph(const int dim, const int nE, const long * elements, long * neighbours){
    int aTdx=0;
    const int dVertex = dim+1;
    const int sqdVertex = pow(dVertex, 2);
    #pragma omp parallel
    {
    int i=0, nEqualVerts=0, nNeighsFound=0, bTdx=0;
    #pragma omp for
    for (aTdx=0; aTdx < nE; aTdx++) {
        for (i = 0; i < dVertex; i++) {
            neighbours[aTdx * dVertex + i] = static_cast<long>(nE);
        }
        nNeighsFound = 0;

        for(bTdx=0; bTdx<nE; bTdx++) {
            nEqualVerts = 0;
            for (i = 0; i < sqdVertex; i++) {
                nEqualVerts += (elements[aTdx*dVertex + (i % dVertex)] == elements[bTdx*dVertex + static_cast<int>(i / dVertex)]);
            }
            //cout << nEqualVerts << endl;
            if (nEqualVerts == dim){
                neighbours[aTdx*dVertex + nNeighsFound] = bTdx;
                nNeighsFound+=1;
                if (nNeighsFound==dVertex){
                    bTdx = nE;
                }
            }
        }
    }
}

}
// Compute A and f -----------------------------------------------------------------------------------------------------
void compute_f(     const ElementType & aT,
                    const QuadratureType &quadRule,
                    const MeshType & mesh,
                    double * termf){
    int i,a;
    double x[mesh.dim];

    for (a=0; a<mesh.dVertex; a++){
        for (i=0; i<quadRule.nPx; i++){
            toPhys(aT.E, &(quadRule.Px[mesh.dim * i]),  mesh,&x[0]);
            termf[a] += quadRule.psix(a, i) * model_f(&x[0]) * aT.absDet * quadRule.dx[i];
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
                 double *P, double *dx, const int dim) {
    int k=0, kk=0;
    const int dVerts = dim+1;


    double tmp_psi[dVerts];
    double *psi = (double *) malloc((dVerts)*nP*sizeof(double));

    for(k=0; k<nP; k++){
       model_basisFunction(&P[dim*k], &tmp_psi[0]);
       for (kk=0; kk<dVerts; kk++) {
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
        int a=0, b=0, j=0, jj=0, aTdx=0;
        double aTE[dim*(dVerts)];
        #pragma omp for
        for (aTdx=0; aTdx < J; aTdx++){
            if (ElementLabels[aTdx] == 1) {
                // Get index of ansatz functions in matrix compute_A.-------------------
                // Continuous Galerkin
                aAdx = &Elements[dVerts* aTdx];
                // Discontinuous Galerkin
                // - Not implemented -

                // Prepare Triangle information aTE and aTdet ------------------
                // Copy coordinates of Triange a to aTE.
                // this is done fore convenience only, actually those are unnecessary copies!
                for (jj=0; jj<dVerts; jj++){
                    for (j = 0; j < dim; j++) {
                        aTE[dim * jj + j] = Verts[dim *aAdx[jj] + j];
                    }
                    //aTE[2 * 0 + j] = Verts[2 * Elements[4 * aTdx + 1] + j];
                    //aTE[2 * 1 + j] = Verts[2 * Elements[4 * aTdx + 2] + j];
                    //aTE[2 * 2 + j] = Verts[2 * Elements[4 * aTdx + 3] + j];
                }
                // compute Determinant
                aTdet = absDet(&aTE[0], dim);

                for (a = 0; a < dVerts; a++) {
                    if (aAdx[a] < K_Omega) {
                        for (b = 0; b < dVerts; b++) {
                            if (aAdx[b] < K_Omega) {
                                for (j = 0; j < nP; j++) {
                                    // Evaluation
                                    #pragma omp atomic update
                                    vd[aAdx[a]] += psi[nP * a + j] * psi[nP * b + j] * aTdet * dx[j] * ud[aAdx[b]];
                                }
                            }
                        }
                    }
                }
            }
        }
    } // Pragma Omp Parallel
}

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(double *ptrAd, const int K_Omega, const int K, double *fd, const long *ptrTriangles,
                  const long *ptrLabelTriangles, const double *ptrVerts, const int J, const int J_Omega, const int L,
                  const int L_Omega, const double *Px, const int nPx, const double *dx, const double *Py, const int nPy,
                  const double *dy, const double sqdelta, const long *ptrNeighbours, const int is_DiscontinuousGalerkin,
                  const int is_NeumannBoundary, const string str_model_kernel, const string str_model_f,
                  const string str_integration_method, const int is_PlacePointOnCap, const int dim) {

    MeshType mesh = {K_Omega, K, ptrTriangles, ptrLabelTriangles, ptrVerts, J, J_Omega,
                     L, L_Omega, sqdelta, ptrNeighbours, is_DiscontinuousGalerkin,
                     is_NeumannBoundary, dim, dim+1};
    QuadratureType quadRule = {Px, Py, dx, dy, nPx, nPy, dim};
    ConfigurationType conf = {str_model_kernel, str_model_f, str_integration_method, static_cast<bool>(is_PlacePointOnCap)};
    par_assemble( ptrAd, fd, mesh, quadRule, conf);
}

void par_assemble( double * ptrAd, double * fd,
                    MeshType & mesh,
                    QuadratureType & quadRule,
                    ConfigurationType & conf){

    printf("Function: par_assemble (generic)\n");
    printf("Mesh dimension: %i\n", mesh.dim);
    lookup_configuration(conf);
    printf("Quadrule outer: %i\n", quadRule.nPx);
    printf("Quadrule inner: %i\n", quadRule.nPy);

    int aTdx=0, h=0;
    //const int dVertex = dim + 1;
    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!
    arma::Mat<double> Ad(ptrAd, mesh.K, mesh.K_Omega, false, true);
    Ad.zeros();
    for(h=0; h<quadRule.nPx; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Px[mesh.dim*h], mesh,& quadRule.psix[mesh.dVertex * h]);
    }
    for(h=0; h<quadRule.nPy; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(& quadRule.Py[mesh.dim*h], mesh,& quadRule.psiy[mesh.dVertex * h]);
    }
    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!
    // Contains one row more than number of verticies as label information is contained here
    //const arma::Mat<long> Triangles(mesh.ptrTriangles, mesh.dVertex+1, mesh.J);
    //mesh.Triangles = arma::Mat<long>(mesh.ptrTriangles, mesh.dVertex+1, mesh.J);
    // Contains number of direct neighbours of an element + 1 (itself).
    //const arma::Mat<long> Neighbours(mesh.ptrNeighbours, mesh.dVertex, mesh.J);
    //const arma::Mat<double> Verts(mesh.ptrVerts, mesh.dim, mesh.L);

    #pragma omp parallel
    {
    // General Loop Indices ---------------------------------------
    int j=0, bTdx=0;

    // Breadth First Search --------------------------------------
    arma::Col<int> visited(mesh.J, arma::fill::zeros);


    // Loop index of current outer triangle in BFS
    int sTdx=0;
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
    // Loop index of basis functions
    int a=0, b=0;
    // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
    const long * aAdx;
    const long * bAdx;
    long aDGdx[mesh.dVertex]; // Index for discontinuous Galerkin
    long bDGdx[mesh.dVertex];

    // Buffers for integration solutions
    double termf[mesh.dVertex];
    double termLocal[mesh.dVertex*mesh.dVertex];
    double termNonloc[mesh.dVertex*mesh.dVertex];
    //[DEBUG]
    /*
    double DEBUG_termTotalLocal[3*3];
    double DEBUG_termTotalNonloc[3*3];
    */
    //[End DEBUG]


    #pragma omp for
    for (aTdx=0; aTdx<mesh.J; aTdx++) {
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
                for (j = 0; j < mesh.dVertex; j++) {
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
            doubleVec_tozero(termf, mesh.dVertex); // Initialize Buffer
            compute_f(aT, quadRule, mesh, termf); // Integrate and fill buffer
            // Add content of buffer to the right side.
            for (a = 0; a < mesh.dVertex; a++) {
                // Assembly happens in the interior of Omega only, so we throw away some values
                // Again, Triangles contains the labels as first entry! Hence, we start with a=1 here!
                // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
                if (mesh.is_DiscontinuousGalerkin || (aAdx[a] < mesh.L_Omega)) {
#pragma omp atomic update
                    fd[aAdx[a]] += termf[a];
                }
            }
            // Of course some uneccessary computation happens but only for some verticies of thos triangles which lie
            // on the boundary. This saves us from the pain to carry the information (a) into the integrator compute_f.

            // BFS -------------------------------------------------------------
            // Intialize search queue with current outer triangle
            queue.push(aTdx);
            // Initialize vector of visited triangles with 0
            visited.zeros();

            // Check whether BFS is completed.
            while (!queue.empty()) {
                // Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
                sTdx = queue.front();
                queue.pop();
                // Get all the neighbours of sTdx.
                NTdx = &mesh.Neighbours(0, sTdx);
                // Run through the list of neighbours.
                // 3 at max in 2D, 4 in 3D.
                for (j = 0; j < mesh.dVertex; j++) {
                    // The next valid neighbour is our candidate for the inner Triangle b.
                    bTdx = NTdx[j];
                    //bTdx = 45;

                    // Check how many neighbours sTdx has. It can be 3 at max.
                    // In order to be able to store the list as contiguous array we fill up the empty spots with the number J
                    // i.e. the total number of Triangles (which cannot be an index).
                    if (bTdx < mesh.J) {

                        // Prepare Triangle information bTE and bTdet ------------------
                        initializeTriangle(bTdx, mesh, bT);

                        // Check whether bTdx is already visited.
                        if (visited[bTdx] == 0) {

                            // Retriangulation and integration ------------------------
                            if (mesh.is_DiscontinuousGalerkin) {
                                // Discontinuous Galerkin
                                for (j = 0; j < mesh.dVertex; j++) {
                                    bDGdx[j] = mesh.dVertex * bTdx + j;
                                }
                                bAdx = bDGdx;
                            } else {
                                // Get (pointer to) intex of basis function (in Continuous Galerkin)
                                bAdx = &mesh.Triangles(0, bTdx);
                                // The first entry (index 0) of each row in triangles contains the Label of each point!
                                // Hence, in order to get an pointer to the three Triangle idices, which we need here
                                // we choose &Triangles[4*aTdx+1];
                            }
                            // Assembly of matrix ---------------------------------------
                            doubleVec_tozero(termLocal, mesh.dVertex * mesh.dVertex); // Initialize Buffer
                            doubleVec_tozero(termNonloc, mesh.dVertex * mesh.dVertex); // Initialize Buffer
                            // Compute integrals and write to buffer
                            integrate(aT, bT, quadRule, mesh, conf, termLocal, termNonloc);
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
                                for (a = 0; a < mesh.dVertex; a++) {
                                    // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
                                    if (mesh.is_DiscontinuousGalerkin || (aAdx[a] < mesh.L_Omega)) {
                                        for (b = 0; b < mesh.dVertex; b++) {
#pragma omp atomic update
                                            Ad(aAdx[b], aAdx[a]) += termLocal[mesh.dVertex * a + b];
#pragma omp atomic update
                                            Ad(bAdx[b], aAdx[a]) -= termNonloc[mesh.dVertex * a + b];
                                        }
                                    }
                                }
                            }// End if (doubleVec_any(termNonloc, ...)

                        }// End if BFS (visited[bTdx] == 0)
                        // Mark bTdx as visited
                        visited[bTdx] = 1;
                    }// End if BFS (bTdx < mesh.J)
                }//End for loop BFS (j = 0; j < mesh.dVertex; j++)
            }//End while loo BFS (!queue.empty())
            //}// End if Label of (aTdx == 1)
        }// End if LabelTriangles == 1
    }// End parallel for
    }// End pragma omp parallel
}// End function par_assemble

// [DEBUG] _____________________________________________________________________________________________________________
/*
double compute_area(double * aTE, double aTdet, long labela, double * bTE, double bTdet, long labelb, double * P, int nP, double * dx, double sqdelta){
    double areaTerm=0.0;
    int rTdx, Rdx;
    double * x;
    double physical_quad[2];
    double reTriangle_list[9*3*2];
    const MeshType mesh = {K_Omega, K, ptrTriangles, ptrVerts, J, J_Omega,
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
