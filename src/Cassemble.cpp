#include <math.h>
#include <omp.h>
#include <queue>
#include <iostream>
#include <Cassemble.h>
using namespace std;

// Model -----------------------------------------f----------------------------------------------------------------

// Define Right side compute_f
static double model_f(double * x){
        return 1.0;
/*
        if ((-.2 < x[0] && x[0] < .2) && (-2 < x[1] && x[1] < .2) )
        {
            return 1.0;
        } else {
            return 0.;
        }
*/
}

static double model_kernel(double * x, long labelx, double * y, long labely, double sqdelta){
    return 4 / (M_PI * pow(sqdelta, 2));
}

/*
double model_kernel_(double * x, long labelx, double * y, long labely, double sqdelta){
    double dist;
    long label;

    label = 10*labelx + labely;
    dist = vec_sqL2dist(y, x, 2);
    if (dist >= sqdelta) {
        cout << "Error in model_kernel. Distance smaller delta not expected." << endl;
        cout << dist << endl;
        abort();
    }
    if (label <= 12) {
        return 0.01 * 3. / (4*pow(sqdelta, 2));
    } else if (label>=21){
        return (100 *  3. / (4*pow(sqdelta, 2))) * (1 - (dist/sqdelta) );
    } else if (label == 13){
        return 0.0;
    } else {
        cout << "No such case " << endl;
        abort();
    }
}
*/

static void model_basisFunction(double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
}

// Integration ---------------------------------------------------------------------------------------------------------
static void outerInt_full(double * aTE, double aTdet, long labela, double * bTE, double bTdet, long labelb,
                    double * Px, int nPx, double * dx,
                    double * Py, int nPy, double * dy,
                    double * psix, double * psiy,
                    double sqdelta,
                    double * termLocal, double * termNonloc){
    int k=0, a=0, b=0;
    double x[2];
    double innerLocal=0;
    double innerNonloc[3];
    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k=0; k<nPx; k++){
        toPhys(aTE, &Px[2 * k], x);
        //printf("\nInner Integral, Iterate %i\n", k);
        //printf("\Physical x [%17.16e, %17.16e]\n",  x[0], x[1]);
        innerInt_retriangulate(x, labela, bTE, labelb, Py, nPy, dy, sqdelta, &innerLocal, innerNonloc);
        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2]);
        for (b=0; b<3; b++){
            for (a=0; a<3; a++){
                termLocal[3*a+b] += aTdet * psix[nPx*a+k] * psix[nPx*b+k] * dx[k] * innerLocal; //innerLocal
                termNonloc[3*a+b] += aTdet * psix[nPx*a+k] * dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}


static void innerInt_retriangulate(double * x, long labela, double * bTE, long labelb, double * P, int nP, double * dy, double sqdelta, double * innerLocal, double * innerNonloc){
    int i=0, rTdx=0, b=0, Rdx=0;
    double ker=0, rTdet=0;
    double physical_quad[2];
    double reference_quad[2];
    double psi_value[3];
    double reTriangle_list[9*3*2];
    bool is_placePointOnCap;

    innerLocal[0] = 0.0;
    doubleVec_tozero(innerNonloc, 3);
    is_placePointOnCap = true;
    Rdx = retriangulate(x, bTE, sqdelta, reTriangle_list, is_placePointOnCap); // innerInt_retriangulate
    //[DEBUG]
    //printf("Retriangulation Rdx %i\n", Rdx);
    for (i=0;i<Rdx;i++){
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i], reTriangle_list[2 * 3 * i+1]);
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i+2], reTriangle_list[2 * 3 * i+3]);
        //printf("[%17.16e, %17.16e]\n", reTriangle_list[2 * 3 * i+4], reTriangle_list[2 * 3 * i+5]);
        //printf("absDet %17.16e\n", absDet(&reTriangle_list[2 * 3 * i]));
    }
    if (Rdx == 0){
        return;
    }
    //printf("\nInner Integral\n");
    for (rTdx=0; rTdx < Rdx; rTdx++){
    //printf("rTdx %i \n",rTdx);
        for (i=0; i<nP; i++){
            // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
            toPhys(&reTriangle_list[2 * 3 * rTdx], &P[2*i], physical_quad);
            // Determinant of Triangle of retriangulation
            rTdet = absDet(&reTriangle_list[2 * 3 * rTdx]);
            // inner Local integral with ker
            *innerLocal += model_kernel(x, labela, physical_quad, labelb, sqdelta) * dy[i] * rTdet; // Local Term
            // Pull resulting physical point ry to the (underlying!) reference Triangle aT.
            toRef(bTE, physical_quad, reference_quad);
            // Evaluate ker on physical quad (note this is ker')
            ker = model_kernel(physical_quad, labelb, x, labela, sqdelta);
            // Evaluate basis function on resulting reference quadrature point
            model_basisFunction(reference_quad, psi_value);
            for (b=0; b<3; b++){
                innerNonloc[b] += psi_value[b] * ker * dy[i] * rTdet; // Nonlocal Term
            }
        //[DEBUG]
        //printf("i %i \n",i);
        //printf("GAM %17.16e\n", ker * dy[i] * rTdet);
        //printf("Basis0 %17.16e\n", psi_value[0]);
        //printf("Basis1 %17.16e\n", psi_value[1]);
        //printf("Basis2 %17.16e\n", psi_value[2]);
        }
       //printf("Chris: v0 %17.16e\nv1 %17.16e\nv2 %17.16e\n", innerNonloc[0], innerNonloc[1], innerNonloc[2]);
       //printf("Chris: v %17.16e\n", innerLocal);
   }
}

// Normal which looks to the right w.r.t the vector from y0 to y1.
static void rightNormal(double * y0, double * y1, double orientation, double * normal){
    normal[0] = y1[1] - y0[1];
    normal[1] = y0[0] - y1[0];
    doubleVec_scale(orientation, normal, normal, 2);
}

static bool inTriangle(double * y_new, double * p, double * q, double * r, double *  nu_a, double * nu_b, double * nu_c){
    bool a, b, c;
    double vec[2];

    doubleVec_subtract(y_new, p, vec, 2);
    a = vec_dot(nu_a, vec , 2) >= 0;

    doubleVec_subtract(y_new, q, vec, 2);
    b = vec_dot(nu_b, vec , 2) >= 0;

    doubleVec_subtract(y_new, r, vec, 2);
    c = vec_dot(nu_c, vec , 2) >= 0;

    return a && b && c;
}

static int placePointOnCap(double * y_predecessor, double * y_current, double * x_center, double sqdelta, double * TE, double * nu_a, double * nu_b, double * nu_c, double orientation, int Rdx, double * R){
    // Place a point on the cap.
    //y_predecessor = &R[2*(Rdx-1)];
    double y_new[2], s_midpoint[2], s_projectionDirection[2];
    double scalingFactor;

    doubleVec_midpoint(y_predecessor, y_current, s_midpoint, 2);
    // Note, this yields the left normal from y_predecessor to y0
    rightNormal(y_current, y_predecessor, orientation, s_projectionDirection);
    // Simple way
    scalingFactor = sqrt( sqdelta / vec_dot(s_projectionDirection, s_projectionDirection, 2));
    doubleVec_scale(scalingFactor, s_projectionDirection, s_projectionDirection, 2);
    doubleVec_add(x_center, s_projectionDirection, y_new, 2);

    if ( inTriangle(y_new, &TE[0], &TE[2], &TE[4], nu_a, nu_b, nu_c)){
        // Append y_new (Point on the cap)
        doubleVec_copyTo(y_new, &R[2*Rdx], 2);
        return 1;
    } else {
        return 0;
    }
}

int retriangulate(double * x_center, double * TE, double sqdelta, double * out_reTriangle_list, int is_placePointOnCap){
        // C Variables and Arrays.
        int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0;
        double v=0, lam1=0, lam2=0, term1=0, term2=0;
        double nu_a[2], nu_b[2], nu_c[2]; // Normals
        double p[2];
        double q[2];
        double a[2];
        double b[2];
        double y1[2];
        double y2[2];
        double orientation;

        bool is_onEdge=false, is_firstPointLiesOnVertex=true;
        // The upper bound for the number of required points is 9
        // Hence 9*3 is an upper bound to encode all resulting triangles
        // Hence we can hardcode how much space needs to bee allocated
        // (This upper bound is thight! Check Christian Vollmann's thesis for more information.)

        double R[9*2]; // Vector containing all intersection points.
        doubleVec_tozero(R, 9*2);

        // Compute Normals of the Triangle
        orientation = -signDet(TE);
        rightNormal(&TE[0], &TE[2], orientation, nu_a);
        rightNormal(&TE[2], &TE[4], orientation, nu_b);
        rightNormal(&TE[4], &TE[0], orientation, nu_c);

        for (k=0; k<3; k++){
            edgdx0 = k;
            edgdx1 = (k+1) % 3;

            doubleVec_copyTo(&TE[2*edgdx0], p, 2);
            doubleVec_copyTo(&TE[2*edgdx1], q, 2);
            doubleVec_subtract(q, x_center, a, 2);
            doubleVec_subtract(p, q, b, 2);

            if (vec_sqL2dist(p, x_center, 2) <= sqdelta){
                doubleVec_copyTo(p, &R[2*Rdx], 2);
                is_onEdge = false; // This point does not lie on the edge.
                Rdx += 1;
            }
            // PQ-Formula to solve quadratic problem
            v = pow(vec_dot(a, b, 2), 2) - (vec_dot(a, a, 2) - sqdelta) * vec_dot(b, b, 2);
            // If there is no sol to the quadratic problem, there is nothing to do.
            if (v >= 0){
                term1 = -vec_dot(a, b, 2) / vec_dot(b, b, 2);
                term2 = sqrt(v) / vec_dot(b, b, 2);
                // Vieta's Formula for computing the roots
                if (term1 > 0){
                    lam1 = term1 + term2;
                    // (vec_dot(a, a, 2) - sqdelta) / vec_dot(b, b, 2) is the q, of the pq-Formula
                    lam2 = 1/lam1*(vec_dot(a, a, 2) - sqdelta) / vec_dot(b, b, 2);
                } else {
                    lam2 = term1 - term2;
                    lam1 = 1/lam2*(vec_dot(a, a, 2) - sqdelta) / vec_dot(b, b, 2);
                }
                //lam1 = term1 + term2;
                //lam2 = term1 - term2;
                // Set Intersection Points
                doubleVec_scale(lam1, b, y1, 2);
                doubleVec_add(y1, q, y1, 2);
                doubleVec_scale(lam2, b, y2, 2);
                doubleVec_add(y2, q, y2, 2);

                // Check whether the first lambda "lies on the Triangle".
                if ((0 <= lam1) && (lam1 <= 1)){
                    is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;
                    // Check whether the predecessor lied on the edge
                    if (is_onEdge && is_placePointOnCap){
                        Rdx += placePointOnCap(&R[2*(Rdx-1)], y1, x_center, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                    }
                    // Append y1
                    doubleVec_copyTo(y1, &R[2*Rdx], 2);
                    is_onEdge = true; // This point lies on the edge.
                    Rdx += 1;
                }
                // Check whether the second lambda "lies on the Triangle".
                if ((0 <= lam2) && (lam2 <= 1) && (scal_sqL2dist(lam1, lam2) > 0)){
                    is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;

                    // Check whether the predecessor lied on the edge
                    if (is_onEdge && is_placePointOnCap){
                        Rdx += placePointOnCap(&R[2*(Rdx-1)], y2, x_center, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                    }

                    // Append y2
                    doubleVec_copyTo(y2, &R[2*Rdx], 2);
                    is_onEdge = true; // This point lies on the edge.
                    Rdx += 1;
                }
            }
        }
        //[DEBUG]
        //(len(RD)>1) cares for the case that either the first and the last point lie on an endge
        // and there is no other point at all.
        //shift=1;
        if (is_onEdge && (!is_firstPointLiesOnVertex && Rdx > 1) && is_placePointOnCap){
            Rdx += placePointOnCap(&R[2*(Rdx-1)], &R[0], x_center, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
        }

        // Construct List of Triangles from intersection points
        if (Rdx < 3){
            // In this case the content of the array out_RE will not be touched.
            return 0;
        } else {

            for (k=0; k < (Rdx - 2); k++){
                for (i=0; i<2; i++){
                    // i is the index which runs first, then h (which does not exist here), then k
                    // hence if we increase i, the *-index (of the pointer) inreases in the same way.
                    // if we increase k, there is quite a 'jump'
                    out_reTriangle_list[2 * (3 * k + 0) + i] = R[i];
                    out_reTriangle_list[2 * (3 * k + 1) + i] = R[2*(k + 1) + i];
                    out_reTriangle_list[2 * (3 * k + 2) + i] = R[2*(k + 2) + i];
                }
            }
            // Excessing the bound out_Rdx will not lead to an error but simply to corrupted data!

            return Rdx - 2; // So that, it acutally contains the number of triangles in the retriangulation
        }
}

// Compute A and f -----------------------------------------------------------------------------------------------------

static void compute_f(double * aTE,
                    double aTdet,
                    double * P,
                    int nP,
                    double * dx,
                    double * psi,
                    double * termf){
    int i,a;
    double x[2];

    for (a=0; a<3; a++){
        for (i=0; i<nP; i++){
            toPhys(aTE, &P[2 * i], &x[0]);
            termf[a] += psi[nP*a + i] * model_f(&x[0]) * aTdet * dx[i];
        }
    }
}

//void compute_A(double * aTE, double aTdet, long labela, double * bTE, double bTdet, long labelb,
//                    double * P,
//                    int nP,
//                    double * dx,
//                    double * dy,
//                    double * psi,
//                    double sqdelta,
//                    double * termLocal, double * termNonloc){
//    outerInt_full(&aTE[0], aTdet, labela, &bTE[0], bTdet, labelb, P, nP, dx, dy, psi, sqdelta, termLocal, termNonloc);
//}

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

void par_evaluateMass(double * vd, double * ud, long * Triangles, double * Verts, int K_Omega, int J_Omega, int nP, double * P, double * dx){
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
                            //vd[aAdx[a]*K_Omega + aAdx[b]] += psi[nP*a+j]*psi[nP*b+j]*aTdet*dx[j];
                            // Evaluation
                            #pragma omp atomic update
                            vd[aAdx[a]] += psi[nP*a+j]*psi[nP*b+j]*aTdet*dx[j]*ud[aAdx[b]];
                        }
                    }
                }
            }
        }
    }

}
// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(  double * Ad,
                    int K_Omega,
                    int K,
                    double * fd,
                    long * Triangles,
                    double * Verts,
                    // Number of Triangles and number of Triangles in Omega
                    int J, int J_Omega,
                    // Number of vertices (in case of CG = K and K_Omega)
                    int L, int L_Omega,
                    double * Px, int nPx, double * dx,
                    double * Py, int nPy, double * dy,
                    double sqdelta,
                    long * Neighbours,
                    int is_DiscontinuousGalerkin,
                    int is_NeumannBoundary
                   ) {
    int aTdx, h=0;
    double tmp_psi[3];


    double *psix = (double *) malloc(3*nPx*sizeof(double));
    double *psiy = (double *) malloc(3*nPy*sizeof(double));

    for(h=0; h<nPx; h++){
       model_basisFunction(&Px[2*h], &tmp_psi[0]);
       psix[nPx*0+h] = tmp_psi[0];
       psix[nPx*1+h] = tmp_psi[1];
       psix[nPx*2+h] = tmp_psi[2];
    }

    for(h=0; h<nPy; h++){
       model_basisFunction(&Py[2*h], &tmp_psi[0]);
       psiy[nPy*0+h] = tmp_psi[0];
       psiy[nPy*1+h] = tmp_psi[1];
       psiy[nPy*2+h] = tmp_psi[2];
    }

    #pragma omp parallel
    {
    // General Loop Indices ---------------------------------------
    int i=0, j=0, bTdx=0;

    // Breadth First Search --------------------------------------
    int *visited = (int *) malloc(J*sizeof(int));

    // Loop index of current outer triangle in BFS
    int sTdx=0;
    // Queue for Breadth first search
    queue<int> queue;
    // List of visited triangles
    long *NTdx;
    // Determinant of Triangle a and b.
    double aTdet, bTdet;
    // Vector containing the coordinates of the vertices of a Triangle
    double aTE[2*3];
    double bTE[2*3];
    // Integration information ------------------------------------
    // Loop index of basis functions
    int a=0, b=0;
    long labela=0, labelb=0;
    // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
    long * aAdx;
    long * bAdx;
    long aDGdx[3]; // Index for discontinuous Galerkin
    long bDGdx[3];

    // Buffers for integration solutions
    double termf[3];
    double termLocal[3*3];
    double termNonloc[3*3];
    //[DEBUG]
    /*
    double DEBUG_termTotalLocal[3*3];
    double DEBUG_termTotalNonloc[3*3];
    */
    //[End DEBUG]


    #pragma omp for
    for (aTdx=0; aTdx<J_Omega; aTdx++)
    {

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
        if(is_DiscontinuousGalerkin){
            // Discontinuous Galerkin
            aDGdx[0] = 3*aTdx; aDGdx[1] = 3*aTdx+1; aDGdx[2] = 3*aTdx+2;
            aAdx = aDGdx;
        } else {
            // Continuous Galerkin
            // The first entry (index 0) of each row in triangles contains the Label of each point!
            // Hence, in order to get an pointer to the three Triangle idices, which we need here
            // we choose &Triangles[4*aTdx+1];
            aAdx = &Triangles[4*aTdx+1];
        }
        // Prepare Triangle information aTE and aTdet ------------------
        // Copy coordinates of Triange a to aTE.
        // this is done fore convenience only, actually those are unnecessary copies!
        for (j=0; j<2; j++){
            aTE[2*0+j] = Verts[2*Triangles[4*aTdx+1] + j];
            aTE[2*1+j] = Verts[2*Triangles[4*aTdx+2] + j];
            aTE[2*2+j] = Verts[2*Triangles[4*aTdx+3] + j];
        }
        // compute Determinant
        aTdet = absDet(aTE);
        labela = Triangles[4*aTdx];

        // Assembly of right side ---------------------------------------
        // We unnecessarily integrate over vertices which might lie on the boundary of Omega for convenience here.
        doubleVec_tozero(termf, 3); // Initialize Buffer
        compute_f(aTE, aTdet, Px, nPx, dx, psix, termf); // Integrate and fill buffer

        // Add content of buffer to the right side.
        for (a=0; a<3; a++){
            // Assembly happens in the interior of Omega only, so we throw away some values
            // Again, Triangles contains the labels as first entry! Hence, we start with a=1 here!
            // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
            if (is_DiscontinuousGalerkin || (aAdx[a] < L_Omega)){
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
        intVec_tozero(&visited[0], J);

        // Check whether BFS is over.
        while (!queue.empty()){
            // Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
            sTdx = queue.front();
            queue.pop();
            // Get all the neighbours of sTdx.
            NTdx =  &Neighbours[4*sTdx];
            // Run through the list of neighbours. (4 at max)
            for (j=0; j<4; j++){
                // The next valid neighbour is our candidate for the inner Triangle b.
                bTdx = NTdx[j];

                // Check how many neighbours sTdx has. It can be 4 at max. (Itself, and the three others)
                // In order to be able to store the list as contiguous array we fill up the empty spots with the number J
                // i.e. the total number of Triangles (which cannot be an index).
                if (bTdx < J){

                    // Prepare Triangle information bTE and bTdet ------------------
                    // Copy coordinates of Triange b to bTE.
                    // again this is done for convenience only, actually those are unnecessary copies!
                    for (i=0; i<2;i++){
                        bTE[2*0+i] = Verts[2*Triangles[4*bTdx+1] + i];
                        bTE[2*1+i] = Verts[2*Triangles[4*bTdx+2] + i];
                        bTE[2*2+i] = Verts[2*Triangles[4*bTdx+3] + i];
                    }
                    bTdet = absDet(&bTE[0]);
                    labelb = Triangles[4*bTdx];
                    // Check whether bTdx is already visited.
                    if (visited[bTdx]==0){

                        // Retriangulation and integration ------------------------
                        if (is_DiscontinuousGalerkin){
                            // Discontinuous Galerkin
                            bDGdx[0] = 3*bTdx; bDGdx[1] = 3*bTdx+1; bDGdx[2] = 3*bTdx+2;
                            bAdx = bDGdx;
                        } else {
                            // Get (pointer to) intex of basis function (in Continuous Galerkin)
                            bAdx = &Triangles[4*bTdx+1];
                            // The first entry (index 0) of each row in triangles contains the Label of each point!
                            // Hence, in order to get an pointer to the three Triangle idices, which we need here
                            // we choose &Triangles[4*aTdx+1];
                        }
                        // Assembly of matrix ---------------------------------------
                        doubleVec_tozero(termLocal, 3 * 3); // Initialize Buffer
                        doubleVec_tozero(termNonloc, 3 * 3); // Initialize Buffer
                        // Compute integrals and write to buffer
                        outerInt_full(aTE, aTdet, labela, bTE, bTdet, labelb, Px, nPx, dx, Py, nPy, dy, psix, psiy, sqdelta, termLocal, termNonloc);
                        // [DEBUG]
                        //doubleVec_add(termLocal, DEBUG_termTotalLocal, DEBUG_termTotalLocal, 9);
                        //doubleVec_add(termNonloc, DEBUG_termTotalNonloc, DEBUG_termTotalNonloc, 9);
                        // [End DEBUG]

                        // If bT interacts it will be a candidate for our BFS, so it is added to the queue

                        //[DEBUG]
                        /*
                        //if (aTdx == 9 && bTdx == 911){
                        if (false){

                        printf("aTdx %i\ndet %17.16e, label %i \n", aTdx, aTdet, labela);
                        printf ("aTE\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n", aTE[0],aTE[1],aTE[2],aTE[3],aTE[4],aTE[5]);
                        printf("bTdx %i\ndet %17.16e, label %i \n", bTdx, bTdet, labelb);
                        printf ("bTE\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n[%17.16e, %17.16e]\n", bTE[0],bTE[1],bTE[2],bTE[3],bTE[4],bTE[5]);

                        cout << endl << "Local Term" << endl ;
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[0], termLocal[1], termLocal[2]);
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[3], termLocal[4], termLocal[5]);
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termLocal[6], termLocal[7], termLocal[8]);

                        cout << endl << "Nonlocal Term" << endl ;
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[0], termNonloc[1], termNonloc[2]);
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[3], termNonloc[4], termNonloc[5]);
                        printf ("[%17.16e, %17.16e, %17.16e] \n", termNonloc[6], termNonloc[7], termNonloc[8]);

                        abort();

                        }
                        */
                        //[End DEBUG]


                        if (doubleVec_any(termNonloc, 3 * 3) || doubleVec_any(termLocal, 3 * 3)){
                            queue.push(bTdx);
                            // In order to speed up the integration we only check whether the integral
                            // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                            // However, this works only if we can guarantee that interacting triangles do actually
                            // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere on its support for example.
                            // The effect (in speedup) of this more precise criterea depends on delta and meshsize.

                            // Copy buffer into matrix. Again solutions which lie on the boundary are ignored (in Continuous Galerkin)
                            for (a=0; a<3; a++){
                            // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
                                if  (is_DiscontinuousGalerkin || (aAdx[a] < L_Omega)){
                                    for (b=0; b<3; b++){
                                        #pragma omp atomic update
                                        Ad[aAdx[a]*K + aAdx[b]] += termLocal[3*a+b];
                                        #pragma omp atomic update
                                        Ad[aAdx[a]*K + bAdx[b]] -= termNonloc[3*a+b];
                                    }
                                }
                            }
                        }

                    }
                    // Mark bTdx as visited
                    visited[bTdx] = 1;
                }
            }
        }
    }
    free(visited);
    }
    free(psix);
    free(psiy);
}

double compute_area(double * aTE, double aTdet, long labela, double * bTE, double bTdet, long labelb, double * P, int nP, double * dx, double sqdelta){
    double areaTerm=0.0;
    int rTdx, Rdx;
    double * x;
    double physical_quad[2];
    double reTriangle_list[9*3*2];

    x = &P[0];
    toPhys(aTE, x, physical_quad);
    Rdx = retriangulate(physical_quad, bTE, sqdelta, reTriangle_list, true);
    for (rTdx=0; rTdx < Rdx; rTdx++){
        areaTerm += absDet(&reTriangle_list[2*3*rTdx])/2;
    }
    return areaTerm;
}

// Math functions ------------------------------------------------------------------------------------------------------

static void solve2x2(double * A, double * b, double * x){
    int dx0 = 0, dx1 = 1;
    double l=0, u=0;

    // Column Pivot Strategy
    if (absolute(A[0]) < absolute(A[2])){
        dx0 = 1;
        dx1 = 0;
    }

    // Check invertibility
    if (A[2*dx0] == 0){
        // raise LinAlgError("in solve2x2. Matrix not invertible.")
        cout << "in solve2x2. Matrix not invertible." << endl;
        abort();
    }

    // LU Decomposition
    l = A[2*dx1]/A[2*dx0];
    u = A[2*dx1+1] - l*A[2*dx0+1];

    // Check invertibility
    if (u == 0){
        // raise LinAlgError("in solve2x2. Matrix not invertible.")
        cout << "in solve2x2. Matrix not invertible." << endl;
        abort();
    }

    // LU Solve
    x[1] = (b[dx1] - l*b[dx0])/u;
    x[0] = (b[dx0] - A[2*dx0+1]*x[1])/A[2*dx0];
    return;
}


// Matrix operations (working with strides only) --------------------------------

static double absDet(double * E){
    double M[2][2];
    int i=0;
    for (i=0; i< 2; i++){
        M[i][0] = E[2*1+i] - E[2*0+i];
        M[i][1] = E[2*2+i] - E[2*0+i];
    }
    return absolute(M[0][0]*M[1][1] - M[0][1]*M[1][0]);
}

static double signDet(double * E){
    double M[2][2], det;
    int i=0;
    for (i=0; i< 2; i++){
        M[i][0] = E[2*1+i] - E[2*0+i];
        M[i][1] = E[2*2+i] - E[2*0+i];
    }
    det = (M[0][0]*M[1][1] - M[0][1]*M[1][0]);
    if (det > 0){
        return 1.;
    } else if ( det < 0){
        return -1.;
    } else {
        cout << "Warning in signDet(): Determinant is 0" << endl;
        return 0.0;
    }
}

static void baryCenter(double * E, double * bary){
    int i=0;
    bary[0] = 0;
    bary[1] = 0;
    for  (i=0; i< 3; i++){
        bary[0] += E[2*i+0];
        bary[1] += E[2*i+1];
    }
    bary[0] = bary[0]/3;
    bary[1] = bary[1]/3;
}
static void baryCenter_polygone(double * P, int nVerticies, double * bary){
    int k=0;
    bary[0] = 0;
    bary[1] = 0;
    for (k=0; k<nVerticies; k++){
        bary[0] += P[2*k+0];
        bary[1] += P[2*k+1];
    }
    bary[0] = bary[0]/nVerticies;
    bary[1] = bary[1]/nVerticies;
}

static void toPhys(double * E, double * p, double * out_x){
    int i=0;
    for (i=0; i<2;i++){
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i];
    }
}

static void toRef(double * E, double * phys_x, double * ref_p){
    double M[2*2];
    double b[2];

    M[0] = E[2] - E[0];
    M[1] = E[4] - E[0];
    M[2] = E[3] - E[1];
    M[3] = E[5] - E[1];

    b[0] = phys_x[0] - E[0];
    b[1] = phys_x[1] - E[1];

    solve2x2(&M[0], &b[0], &ref_p[0]);
    return;
}
// Vector operations ---------------------------------------------
// Double

// Check whether any, or all elements of a vector are zero --------------
static int doubleVec_any(double * vec, int len){
    int i=0;
    for (i=0; i < len; i++){
        if (vec[i] != 0){
            return 1;
        }
    }
    return 0;
}

static double vec_dot(double * x, double * y, int len){
    double r=0;
    int i=0;
    for (i=0; i<len; i++){
        r += x[i]*y[i];
    }
    return r;
}

static double vec_sqL2dist(double * x, double * y, int len){
    double r=0;
    int i=0;
    for (i=0; i<len; i++){
        r += pow((x[i] - y[i]), 2);
    }
    return r;
}

static void doubleVec_tozero(double * vec, int len){
    int i=0;
    for (i=0; i< len; i++){
        vec[i]  = 0;
    }
}

static void doubleVec_midpoint(double * vec1, double * vec2, double * midpoint, int len){
    int i = 0;
    for (i=0; i < len; i++){
        midpoint[i]  = (vec1[i] + vec2[i])/2;
    }
}

static void doubleVec_subtract(double * vec1, double * vec2, double * out, int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec1[i] - vec2[i];
    }
}

static void doubleVec_add(double * vec1, double * vec2, double * out, int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec1[i] + vec2[i];
    }
}

static void doubleVec_scale(double lambda, double * vec, double * out, int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec[i]*lambda;
    }
}

static void doubleVec_copyTo(double * input, double * output, int len){
    int i=0;
    for (i=0; i<len; i++){
        output[i] = input[i];
    }
}
// Long

static int longVec_all(long * vec, int len){
    int i=0;
    for (i=0; i<len; i++){
        if (vec[i] == 0){
            return 0;
        }
    }
    return 1;
}

static int longVec_any(long * vec, int len){
    int i=0;
    for (i=0; i<len; i++){
            if (vec[i] != 0){
                return 1;
            }
    }
    return 0;
}

// Int

// Set Vectors to Zero -------------------------------------------------
static void intVec_tozero(int * vec, int len){
    int i=0;
    for (i=0; i< len; i++){
        vec[i]  = 0;
    }
}
// Scalar --------------------------------------------------------

static double absolute(double value){
    if (value < 0){
        return - value;
    } else {
        return value;
    }
}

static double scal_sqL2dist(double x, double y){
    return pow((x-y), 2);
}

//[DEBUG] Christians order routine
/*
void bubbleSort2(double * keyArray, int rows, int * indexList){
  // We assume keyArray has 2 Columns!
  // Sorting is done w.r.t first col, then second col.
  int n, i, cols=2;
  int buffer;
  for (n=rows; n>1; --n){
    for (i=0; i<n-1; ++i){
      if (keyArray[cols*i] > keyArray[cols*(i+1)]){
        buffer = indexList[i];
        indexList[i] = indexList[i+1];
        indexList[i+1] = buffer;

      } else if ((keyArray[cols*i] == keyArray[cols*(i+1)]) && (keyArray[cols*i+1] > keyArray[cols*(i+1)+1])){
        buffer = indexList[i];
        indexList[i] = indexList[i+1];
        indexList[i+1] = buffer;
      } // End if
    } // End inner for
  } // End outer for
}

void relativePosition(double * origin, double * refvec, double * x, double * angle, double * length){
    double vector[2], normalized[2];
    double lenVector, dotProd, diffProd;
    doubleVec_subtract(x, origin, vector, 2);
    lenVector = sqrt(vec_sqL2dist(x, origin, 2));
    if (lenVector == 0){
        angle[0] = -M_PI;
        length[0] = 0;
        return;
    }
    doubleVec_scale(1/lenVector, vector, normalized, 2);
    dotProd = vec_dot(normalized, refvec, 2);
    diffProd = refvec[1] * normalized[0] - refvec[0] * normalized[1];
    angle[0] = atan2(diffProd, dotProd);
    length[0] = lenVector;
}

void order(double * pointList, int lenList, double * orderedPointList){
    double * origin;
    double refvec[2];
    double angle[9], length[9];
    int i;

    origin = &pointList[0];
    doubleVec_subtract(&pointList[2*1], origin, refvec, 2);

    for (i=0; i<lenList; i++){
        relativePosition(origin, refvec, &pointList[2*i], &angle[i], &length[i]);
        //printf("Angle %17.16e\nLength %17.16e\n", angle[i], length[i]);
    }
}

*/
// [END DEBUG]