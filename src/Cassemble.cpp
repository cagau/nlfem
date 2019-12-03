#include <math.h>
#include <omp.h>
#include <queue>
#include <iostream>
#include <Cassemble.h>
#include <armadillo>
using namespace std;

// Declaration of internal helper functions ----------------------------------------------------------------------------
struct ElementStruct{
    arma::vec matE;
    double * E;
    int dim;
    long label;
    double absDet;
    int signDet;
};
typedef ElementStruct ElementType;

// Model ---------------------------------------------------------------------------------------------------------------
static double model_f(const double *);
static double model_kernel(const double * x, const long labelx, const double * y, const long labely, const double sqdelta);
static void model_basisFunction(const double * p, double *psi_vals);

// Integration ---------------------------------------------------------------------------------------------------------
static int placePointOnCap(const double * y_predecessor, const double * y_current,
        const double * x_center, const double sqdelta, const double * TE,
        const double * nu_a, const double * nu_b, const double * nu_c,
        const double orientation, const int Rdx, double * R);

// Math functions ------------------------------------------------------------------------------------------------------
static void solve2x2(const double *, const double *, double *);                 // Solve 2x2 System with LU
static void rightNormal(const double * y0, const double * y1, const double orientation, double * normal);
// Matrix operations (via * only) ------------------------------------------------------------------------------------
// Double
static double absDet(const double * E);                                         // Compute determinant
static double signDet(const double * E);
static void baryCenter(const double * E, double * bary);                        // Bary Center
static void toRef(const double * E, const double * phys_x, double * ref_p);     // Pull point to Reference Element (performs 2x2 Solve)
static void toPhys(const double * E, const double * p, double * out_x);         // Push point to Physical Element

// Vector operations -------------------------------------------------------------------------------------------------
// Double
static double vec_sqL2dist(const double * x, const double * y, const int len);      // L2 Distance
static double vec_dot(const double * x, const double * y, const int len);           // Scalar Product
static int doubleVec_any(const double * vec, const int len);                        // Any
static void doubleVec_tozero(double *, int);               // Reset to zero
static void doubleVec_subtract(const double * vec1, const double * vec2, double * out, const int len);
static void doubleVec_midpoint(const double * vec1, const double * vec2, double * midpoint, const int len);
static void doubleVec_scale(const double lambda, const double * vec, double * out, const int len);
static void doubleVec_add(const double * vec1, const double * vec2, double * out, const int len);
static void doubleVec_copyTo(const double * input, double * output, const int len);

// Long
static int longVec_all(const long *, const int);                       // All
static int longVec_any(const long *, const int);                       // Any

// Int
static void intVec_tozero(int *, const int);                     // Reset to zero

// Scalar ----------------------------------------------------------
static double absolute(const double);                                  // Get absolute value
static double scal_sqL2dist(const double x, const double y);           // L2 Distance

static void initializeTriangle( const int Tdx, const MeshType & mesh, ElementType & T);


// Model -----------------------------------------f-------------------------------------------------------------------

// Define Right side compute_f
double model_f(const double * x){
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

double model_kernel(const double * x, const long labelx, const double * y, const long labely, const double sqdelta){
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

void model_basisFunction(const double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
}

// Integration ---------------------------------------------------------------------------------------------------------


void integrate(     const ElementType aT,
                    const ElementType bT,
                    const QuadratureType & quadRule,
                    const arma::mat psix,
                    const double sqdelta,
                    double * termLocal, double * termNonloc){
    int k=0, a=0, b=0;
    double x[2];
    double innerLocal=0;
    double innerNonloc[3];

    int i=0, rTdx=0, Rdx=0;
    double ker=0, rTdet=0;
    double physical_quad[2];
    double reference_quad[2];
    double psi_value[3];
    double reTriangle_list[9*3*2];
    bool is_placePointOnCap;

    //[DEBUG]
    //printf("\nouterInt_full----------------------------------------\n");
    for (k=0; k<quadRule.nPx; k++){
        toPhys(aT.E, &(quadRule.Px[2*k]), x);
        //printf("\nInner Integral, Iterate %i\n", k);
        //printf("\Physical x [%17.16e, %17.16e]\n",  x[0], x[1]);
        //innerInt_retriangulate(x, aT, bT, quadRule, sqdelta, &innerLocal, innerNonloc);

        innerLocal = 0.0;
        doubleVec_tozero(innerNonloc, 3);
        is_placePointOnCap = true;

        Rdx = retriangulate(x, bT.E, sqdelta, reTriangle_list, is_placePointOnCap); // innerInt_retriangulate

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
            for (i=0; i<quadRule.nPy; i++){
                // Push quadrature point P[i] to physical triangle reTriangle_list[rTdx] (of the retriangulation!)
                toPhys(&reTriangle_list[2 * 3 * rTdx], &(quadRule.Py[2*i]), physical_quad);
                // Determinant of Triangle of retriangulation
                rTdet = absDet(&reTriangle_list[2 * 3 * rTdx]);
                // inner Local integral with ker
                innerLocal += model_kernel(x, aT.label, physical_quad, bT.label, sqdelta) * quadRule.dy[i] * rTdet; // Local Term
                // Pull resulting physical point ry to the (underlying!) reference Triangle aT.
                toRef(bT.E, physical_quad, reference_quad);
                // Evaluate ker on physical quad (note this is ker')
                ker = model_kernel(physical_quad, bT.label, x, aT.label, sqdelta);
                // Evaluate basis function on resulting reference quadrature point
                model_basisFunction(reference_quad, psi_value);
                for (b=0; b<3; b++){
                    innerNonloc[b] += psi_value[b] * ker * quadRule.dy[i] * rTdet; // Nonlocal Term
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

        //printf("Local %17.16e\n", innerLocal);
        //printf("Nonloc [%17.16e, %17.16e, %17.16e] \n", innerNonloc[0], innerNonloc[1], innerNonloc[2]);
        for (b=0; b<3; b++){
            for (a=0; a<3; a++){
                termLocal[3*a+b] += aT.absDet * psix(a,k) * psix(b,k) * quadRule.dx[k] * innerLocal; //innerLocal
                termNonloc[3*a+b] += aT.absDet * psix(a,k) * quadRule.dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}

// Normal which looks to the right w.r.t the vector from y0 to y1.
void rightNormal(const double * y0, const double * y1, const double orientation, double * normal){
    normal[0] = y1[1] - y0[1];
    normal[1] = y0[0] - y1[0];
    doubleVec_scale(orientation, normal, normal, 2);
}

bool inTriangle(const double * y_new, const double * p, const double * q, const double * r,
        const double *  nu_a, const double * nu_b, const double * nu_c){
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

int placePointOnCap(const double * y_predecessor, const double * y_current,
                    const double * x_center, const double sqdelta, const double * TE,
                    const double * nu_a, const double * nu_b, const double * nu_c,
                    const double orientation, const int Rdx, double * R){
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

int retriangulate(const double * x_center, const double * TE, const double sqdelta, double * out_reTriangle_list, const int is_placePointOnCap){
        // C Variables and Arrays.
        int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0;
        double v=0, lam1=0, lam2=0, term1=0, term2=0;
        double nu_a[2], nu_b[2], nu_c[2]; // Normals
        arma::vec p(2);
        arma::vec q(2);
        arma::vec a(2);
        arma::vec b(2);
        arma::vec y1(2);
        arma::vec y2(2);
        arma::vec vec_x_center(x_center, 2);
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

            doubleVec_copyTo(&TE[2*edgdx0], &p[0], 2);
            doubleVec_copyTo(&TE[2*edgdx1], &q[0], 2);

            a = q - vec_x_center;
            b = p - q;

            if (vec_sqL2dist(&p[0], x_center, 2) <= sqdelta){
                doubleVec_copyTo(&p[0], &R[2*Rdx], 2);
                is_onEdge = false; // This point does not lie on the edge.
                Rdx += 1;
            }
            // PQ-Formula to solve quadratic problem
            v = pow( dot(a, b), 2) - (dot(a, a) - sqdelta) * dot(b, b);
            // If there is no sol to the quadratic problem, there is nothing to do.
            if (v >= 0){
                term1 = - dot(a, b) / dot(b, b);
                term2 = sqrt(v) / dot(b, b);

                // Vieta's Formula for computing the roots
                if (term1 > 0){
                    lam1 = term1 + term2;
                    lam2 = 1/lam1*(dot(a, a) - sqdelta);
                } else {
                    lam2 = term1 - term2;
                    lam1 = 1/lam2*(dot(a, a) - sqdelta) / dot(b, b);
                }
                y1 = lam1*b + q;
                y2 = lam2*b + q;

                // Check whether the first lambda "lies on the Triangle".
                if ((0 <= lam1) && (lam1 <= 1)){
                    is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;
                    // Check whether the predecessor lied on the edge
                    if (is_onEdge && is_placePointOnCap){
                        Rdx += placePointOnCap(&R[2*(Rdx-1)], &y1[0], x_center, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                    }
                    // Append y1
                    doubleVec_copyTo(&y1[0], &R[2*Rdx], 2);
                    is_onEdge = true; // This point lies on the edge.
                    Rdx += 1;
                }
                // Check whether the second lambda "lies on the Triangle".
                if ((0 <= lam2) && (lam2 <= 1) && (scal_sqL2dist(lam1, lam2) > 0)){
                    is_firstPointLiesOnVertex = is_firstPointLiesOnVertex && (bool)Rdx;

                    // Check whether the predecessor lied on the edge
                    if (is_onEdge && is_placePointOnCap){
                        Rdx += placePointOnCap(&R[2*(Rdx-1)], &y2[0], x_center, sqdelta, TE, nu_a, nu_b, nu_c, orientation, Rdx, R);
                    }
                    // Append y2
                    doubleVec_copyTo(&y2[0], &R[2*Rdx], 2);
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

void compute_f(     const ElementType & aT,
                    const QuadratureType &quadRule,
                    const arma::mat psi,
                    double * termf){
    int i,a;
    double x[2];

    for (a=0; a<3; a++){
        for (i=0; i<quadRule.nPx; i++){
            toPhys(aT.E, &(quadRule.Px[2 * i]), &x[0]);
            termf[a] += psi(a, i) * model_f(&x[0]) * aT.absDet * quadRule.dx[i];
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
void par_assemble(  const MeshType& mesh,
                    const QuadratureType& quadRule,
                    double * Ad,
                    double * fd
                    ){
    int aTdx, h=0;
    //const int dVertex = dim + 1;
    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!

    arma::Mat<double> armAd(Ad, mesh.K, mesh.K_Omega, false, true);
    arma::Mat<double> armapsix(mesh.dVertex, quadRule.nPx);

    for(h=0; h<quadRule.nPx; h++){
        // This works due to Column Major ordering of Armadillo Matricies!
        model_basisFunction(&quadRule.Px[mesh.dim*h], & armapsix[mesh.dVertex*h]);
    }

    // Unfortunately Armadillo thinks in Column-Major order. So everything is transposed!
    // Contains one row more than number of verticies as label information is contained here
    const arma::Mat<long> Triangles(mesh.ptrTriangles, mesh.dVertex+1, mesh.J);
    // Contains number of direct neighbours of an element + 1 (itself).
    const arma::Mat<long> Neighbours(mesh.ptrNeighbours, mesh.dVertex, mesh.J);
    const arma::Mat<double> Verts(mesh.ptrVerts, mesh.dim, mesh.L);

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
    for (aTdx=0; aTdx<mesh.J_Omega; aTdx++)
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
        if(mesh.is_DiscontinuousGalerkin){
            // Discontinuous Galerkin
            //aDGdx[0] = (dVertex+1)*aTdx+1; aDGdx[1] = (dVertex+1)*aTdx+2; aDGdx[2] = (dVertex+1)*aTdx+3;
            for (j=0; j<mesh.dVertex; j++){
                aDGdx[j] =  mesh.dVertex*aTdx+j;
            }
            aAdx = aDGdx;
        } else {
            // Continuous Galerkin
            // The first entry (index 0) of each row in triangles contains the Label of each point!
            // Hence, in order to get an pointer to the three Triangle idices, which we need here
            // we choose &Triangles[4*aTdx+1];
            aAdx = &Triangles(1, aTdx);
        }
        // Prepare Triangle information aTE and aTdet ------------------
        initializeTriangle(aTdx, mesh, aT);

        // Assembly of right side ---------------------------------------
        // We unnecessarily integrate over vertices which might lie on the boundary of Omega for convenience here.
        doubleVec_tozero(termf, mesh.dVertex); // Initialize Buffer
        compute_f(aT, quadRule, armapsix, termf); // Integrate and fill buffer

        // Add content of buffer to the right side.
        for (a=0; a<mesh.dVertex; a++){
            // Assembly happens in the interior of Omega only, so we throw away some values
            // Again, Triangles contains the labels as first entry! Hence, we start with a=1 here!
            // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
            if (mesh.is_DiscontinuousGalerkin || (aAdx[a] < mesh.L_Omega)){
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
        while (!queue.empty()){
            // Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
            sTdx = queue.front();
            queue.pop();
            // Get all the neighbours of sTdx.
            NTdx =  &Neighbours(0, sTdx);
            // Run through the list of neighbours.
            // 3 at max in 2D, 4 in 3D.
            for (j=0; j<mesh.dVertex; j++){
                // The next valid neighbour is our candidate for the inner Triangle b.
                bTdx = NTdx[j];

                // Check how many neighbours sTdx has. It can be 3 at max.
                // In order to be able to store the list as contiguous array we fill up the empty spots with the number J
                // i.e. the total number of Triangles (which cannot be an index).
                if (bTdx < mesh.J){

                    // Prepare Triangle information bTE and bTdet ------------------
                    initializeTriangle(bTdx, mesh, bT);

                    // Check whether bTdx is already visited.
                    if (visited[bTdx]==0){

                        // Retriangulation and integration ------------------------
                        if (mesh.is_DiscontinuousGalerkin){
                            // Discontinuous Galerkin
                            for (j=0; j<mesh.dVertex; j++){
                                bDGdx[j] =  mesh.dVertex*bTdx+j;
                            }
                            bAdx = bDGdx;
                        } else {
                            // Get (pointer to) intex of basis function (in Continuous Galerkin)
                            bAdx = &Triangles(1, bTdx);
                            // The first entry (index 0) of each row in triangles contains the Label of each point!
                            // Hence, in order to get an pointer to the three Triangle idices, which we need here
                            // we choose &Triangles[4*aTdx+1];
                        }
                        // Assembly of matrix ---------------------------------------
                        doubleVec_tozero(termLocal, mesh.dVertex * mesh.dVertex); // Initialize Buffer
                        doubleVec_tozero(termNonloc, mesh.dVertex * mesh.dVertex); // Initialize Buffer
                        // Compute integrals and write to buffer
                        integrate(aT, bT, quadRule, armapsix, mesh.sqdelta, termLocal, termNonloc);
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


                        if (doubleVec_any(termNonloc, mesh.dVertex * mesh.dVertex) || doubleVec_any(termLocal, mesh.dVertex * mesh.dVertex)){
                            queue.push(bTdx);
                            // In order to speed up the integration we only check whether the integral
                            // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                            // However, this works only if we can guarantee that interacting triangles do actually
                            // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere on its support for example.
                            // The effect (in speedup) of this more precise criterea depends on delta and meshsize.

                            // Copy buffer into matrix. Again solutions which lie on the boundary are ignored (in Continuous Galerkin)
                            for (a=0; a<mesh.dVertex; a++){
                            // Note: aAdx[a] == Triangles[4*aTdx+1 + a]!
                                if  (mesh.is_DiscontinuousGalerkin || (aAdx[a] < mesh.L_Omega)){
                                    for (b=0; b<mesh.dVertex; b++){
                                        #pragma omp atomic update
                                        armAd(aAdx[b], aAdx[a]) += termLocal[mesh.dVertex*a+b];
                                        #pragma omp atomic update
                                        armAd(bAdx[b], aAdx[a]) -= termNonloc[mesh.dVertex*a+b];
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
    }
}

//void initializeTriangle(const arma::mat & Verts, const arma::Mat<long> & Triangles, const int Tdx, const int dim, ElementType & T){
void initializeTriangle( const int Tdx, const MeshType & mesh, ElementType & T){
    // Copy coordinates of Triange b to bTE.
    int j, k, Vdx;
    //T.matE = arma::vec(dim*(dim+1));
    for (k=0; k<mesh.dim+1; k++) {
        Vdx = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx + k+1];
        for (j=0; j<mesh.dim; j++){
            //T.matE[mesh.dim * k + j] = mesh.Verts(j, mesh.Triangles(k+1, Tdx));
            T.matE[mesh.dim * k + j] = mesh.ptrVerts[ mesh.dim*Vdx + j];
        }
    }
    // Initialize Struct
    T.E = T.matE.memptr();
    T.absDet = absDet(T.E);
    T.signDet = signDet(T.E);
    //T.label = Triangles(0, Tdx);
    T.label = mesh.ptrTriangles[(mesh.dVertex+1)*Tdx];
    //T.dim = dim;
    T.dim = mesh.dim;
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

void solve2x2(const double * A, const double * b, double * x){
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

double absDet(const double * E){
    double M[2][2];
    int i=0;
    for (i=0; i< 2; i++){
        M[i][0] = E[2*1+i] - E[2*0+i];
        M[i][1] = E[2*2+i] - E[2*0+i];
    }
    return absolute(M[0][0]*M[1][1] - M[0][1]*M[1][0]);
}

double signDet(const double * E){
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

void baryCenter(const double * E, double * bary){
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
void baryCenter_polygone(const double * P, const int nVerticies, double * bary){
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

void toPhys(const double * E, const double * p, double * out_x){
    int i=0;
    for (i=0; i<2;i++){
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i];
    }
}

void toRef(const double * E, const double * phys_x, double * ref_p){
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
int doubleVec_any(const double * vec, const int len){
    int i=0;
    for (i=0; i < len; i++){
        if (vec[i] != 0){
            return 1;
        }
    }
    return 0;
}

double vec_dot(const double * x, const double * y, const int len){
    double r=0;
    int i=0;
    for (i=0; i<len; i++){
        r += x[i]*y[i];
    }
    return r;
}

double vec_sqL2dist(const double * x, const double * y, const int len){
    double r=0;
    int i=0;
    for (i=0; i<len; i++){
        r += pow((x[i] - y[i]), 2);
    }
    return r;
}

void doubleVec_tozero(double * vec, const int len){
    int i=0;
    for (i=0; i< len; i++){
        vec[i]  = 0;
    }
}

void doubleVec_midpoint(const double * vec1, const double * vec2, double * midpoint, const int len){
    int i = 0;
    for (i=0; i < len; i++){
        midpoint[i]  = (vec1[i] + vec2[i])/2;
    }
}

void doubleVec_subtract(const double * vec1, const double * vec2, double * out, const int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec1[i] - vec2[i];
    }
}

void doubleVec_add(const double * vec1, const double * vec2, double * out, const int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec1[i] + vec2[i];
    }
}

void doubleVec_scale(const double lambda, const double * vec, double * out, const int len){
    int i=0;
    for (i=0; i < len; i++){
        out[i]  = vec[i]*lambda;
    }
}

void doubleVec_copyTo(const double * input, double * output, const int len){
    int i=0;
    for (i=0; i<len; i++){
        output[i] = input[i];
    }
}
// Long

int longVec_all(const long * vec, const int len){
    int i=0;
    for (i=0; i<len; i++){
        if (vec[i] == 0){
            return 0;
        }
    }
    return 1;
}

int longVec_any(const long * vec, const int len){
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
void intVec_tozero(int * vec, const int len){
    int i=0;
    for (i=0; i< len; i++){
        vec[i]  = 0;
    }
}
// Scalar --------------------------------------------------------

double absolute(const double value){
    if (value < 0){
        return - value;
    } else {
        return value;
    }
}

double scal_sqL2dist(const double x, const double y){
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