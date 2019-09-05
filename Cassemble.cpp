#include <math.h>
#include <omp.h>
#include <queue>
#include <iostream>
#include "Cassemble.h"

using namespace std;

// Model ---------------------------------------------------------------------------------------------------------

// Define Right side compute_f
static double model_f(double * x){
        return 1.0;
}

static double model_kernel(double * x, double * y, double sqdelta){
    // Nonconstant Kernel does not yet work
    // pow(1-vec_sqL2dist(x,y, 2)/sqdelta, 2)
    return 4 / (M_PI * pow(sqdelta, 2));
}

static void model_basisFunction(double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
}

static void inNbhd(double * aTE, double * bTE, double sqdelta, long * M){
    /*
    Check whether two triangles interact w.r.t :math:`L_{2}`-ball of size delta.
    Returns an array of boolen values. Compares the barycenter of bT with the vertices of Ea.
    */

    int i=0;
    double bary[2];

    baryCenter(bTE, &bary[0]);
    for (i=0; i<3; i++){
        M[i] = (vec_sqL2dist(&aTE[2 * i], &bary[0], 2) <= sqdelta);
    }
    return;
}

// Integration ---------------------------------------------------------------------------------------------------------

static void outerInt_full(double * aTE, double aTdet,
                        double * bTE, double bTdet,
                        double * P,
                        int nP,
                        double * dx,
                        double * dy,
                        double * psi,
                        double sqdelta,
                        double * cy_termLocal,
                        double * cy_termNonloc){
    int k=0, Rdx=0, a=0, b=0;
    double x[2];
    double innerLocal=0;
    double innerNonloc[3];
    double RT[9*3*2];

    for (k=0; k<nP; k++){
        toPhys(&aTE[0], &P[2 * k], &x[0]);
        Rdx = retriangulate(&x[0], bTE, sqdelta, &RT[0]);
        innerInt_retriangulate(x, bTE, P, nP, dy, sqdelta, Rdx, &RT[0], &innerLocal, &innerNonloc[0]);
        for (b=0; b<3; b++){
            for (a=0; a<3; a++){
                cy_termLocal[3*a+b] += aTdet * psi[nP*a+k] * psi[nP*b+k] * dx[k] * innerLocal; //innerLocal
                cy_termNonloc[3*a+b] += aTdet * psi[nP*a+k] * dx[k] * innerNonloc[b]; //innerNonloc
            }
        }
    }
}

static void innerInt_bary(double * x, double * T, double * P, int nP, double * dy, double sqdelta, int Rdx,
                                 double * RT, double * innerLocal, double * innerNonloc){
    // if Barycenter of T is to far away from x -> innerLocal = 0

    // Else
    // innerLocal = (ker @ dy) * T.absDet()
    // innerNonloc = (psi * ker @ dy)*T.absDet()
}

static void innerInt_retriangulate(double * x, double * T, double * P, int nP, double * dy, double sqdelta, int Rdx,
                                 double * RT, double * innerLocal, double * innerNonloc){
    int i=0, rTdx=0, b=0;
    double ker=0, rTdet=0;
    double ry[2];
    double rp[2];
    double psi_rp[3];

    innerLocal[0] = 0;
    for (b=0; b<3; b++){
        innerNonloc[b] = 0;
    }
    if (Rdx == 0){
        return;
    }

    for (rTdx=0; rTdx < Rdx; rTdx++){
        for (i=0; i<nP; i++){
            toPhys(&RT[2 * 3 * rTdx], &P[2 * i], &ry[0]);
            toRef(T, ry, rp);
            ker = model_kernel(&x[0], &ry[0], sqdelta);
            rTdet = absDet(&RT[2 * 3 * rTdx]);
            innerLocal[0] += (ker * dy[i]) * rTdet; // Local Term
            model_basisFunction(&rp[0], &psi_rp[0]);
            for (b=0; b<3; b++){
                innerNonloc[b] += (psi_rp[b] * ker * dy[i]) * rTdet; // Nonlocal Term
            }
        }
    }
}

static int retriangulate(double * x_center, double * TE, double sqdelta, double * out_RT){
        // C Variables and Arrays.
        int i=0, k=0, edgdx0=0, edgdx1=0, Rdx=0;
        double v=0, lam1=0, lam2=0, term1=0, term2=0;
        double c_p[2];
        double c_q[2];
        double c_a[2];
        double c_b[2];
        double c_y1[2];
        double c_y2[2];
        // An upper bound for the number of intersections between a circle and a triangle is 9
        // Hence we can hardcode how much space needs to bee allocated
        double c_R[9][2];
        // Hence 9*3 is an upper bound to encode all resulting triangles
        for (i=0; i<9; i++){
            for (k=0; k<2; k++){
                c_R[i][k] = 0.0;
            }
        }


        for (k=0; k<3; k++){
            edgdx0 = k;
            edgdx1 = (k+1) % 3;

            for (i=0; i<2; i++){
                c_p[i] = TE[2*edgdx0+i];
                c_q[i] = TE[2*edgdx1+i];
                c_a[i] = c_q[i] - x_center[i];
                c_b[i] = c_p[i] - c_q[i];
            }
            v = pow(vec_dot(&c_a[0], &c_b[0], 2), 2) - (vec_dot(&c_a[0], &c_a[0], 2) - sqdelta) * vec_dot(&c_b[0], &c_b[0], 2);

            if (v >= 0){
                term1 = -vec_dot(&c_a[0], &c_b[0], 2) / vec_dot(&c_b[0], &c_b[0], 2);
                term2 = sqrt(v) / vec_dot(&c_b[0], &c_b[0], 2);
                lam1 = term1 + term2;
                lam2 = term1 - term2;
                for (i=0; i<2; i++){
                    c_y1[i] = lam1*(c_p[i]-c_q[i]) + c_q[i];
                    c_y2[i] = lam2*(c_p[i]-c_q[i]) + c_q[i];
                }
                if (vec_sqL2dist(&c_p[0], &x_center[0], 2) <= sqdelta){
                    for (i=0;i<2;i++){
                        c_R[Rdx][i] = c_p[i];
                    }
                    Rdx += 1;
                }
                if ((0 <= lam1) && (lam1 <= 1)){
                    for (i=0;i<2;i++){
                        c_R[Rdx][i] = c_y1[i];
                    }
                    Rdx += 1;
                }
                if ((0 <= lam2) && (lam2 <= 1) && (scal_sqL2dist(lam1, lam2) > 0)){
                    for (i=0; i<2; i++){
                        c_R[Rdx][i] = c_y2[i];
                    }
                    Rdx += 1;
                }
            } else {
                if (vec_sqL2dist(c_p, &x_center[0], 2)  <= sqdelta){
                    for (i=0; i<2; i++){
                        c_R[Rdx][i] = c_p[i];
                    }
                    Rdx += 1;
                }
            }
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
                    out_RT[2 * (3 * k + 0) + i] = c_R[0][i];
                    out_RT[2 * (3 * k + 1) + i] = c_R[k + 1][i];
                    out_RT[2 * (3 * k + 2) + i] = c_R[k + 2][i];
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

static void compute_A(double * aTE, double aTdet, double * bTE, double bTdet,
                    double * P,
                    int nP,
                    double * dx,
                    double * dy,
                    double * psi,
                    double sqdelta, bool is_allInteract,
                    double * cy_termLocal, double * cy_termNonloc){

    int i=0, j=0, a, b;
    double innerIntLocal, innerIntNonloc;

    if (is_allInteract){
        for (a=0; a<3; a++){
            for (b=0; b<3; b++){
                for (i=0; i<nP; i++){
                    innerIntLocal=0;
                    innerIntNonloc=0;
                    for (j=0; j<nP; j++){
                        innerIntLocal += model_kernel(&P[2*i], &P[2*j], sqdelta) * dy[j];
                        innerIntNonloc += model_kernel(&P[2*i], &P[2*j], sqdelta) * dy[j] * psi[nP*b+j];
                    }
                    cy_termLocal[3*a+b] += psi[nP*a+i] * psi[nP*b+i] * innerIntLocal * dx[i] * aTdet*bTdet;
                    cy_termNonloc[3*a+b] += psi[nP*a+i] * innerIntNonloc * dx[i] * aTdet*bTdet;
                }
            }
        }
    } else {
        outerInt_full(&aTE[0], aTdet, &bTE[0], bTdet, P, nP, dx, dy, psi, sqdelta, cy_termLocal, cy_termNonloc);
    }
}

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------

static void par_assemble(  double * Ad,
                    int K,
                    double * fd,
                    long * c_Triangles,
                    double * c_Verts,
                    // Number of Triangles and number of Triangles in Omega
                    int J, int J_Omega,
                    // Number of vertices (in case of CG = K and K_Omega)
                    int L, int L_Omega,
                    int nP, double * P,
                    double * dx,
                    double * dy,
                    double sqdelta,
                    long * Neighbours
                   ) {
    int aTdx;

    // BFS ------------------------------------------------
    // Allocate Graph of Neighbours
    // Further definitions

    // General Loop Indices ---------------------------------------
    int i=0, j=0, bTdx=0;
    // Breadth First Search --------------------------------------
    // Loop index of current outer triangle in BFS
    int sTdx=0;
    // Queue for Breadth first search
    queue<int> c_queue;
    // List of visited triangles
    // np.ndarray[int, ndim=1, mode="c"] visited = py_visited
    // Matrix telling whether some vertex of Triangle a interactions with the baryCenter of Triangle b
    // np.ndarray[long, ndim=1, mode="c"] Mis_interact = py_Mis_interact
    int *visited;
    long *NTdx;
    long* Mis_interact;
    // Determinant of Triangle a and b.
    double aTdet, bTdet;
    // Vector containing the coordinates of the vertices of a Triangle
    double aTE[2*3];
    double bTE[2*3];
    // Integration information ------------------------------------
    // Loop index of basis functions
    int a=0, b=0, aAdxj =0;
    // (Pointer to) Vector of indices of Basisfuntions (Adx) for triangle a and b
    long * aAdx;
    long * bAdx;

    // Buffers for integration solutions
    double termf[3];
    double tmp_psi[3];
    double termLocal[3*3];
    double termNonloc[3*3];

    double *psi = (double *) malloc(nP*sizeof(double));

    for(j=0; j<nP; j++){
       model_basisFunction(&P[2*j], &tmp_psi[0]);
       psi[nP*0+j] = tmp_psi[0];
       psi[nP*1+j] = tmp_psi[1];
       psi[nP*2+j] = tmp_psi[2];
    }

    #pragma omp parallel private(termf, termLocal, termNonloc, aAdx, bAdx, a, b, aAdxj, aTE, bTE, aTdet, bTdet, NTdx, c_queue, sTdx, i, j, bTdx, visited, Mis_interact)
    {
    int *visited = (int *) malloc(J*sizeof(int));
    long *Mis_interact = (long *) malloc(3*sizeof(long));

    #pragma omp for
    for (aTdx=0; aTdx<J; aTdx++)
    {
        // Get index of ansatz functions in matrix compute_A.-------------------
        // Continuous Galerkin
        aAdx = &c_Triangles[3*aTdx];
        // Discontinuous Galerkin
        // - Not implemented -

        // Prepare Triangle information aTE and aTdet ------------------
        // Copy coordinates of Triange a to aTE.
        // this is done fore convenience only, actually those are unnecessary copies!
        for (j=0; j<2; j++){
            aTE[2*0+j] = c_Verts[2*c_Triangles[3*aTdx]   + j];
            aTE[2*1+j] = c_Verts[2*c_Triangles[3*aTdx+1] + j];
            aTE[2*2+j] = c_Verts[2*c_Triangles[3*aTdx+2] + j];
        }
        // compute Determinant
        aTdet = absDet(&aTE[0]);

        // Assembly of right side ---------------------------------------
        // We unnecessarily integrate over vertices which might lie on the boundary of Omega for convenience here.
        doubleVec_tozero(&termf[0], 3); // Initialize Buffer
        compute_f(&aTE[0], aTdet, &P[0], nP, dx, &psi[0], &termf[0]); // Integrate and fill buffer

        // Add content of buffer to the right side.
        //#pragma omp critical
        //{
            for (a=0; a<3; a++){
                // Assembly happens in the interior of Omega only, so we throw away some values
                if (c_Triangles[3*aTdx + a] < L_Omega){
                    aAdxj = aAdx[a];
                    #pragma omp atomic update
                    fd[aAdxj] += termf[a];
                }
            }
        //}
        // Of course some uneccessary computation happens but only for some verticies of thos triangles which lie
        // on the boundary. This saves us from the pain to carry the information (a) into the integrator compute_f.

        // BFS -------------------------------------------------------------
        // Intialize search queue with current outer triangle
        c_queue.push(aTdx);
        // Initialize vector of visited triangles with 0
        intVec_tozero(&visited[0], J);

        // Check whether BFS is over.
        while (!c_queue.empty()){
            // Get and delete the next Triangle index of the queue. The first one will be the triangle aTdx itself.
            sTdx = c_queue.front();
            c_queue.pop();
            // Get all the neighbours of sTdx.
            NTdx =  &Neighbours[4*sTdx];
            // Run through the list of neighbours. (4 at max)
            for (j=0; j<4; j++){
                // The next valid neighbour is our candidate for the inner Triangle b.
                bTdx = NTdx[j];

                // Check how many neighbours sTdx has. It can be 4 at max. (Itself, and the three others)
                // In order to be able to store the list as contiguous array we fill upp the empty spots with J
                // i.e. the total number of Triangles, which cannot be an index.
                if (bTdx < J){

                    // Prepare Triangle information bTE and bTdet ------------------
                    // Copy coordinates of Triange b to bTE.
                    // again this is done fore convenience only, actually those are unnecessary copies!
                    for (i=0; i<2;i++){
                        bTE[2*0+i] = c_Verts[2*c_Triangles[3*bTdx]   + i];
                        bTE[2*1+i] = c_Verts[2*c_Triangles[3*bTdx+1] + i];
                        bTE[2*2+i] = c_Verts[2*c_Triangles[3*bTdx+2] + i];
                    }
                    bTdet = absDet(&bTE[0]);

                    // Check wheter bTdx is already visited.
                    if (visited[bTdx]==0){

                        // Check whether the verts of aT interact with (bary Center of) bT
                        inNbhd(&aTE[0], &bTE[0], sqdelta, &Mis_interact[0]);
                        // If any of the verts interact we enter the retriangulation and integration
                        if (longVec_any(&Mis_interact[0], 3)){
                            // Retriangulation and integration ------------------------

                            // Get (pointer to) intex of basis function (in Continuous Galerkin)
                            bAdx = &c_Triangles[3*bTdx+0];

                            // Assembly of matrix ---------------------------------------
                            doubleVec_tozero(&termLocal[0], 3 * 3); // Initialize Buffer
                            doubleVec_tozero(&termNonloc[0], 3 * 3); // Initialize Buffer
                            // Compute integrals and write to buffer
                            compute_A(&aTE[0], aTdet, &bTE[0], bTdet,
                                      &P[0], nP, dx, dy, &psi[0], sqdelta,
                                      longVec_all(&Mis_interact[0], 3), termLocal, termNonloc);
                            // If bT interacts it will be a candidate for our BFS, so it is added to the queue
                            if (doubleVec_any(termNonloc, 3 * 3) || doubleVec_any(termLocal, 3 * 3)){
                                c_queue.push(bTdx);
                                // In order to further speed up the integration we only check whether the integral
                                // (termLocal, termNonloc) are 0, in which case we dont add bTdx to the queue.
                                // However, this works only if we can guarantee that interacting triangles do actually
                                // also contribute a non-zero entry, i.e. the Kernel as to be > 0 everywhere for example.
                                // This works for constant kernels, or franctional kernels
                                // The effect of this more precise criterea depends on delta and meshsize.

                                // Copy buffer into matrix. Again solutions which lie on the boundary are ignored
                                //#pragma omp critical
                                //{
                                    for (a=0; a<3; a++){
                                        if (c_Triangles[3*aTdx + a] < L_Omega){
                                            aAdxj = aAdx[a];
                                            for (b=0; b<3; b++){
                                                #pragma omp atomic update
                                                Ad[aAdxj*K + aAdx[b]] += termLocal[3*a+b];
                                                #pragma omp atomic update
                                                Ad[aAdxj*K + bAdx[b]] -= termNonloc[3*a+b];
                                            }
                                        }
                                    }
                                //}
                            }
                        }
                    }
                    // Mark bTdx as visited
                    visited[bTdx] = 1;
                }
            }
        }
        // I dont know wheter this makes sense i parallel case.
        //cout << aTdx << "\r" << flush;
    }
    free(visited);
    free(Mis_interact);
    free(psi);
    }

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
        abort();
    }

    // LU Decomposition
    l = A[2*dx1]/A[2*dx0];
    u = A[2*dx1+1] - l*A[2*dx0+1];

    // Check invertibility
    if (u == 0){
        // raise LinAlgError("in solve2x2. Matrix not invertible.")
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

static void toPhys(double * E, double * p, double * out_x){
    int i=0;
    for (i=0; i<2;i++){
        out_x[i] = (E[2*1+i] - E[2*0+i])*p[0] + (E[2*2+i] - E[2*0+i])*p[1] + E[2*0+i];
    }
}

static void toRef(double * E, double * phys_x, double * ref_p){
    double M[2*2];
    double b[2];
    int i=0;
    for (i=0; i< 2; i++){
        M[2*i] = E[2*1+i] - E[2*0+i];
        M[2*i+1] = E[2*2+i] - E[2*0+i];
        b[i] = phys_x[i] - E[2*0+i];
    }
    solve2x2(&M[0], &b[0], &ref_p[0]);
    return;
}
// Vector operations ---------------------------------------------
// Double

// Check whether any, or all elements of a vector are zero --------------
static int doubleVec_any(double * vec, int len){
    int i=0;
    for (i=0; i< len; i++){
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