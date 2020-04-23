//
// Created by klar on 16.03.20.
//
#ifndef NONLOCAL_ASSEMBLY_MATHHELPERS_CPP
#define NONLOCAL_ASSEMBLY_MATHHELPERS_CPP
using namespace std;

// ___ MATH HELERPS DECLARATION ________________________________________________________________________________________

// Miscellaneous helpers ###############################################################################################
void solve2x2(const double *, const double *, double *);                 // Solve 2x2 System with LU
void rightNormal(const double * y0, const double * y1, double orientation, double * normal);

// Matrix operations ###################################################################################################

// Double
double absDet(const double * E);                                         // Compute determinant
double absDet(const double * E, int dim);
double signDet(const double * E);
double signDet(const double * E, const MeshType & mesh);
void baryCenter(const double * E, double * bary);                        // Bary Center
void baryCenter(int dim, const double * E, double * bary);
void toRef(const double * E, const double * phys_x, double * ref_p);     // Pull point to Reference Element (performs 2x2 Solve)
void toPhys(const double * E, const double * p, double * out_x);         // Push point to Physical Element
void toPhys(const double * E, const double * p, const MeshType & mesh, double * out_x);

// Vector operations ###################################################################################################

// Double
double vec_sqL2dist(const double * x, const double * y, int len);      // L2 Distance
double vec_dot(const double * x, const double * y, int len);           // Scalar Product
int doubleVec_any(const double * vec, int len);                        // Any
void doubleVec_tozero(double *, int);               // Reset to zero
void doubleVec_subtract(const double * vec1, const double * vec2, double * out, int len);
void doubleVec_midpoint(const double * vec1, const double * vec2, double * midpoint, int len);
void doubleVec_scale(double lambda, const double * vec, double * out, int len);
void doubleVec_add(const double * vec1, const double * vec2, double * out, int len);
void doubleVec_copyTo(const double * input, double * output, int len);

// Long
int longVec_all(const long *, int);                // All
int longVec_any(const long *, int);                // Any

// Int
void intVec_tozero(int *, int);                    // Reset to zero

// Scalar operations ###################################################################################################
double absolute(double);                                  // Get absolute value
double scal_sqL2dist(double x, double y);           // L2 Distance


// ___ MATH HELERPS IMPLEMENTATION _____________________________________________________________________________________

// Miscellaneous helpers ###############################################################################################

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

// Normal which looks to the right w.r.t the vector from y0 to y1.
void rightNormal(const double * y0, const double * y1, const double orientation, double * normal){
    normal[0] = y1[1] - y0[1];
    normal[1] = y0[0] - y1[0];
    doubleVec_scale(orientation, normal, normal, 2);
}

// ### MATRIX OPERATIONS ###############################################################################################

double absDet(const double * E){
    double M[2][2];
    int i=0;
    for (i=0; i< 2; i++){
        M[i][0] = E[2*1+i] - E[2*0+i];
        M[i][1] = E[2*2+i] - E[2*0+i];
    }
    return absolute(M[0][0]*M[1][1] - M[0][1]*M[1][0]);
}

double absDet(const double * E, const int dim){
    // Let a,b,c,d be the Verticies of a Tetrahedon (3D)
    // Then M will be the 3x3 matrix containg [b-a,c-a,d-a]
    int dVertex = dim+1;

    arma::mat M(dim, dim, arma::fill::zeros);

    // Copy Values
    int i=0, j=0;
    for (i = 1; i < dVertex; i++) {
        for (j = 0; j < dim; j++) {
            M(j,i-1) += (E[j + dim*i] - E[j + 0]);
            //printf("%3.2f ", M(j,i-1));
        }
        //printf("\n");
    }
    return absolute(arma::det(M));
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

double signDet(const double * E, const MeshType & mesh){
    // Let a,b,c,d be the Verticies of a Tetrahedon (3D)
    // Then M will be the 3x3 matrix containg [b-a,c-a,d-a]
    arma::mat M(mesh.dim, mesh.dim, arma::fill::zeros);
    double det;
    // Copy Values
    int i=0, j=0;
    for (i = 1; i < mesh.dVertex; i++) {
        for (j = 0; j < mesh.dim; j++) {
            M(j,i-1) += (E[j + mesh.dim*i] - E[j + 0]);
        }
    }
    det = arma::det(M);
    if (det > 0){
        return 1.;
    } else if ( det < 0){
        return -1.;
    } else {
        //cout << "Warning in signDet() 3D: Determinant is 0" << endl;
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

void baryCenter(const int dim, const double * E, double * bary){
    int i=0, j=0;
    int dVert = dim+1;
    doubleVec_tozero(bary, dim);

    for (j=0; j<dim; j++){
        for  (i=0; i<dVert; i++) {
            bary[j] += E[dim * i + j];
            //bary[1] += E[2*i+1];
        }
    bary[j] = bary[j]/ static_cast<double>(dVert);
    }
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

void toPhys(const double * E, const double * p, const MeshType & mesh, double * out_x) {
    int i = 0, j = 0;
    doubleVec_tozero(out_x, mesh.dim);
    for (i=0; i<mesh.dim;i++){
        for(j=0; j<mesh.dim;j++){
            out_x[i] += p[j]*(E[mesh.dim*(j+1)+i] - E[i]);
        }
        out_x[i] += E[i];
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

// ### VECTOR OPERATIONS ###############################################################################################

// Check whether any, or all elements of a vector are zero -----------
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

#endif //NONLOCAL_ASSEMBLY_MATHHELPERS_CPP