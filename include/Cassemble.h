#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Model ---------------------------------------------------------------------------------------------------------------
double model_f(double *);
double model_kernel(double *, long, double *, long, double);
void model_basisFunction(double *, double *);
void model_basisFunction(double * , double * );

// Integration ---------------------------------------------------------------------------------------------------------
void innerInt_retriangulate(double *, long, double *, long, double *, int, double *, double,  double *, double *);
void outerInt_full(double *, double, long, double *, double, long, double *, int, double *, double *, int, double *, double *, double *, double, double *, double *);
//                            (aTE,   aTdet, labela, bTE,     bTdet, labelb, Px,      nPx, dx,        Py,      nPy, dy,       psix,       psiy,   sqdelta, termLocal, termNonloc);
int retriangulate(double * , double * , double , double * , int );

int placePointOnCap(double *, double *, double *, double, double *, double *, double *, double *, double, int, double *);

// Compute A and f -----------------------------------------------------------------------------------------------------
void compute_f(double *, double, double *, int, double *, double *, double *);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(double *, int, int, double *, long *, double *,int, int, int, int, double *, int, double *, double *, int, double*, double, long *, int, int);
void par_assembleMass(double *, long *, double *, int, int, int, double *, double *);
void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *);
double compute_area(double *, double, long, double *, double, long, double *, int, double *, double);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *);

// Math functions ------------------------------------------------------------------------------------------------------
void solve2x2(double *, double *, double *);        // Solve 2x2 System with LU

// Matrix operations (via * only) --------------------------------
// Double
double absDet(double *);                            // Compute determinant
double signDet(double *);
void baryCenter(double *, double *);                // Bary Center
void toRef(double *, double *, double *);           // Pull point to Reference Element (performs 2x2 Solve)
void toPhys(double *, double *, double *);          // Push point to Physical Element

// Vector operations ---------------------------------------------
// Double
double vec_sqL2dist(double *, double *, int);       // L2 Distance
double vec_dot(double * x, double * y, int len);    // Scalar Product
int doubleVec_any(double *, int);                   // Any
void doubleVec_tozero(double *, int);               // Reset to zero
void doubleVec_subtract(double *, double *, double *, int);
void doubleVec_midpoint(double * , double * , double * , int );
void doubleVec_scale(double, double *, double *, int);
void doubleVec_add(double *, double *, double *, int);
void doubleVec_copyTo(double *, double *, int);

// Long
int longVec_all(long *, int);                       // All
int longVec_any(long *, int);                       // Any

// Int
void intVec_tozero(int *, int);                     // Reset to zero

// Scalar ----------------------------------------------------------
double absolute(double);                            // Get absolute value
double scal_sqL2dist(double x, double y);           // L2 Distance

//[DEBUG]
//void relativePosition(double *, double *, double *, double *, double *);
//void order(double *, int, double *);

#endif /* Cassemble.h */