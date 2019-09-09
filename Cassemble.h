#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Model ---------------------------------------------------------------------------------------------------------------
static double model_f(double *);
static double model_kernel(double *, double *, double);
static void model_basisFunction(double *, int, double *);
static void inNbhd(double *, double *, double, long *);
static double model_basisFunction(double * , int );

// Integration ---------------------------------------------------------------------------------------------------------
static void innerInt_bary(double *, double *, double *, int, double *, double, double *, double *, double *);
static void innerInt_retriangulate(double *, double *, double *, int, double *, double,  double *, double *);
static void outerInt_full(double *, double, double * , double , double *, int, double *, double *, double *, double, double *, double *);
static void outerInt_retriangulate(double *, double, double * , double , double *, int, double *, double *, double *, double, double *, double *);
static int retriangulate(double *, double *, double, double *);

// Compute A and f -----------------------------------------------------------------------------------------------------
static void compute_f(double *, double, double *, int, double *, double *, double *);
static void compute_A(double *, double, double *, double, double *, int, double *, double *, double *, double, bool, double *, double *);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
static void par_assemble(double *, int, double *, long *, double *,int, int, int, int, int, double *, double *, double *, double, long *);
static void par_evaluateA(double *, double *, int, long *, double *,int, int, int, int, int, double *, double *, double *, double, long *);
static void par_assemblef(double *, long *, double *, int, int, int, double *, double *);
// Math functions ------------------------------------------------------------------------------------------------------
static void solve2x2(double *, double *, double *);        // Solve 2x2 System with LU

// Matrix operations (via * only) --------------------------------
// Double
static double absDet(double *);                            // Compute determinant
static void baryCenter(double *, double *);                // Bary Center
static void toRef(double *, double *, double *);           // Pull point to Reference Element (performs 2x2 Solve)
static void toPhys(double *, double *, double *);          // Push point to Physical Element

// Vector operations ---------------------------------------------
// Double
static double vec_sqL2dist(double *, double *, int);       // L2 Distance
static double vec_dot(double * x, double * y, int len);    // Scalar Product
static int doubleVec_any(double *, int);                   // Any
static void doubleVec_tozero(double *, int);               // Reset to zero

// Long
static int longVec_all(long *, int);                       // All
static int longVec_any(long *, int);                       // Any

// Int
static void intVec_tozero(int *, int);                     // Reset to zero

// Scalar ----------------------------------------------------------
static double absolute(double);                            // Get absolute value
static double scal_sqL2dist(double x, double y);           // L2 Distance


#endif /* Cassemble.h */