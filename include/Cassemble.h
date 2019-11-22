#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Model ---------------------------------------------------------------------------------------------------------------

static double model_f(double *);
static double model_kernel(double *, long, double *, long, double);
// Used in DEBUG Helper Functions!
void model_basisFunction(double *, double *);

// Integration ---------------------------------------------------------------------------------------------------------
static void innerInt_retriangulate(double *, long, double *, long, double *, int, double *, double,  double *, double *);
static void outerInt_full(double *, double, long, double *, double, long, double *, int, double *, double *, int, double *, double *, double *, double, double *, double *);
//                            (aTE,   aTdet, labela, bTE,     bTdet, labelb, Px,      nPx, dx,        Py,      nPy, dy,       psix,       psiy,   sqdelta, termLocal, termNonloc);
int retriangulate(double * , double * , double , double * , int );

static int placePointOnCap(double *, double *, double *, double, double *, double *, double *, double *, double, int, double *);

// Compute A and f -----------------------------------------------------------------------------------------------------
static void compute_f(double *, double, double *, int, double *, double *, double *);

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(double *, int, int, double *, long *, double *,int, int, int, int, double *, int, double *, double *, int, double*, double, long *, int, int);
void par_assembleMass(double *, long *, double *, int, int, int, double *, double *);
void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *);
double compute_area(double *, double, long, double *, double, long, double *, int, double *, double);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *);

// Math functions ------------------------------------------------------------------------------------------------------
static void solve2x2(double *, double *, double *);        // Solve 2x2 System with LU

// Matrix operations (via * only) --------------------------------
// Double
static double absDet(double *);                            // Compute determinant
static double signDet(double *);
static void baryCenter(double *, double *);                // Bary Center
// Used in DEBUG Helper Functions!
void toRef(double *, double *, double *);           // Pull point to Reference Element (performs 2x2 Solve)
void toPhys(double *, double *, double *);          // Push point to Physical Element

// Vector operations ---------------------------------------------
// Double
static double vec_sqL2dist(double *, double *, int);       // L2 Distance
static double vec_dot(double * x, double * y, int len);    // Scalar Product
static int doubleVec_any(double *, int);                   // Any
static void doubleVec_tozero(double *, int);               // Reset to zero
static void doubleVec_subtract(double *, double *, double *, int);
static void doubleVec_midpoint(double * , double * , double * , int );
static void doubleVec_scale(double, double *, double *, int);
static void doubleVec_add(double *, double *, double *, int);
static void doubleVec_copyTo(double *, double *, int);

// Long
static int longVec_all(long *, int);                       // All
static int longVec_any(long *, int);                       // Any

// Int
static void intVec_tozero(int *, int);                     // Reset to zero

// Scalar ----------------------------------------------------------
static double absolute(double);                            // Get absolute value
static double scal_sqL2dist(double x, double y);           // L2 Distance

//[DEBUG]
//void relativePosition(double *, double *, double *, double *, double *);
//void order(double *, int, double *);

#endif /* Cassemble.h */