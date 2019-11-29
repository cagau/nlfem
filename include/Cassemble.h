#ifndef CASSEMBLE_H
#define CASSEMBLE_H

// Retriangulation Routine ---------------------------------------------------------------------------------------------
int retriangulate(double * , double * , double , double * , int );

// Assembly algorithm with BFS -----------------------------------------------------------------------------------------
void par_assemble(double *, int, int, double *, long *, double *,int, int, int, int, double *, int, double *, double *, int, double*, double, long *, int, int);
void par_assembleMass(double *, long *, double *, int, int, int, double *, double *);
void check_par_assemble(double *, long *, double *, int, int, int, double *, double *, double, long *);
double compute_area(double *, double, long, double *, double, long, double *, int, double *, double);

// Mass matrix evaluation ----------------------------------------------------------------------------------------------
void par_evaluateMass(double *, double *, long *, double *, int, int, int, double *, double *);

//[DEBUG]
//void relativePosition(double *, double *, double *, double *, double *);
//void order(double *, int, double *);

#endif /* Cassemble.h */