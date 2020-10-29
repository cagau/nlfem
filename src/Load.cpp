/**
    Contains kernel and forcing functions which are used in the assembly. Different additional kernels can be added
    here.

    @file model.cpp
    @author Manuel Klar
    @version 0.1 25/08/20
**/
#include "model.h"

using namespace std;

// ### RIGHT HAND SIDE #################################################################################################
void (*model_f)(const double * x, double * forcing_out);
// Implementations -----------------------------------------------------------------------------------------------------
void f_constant(const double * x, double * forcing_out){
    forcing_out[0] = 1.0;
}
void fField_linear(const double * x, double * forcing_out){
    forcing_out[0] = 0.0;
    forcing_out[1] = -2. * (x[1] + 1.);
}
void fField_constantRight(const double * x, double * forcing_out){
    forcing_out[0] = 1.0e-1;
    forcing_out[1] = 0.0;
}
void fField_constantDown(const double * x, double * forcing_out){
    const double f1 = 0; //1e-3;
    const double f2 = -1.*1e-3;
    forcing_out[0] = f1;
    forcing_out[1] = f2;
}
void fField_constantBoth(const double * x, double * forcing_out){
    const double f1 = .5*1e-3;
    const double f2 = 3./2.*1e-3;
    forcing_out[0] = f1;
    forcing_out[1] = f2;
}
void f_linear(const double * x, double * forcing_out){
    *forcing_out = -2. * (x[1] + 1.);
}
void f_linear1D(const double * x, double * forcing_out){
    *forcing_out = -2. * (x[0] + 1.);
}
void f_linear3D(const double * x, double * forcing_out){
    *forcing_out = -2. * (x[1] + 2.);
}
