/**
    Contains kernel and forcing functions which are used in the assembly. Different additional kernels can be added
    here.

    @file model.cpp
    @author Manuel Klar
    @version 0.1 25/08/20
**/
#include <iostream>

#include "model.h"
#include "mathhelpers.h"

using namespace std;

// ### KERNEL ##########################################################################################################
void (*model_kernel)(const double * x, long labelx, const double * y, long labely, const MeshType &mesh, double * kernel_val);

// Implementations -----------------------------------------------------------------------------------------------------
void kernel_constant(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                     double *kernel_val) {
    *kernel_val = 4 / (M_PI * pow(mesh.sqdelta, 2));
}

void kernel_constantLinf2D(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                     double *kernel_val) {
    *kernel_val = 3 / (4 * pow(mesh.sqdelta, 2));
}

void kernel_constantTruncated(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                     double *kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    bool doesInteract = (mesh.sqdelta > vec_dot(z,z,2) );
    *kernel_val = doesInteract * 4 / (M_PI * pow(mesh.sqdelta, 2));
}

void kernel_constant1D(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                     double *kernel_val) {
    *kernel_val = 3./(2. * pow(mesh.delta, 3));
}
void kernel_parabola(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                     double *kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    const double value = mesh.sqdelta - vec_dot(z,z,2);
    const double c =  12./(M_PI * pow(mesh.sqdelta, 3));
    *kernel_val = c*value;
}

void kernel_constant3D(const double *x, const long labelx, const double *y, const long labely, const MeshType &mesh,
                       double *kernel_val) {
    *kernel_val = 15 / (M_PI * 4 * pow(mesh.delta, 5));
}

void kernel_labeled(const double * x, const long labelx, const double * y, const long labely, const MeshType &mesh,
                      double * kernel_val){
    double dist;
    long label;

    label = 10*labelx + labely;
    dist = vec_sqL2dist(x, y, 2);
    if (dist >= mesh.sqdelta) {
        cout << "Error in model_kernel. Distance smaller delta not expected." << endl;
        cout << dist << endl;
        abort();
    }
    if (label <= 12) {
        *kernel_val = 0.01 * 3. / (4*pow(mesh.sqdelta, 2));
    } else if (label>=21){
        *kernel_val = (100 *  3. / (4*pow(mesh.sqdelta, 2))) * (1 - (dist/mesh.sqdelta) );
    } else if (label == 13){
        *kernel_val = 0.0;
    } else {
        cout << "No such case " << endl;
        abort();
    }
}
void kernel_linearPrototypeMicroelastic(const double * x, const long labelx, const double * y, const long labely,
                                             const MeshType &mesh, double * kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    const double denominator = 1.0/sqrt(vec_dot(z,z,2));
    const double c =  3.0/(M_PI * pow(mesh.delta,3));
    *kernel_val = c*denominator;
}

void kernel_fractional(const double * x, const long labelx, const double * y, const long labely,
                                        const MeshType &mesh, double * kernel_val) {
    const double exponent = mesh.dim+2.*mesh.fractional_s;
    const double constant_exponent = 2. - 2.*mesh.fractional_s;
    double z[mesh.dim];
    for (int i = 0; i < mesh.dim; i++){
        z[i] = x[i] - y[i];
    }
    const double denominator = 1.0/pow(sqrt(vec_dot(z,z,mesh.dim)), exponent);
    // Constant is wrong I guess..
    const double c =  (constant_exponent)/(M_PI*pow(mesh.delta, constant_exponent));
    *kernel_val = c*denominator;
}

void kernelField_linearPrototypeMicroelastic(const double * x, const long labelx, const double * y, const long labely,
                                        const MeshType &mesh, double * kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    double denominator = 1.0/pow(sqrt(vec_dot(z,z,2)),3);
    //double f0 = 2*mesh.sqdelta - vec_dot(z,z,2); // Sign changing.
    //double c =  12.0/(M_PI * pow(mesh.delta,3));
    double c =  3.0/pow(mesh.delta,3);
    kernel_val[0] = c*denominator*z[0]*z[0] ;//+ f0;
    kernel_val[1] = c*denominator*z[0]*z[1];
    kernel_val[2] = c*denominator*z[1]*z[0];
    kernel_val[3] = c*denominator*z[1]*z[1] ;//+ f0;
}
void kernelField_constant(const double * x, const long labelx, const double * y, const long labely,
                                             const MeshType &mesh, double * kernel_val) {
    //double z[2];
    //z[0] = x[0] - y[0];
    //z[1] = x[1] - y[1];
    //*kernel_val = 1./sqrt(vec_dot(z,z,2));
    // KERNEL ORDER [ker (0,0), ker (0,1), ker (1,0), ker (1,1)]
    kernel_val[0] = 4. / (M_PI * pow(mesh.sqdelta, 2));
    kernel_val[1] = 0.0;
    kernel_val[2] = 4. / (M_PI * pow(mesh.sqdelta, 2));
    kernel_val[3] = 0.0;
}

// ### RIGHT HAND SIDE #################################################################################################
void (*model_f)(const double * x, double * forcing_out);
// Implementations -----------------------------------------------------------------------------------------------------
void f_constant(const double * x, double * forcing_out){
    forcing_out[0] = 1.0;
}
void fField_linear(const double * x, double * forcing_out){
    const double c = M_PI / 5.0;
    forcing_out[0] = -c*(1.0 + 2*x[0]);
    forcing_out[1] = -c*x[1];

}
void fField_constantRight(const double * x, double * forcing_out){
    forcing_out[0] = 1.0e-1;
    forcing_out[1] = 0.0;
}
void fField_constantDown(const double * x, double * forcing_out){
    //const double f1 = 0; //1e-3;
    //const double f2 = -1.*1e-3;
    const double c = - M_PI;
    forcing_out[0] = 0;
    forcing_out[1] = c*2;
}
void fField_constantBoth(const double * x, double * forcing_out){
    //const double f1 = .5*1e-3;
    //const double f2 = 3./2.*1e-3;
    //forcing_out[0] = f1;
    //forcing_out[1] = f2;

    const double c = - M_PI;
    forcing_out[0] = c;
    forcing_out[1] = c*2;
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
