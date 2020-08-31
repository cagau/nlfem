/**
    Contains kernel and forcing functions which are used in the assembly. Different additional kernels can be added
    here.
    @file model.cpp
    @author Manuel Klar
    @version 0.1 25/08/20
*/
#ifndef NONLOCAL_ASSEMBLY_MODEL_CPP
#define NONLOCAL_ASSEMBLY_MODEL_CPP

#include "mathhelpers.cpp"

// ### KERNEL ##########################################################################################################
// Pointer -------------------------------------------------------------------------------------------------------------
void (*model_kernel)(const double * x, long labelx, const double * y, long labely, double sqdelta, double * kernel_val);

// Implementations -----------------------------------------------------------------------------------------------------
void kernel_constant(const double * x, const long labelx, const double * y, const long labely, const double sqdelta,
                       double * kernel_val){
    *kernel_val = 4 / (M_PI * pow(sqdelta, 2));
}

void kernel_constant3D(const double * x, const long labelx, const double * y, const long labely, const double sqdelta,
                         double * kernel_val){
    *kernel_val = 15 / (M_PI * 4 * pow(sqrt(sqdelta), 5));
}

void kernel_labeled(const double * x, const long labelx, const double * y, const long labely, const double sqdelta,
                      double * kernel_val){
    double dist;
    long label;

    label = 10*labelx + labely;
    dist = vec_sqL2dist(x, y, 2);
    if (dist >= sqdelta) {
        cout << "Error in model_kernel. Distance smaller delta not expected." << endl;
        cout << dist << endl;
        abort();
    }
    if (label <= 12) {
        *kernel_val = 0.01 * 3. / (4*pow(sqdelta, 2));
    } else if (label>=21){
        *kernel_val = (100 *  3. / (4*pow(sqdelta, 2))) * (1 - (dist/sqdelta) );
    } else if (label == 13){
        *kernel_val = 0.0;
    } else {
        cout << "No such case " << endl;
        abort();
    }
}

void kernelField_linearPrototypeMicroelastic(const double * x, const long labelx, const double * y, const long labely,
                                        const double sqdelta, double * kernel_val) {
}
void kernelField_constant(const double * x, const long labelx, const double * y, const long labely,
                                             const double sqdelta, double * kernel_val) {
    //double z[2];
    //z[0] = x[0] - y[0];
    //z[1] = x[1] - y[1];
    //*kernel_val = 1./sqrt(vec_dot(z,z,2));
    // KERNEL ORDER [ker (0,0), ker (0,1), ker (1,0), ker (1,1)]
    kernel_val[0] = 0.0;
    kernel_val[1] = 0.0;
    kernel_val[2] = 0.0;
    kernel_val[3] = 4 / (M_PI * pow(sqdelta, 2));
}

// ### RIGHT HAND SIDE #################################################################################################

// Pointer -------------------------------------------------------------------------------------------------------------
void (*model_f)(const double * x, double * forcing_out);

// Implementations -----------------------------------------------------------------------------------------------------
void f_constant(const double * x, double * forcing_out){
    forcing_out[0] = 1.0;
}
void fField_linear(const double * x, double * forcing_out){
    forcing_out[0] = -2. * (x[1] + 1.);
    forcing_out[1] = -2. * (x[1] + 1.);
}
void f_linear(const double * x, double * forcing_out){
    *forcing_out = -2. * (x[1] + 1.);
}
void f_linear3D(const double * x, double * forcing_out){
    *forcing_out = -2. * (x[1] + 2.);
}
// ### BASIS FUNCTION ##################################################################################################

void model_basisFunction(const double * p, double *psi_vals){
    psi_vals[0] = 1 - p[0] - p[1];
    psi_vals[1] = p[0];
    psi_vals[2] = p[1];
}

void model_basisFunction(const double * p, const int dim, double *psi_vals){
    int i=0;

    psi_vals[0] = 1;
    for (i=0; i<dim; i++){
        psi_vals[0] -= p[i];
        psi_vals[i+1] = p[i];
    }
}

#endif //NONLOCAL_ASSEMBLY_MODEL_CPP