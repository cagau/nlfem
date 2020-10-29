//
// Created by klar on 29.10.20.
//

#include "Kernel.h"
#include "mathhelpers.h"

void constant2D::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
        *kernel_val = 4 / (M_PI * sqsqdelta);
};

void constant1D::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
    *kernel_val = 3./(2. * trdelta);
};

void parabola2D::operator() (double * x, double * y, long labelx, long labely, double * kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    const double value = sqdelta - vec_dot(z,z,2);
    const double c =  12./(M_PI * qtdelta);
    *kernel_val = c*value;
};

void constant3D::operator()  (double * x, double * y, long labelx, long labely, double * kernel_val) {
    *kernel_val = 15 / (M_PI * 4 * qtdelta);
};

void labeled2D::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
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
        *kernel_val = 0.01 * 3. / (4*qddelta);
    } else if (label>=21){
        *kernel_val = (100 *  3. / (4*qddelta)) * (1 - (dist/sqdelta) );
    } else if (label == 13){
        *kernel_val = 0.0;
    } else {
        cout << "No such case " << endl;
        abort();
    }
};

void linearPrototypeMicroelastic2D::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    const double denominator = 1.0/sqrt(vec_dot(z,z,2));
    const double c =  3.0/(M_PI * trdelta);
    *kernel_val = c*denominator;
};

void linearPrototypeMicroelastic2DField::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
    double z[2];
    z[0] = x[0] - y[0];
    z[1] = x[1] - y[1];
    double denominator = 1.0/pow(sqrt(vec_dot(z,z,2)),3);
    double c =  12.0/(M_PI * trdelta);

    kernel_val[0] = c*denominator*z[0]*z[0];
    kernel_val[1] = c*denominator*z[0]*z[1];
    kernel_val[2] = c*denominator*z[1]*z[0];
    kernel_val[3] = c*denominator*z[1]*z[1];
};

void constant2DField::operator()(double *x, double *y, long labelx, long labely, double *kernel_val) {
    //double z[2];
    //z[0] = x[0] - y[0];
    //z[1] = x[1] - y[1];
    //*kernel_val = 1./sqrt(vec_dot(z,z,2));
    // KERNEL ORDER [ker (0,0), ker (0,1), ker (1,0), ker (1,1)]
    kernel_val[0] = 4. / (M_PI * qddelta);
    kernel_val[1] = 0.0;
    kernel_val[2] = 4. / (M_PI * qddelta);
    kernel_val[3] = 0.0;
};
