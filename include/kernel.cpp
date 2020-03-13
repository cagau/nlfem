//
// Created by klar on 13.03.20.
//
// Model ---------------------------------------------------
double model_kernel(const double * x, long labelx, const double * y, long labely, double sqdelta);


double model_kernel(const double * x, const long labelx, const double * y, const long labely, const double sqdelta){
    return 4 / (M_PI * pow(sqdelta, 2));
}