//
// Created by klar on 18.09.20.
//

#ifndef NONLOCAL_ASSEMBLY_MODEL_H
#define NONLOCAL_ASSEMBLY_MODEL_H
// Pointer -------------------------------------------------------------------------------------------------------------
//// Kernel pointer and implementations

/**
 * Pointer to kernel functions. This function describes the behaviour if the kernel function on its
 * support only. Hence. it is not supposed to check whether to points interact.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void (*model_kernel)(const double * x, long labelx, const double * y, long labely, double sqdelta, double * kernel_val);
/**
 * Constant kernel in 2D case. The constant is chosen such that the operator is equivalent to the laplacian for
 * polynomials of degree less or equal to 2.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void kernel_constant(const double * x, long labelx, const double * y, long labely, double sqdelta,
                     double * kernel_val);

/**
 * Constant kernel in 3D case. The constant is chosen such that the operator is equivalent to the laplacian for
 * polynomials of degree less or equal to 2.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void kernel_constant3D(const double * x, long labelx, const double * y, long labely, double sqdelta,
                       double * kernel_val);
/**
 * Kernel depending on the triangle labels. Can be used to model nonlocal to nonlocal coupling.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void kernel_labeled(const double * x, long labelx, const double * y, long labely, double sqdelta,
                    double * kernel_val);

/**
 * Kernel for peridynamics diffusion model. The scalar valued weakly singular kernel reads as
 *
 *  * \f[
 * \gamma(x,y) = \frac{1}{\| x - y \|}  \frac{3}{\pi \delta ^3}.
 *  \f]
 * The constant is chosen such that the operator is equivalent to the laplacian for
 * polynomials of degree less or equal to 2.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void kernel_linearPrototypeMicroelastic(const double * x, long labelx, const double * y, long labely,
                                        double sqdelta, double * kernel_val);
/**
 * Kernel for the peridynamics model. The matrix valued weakly singular kernel reads as
 *
 *  \f[
 * \gamma(x,y) = (x-y) \otimes (x-y) \frac{1}{\| x - y \|^3}  \frac{12}{\pi \delta ^3}.
 *  \f]
 *
 * The constant is chosen such that the operator is equivalent to linear elasticity for
 * polynomials of degree less or equal to 2 and lame paramter
 * \f$\mu = \lambda = \frac{\pi}{4}\f$.
 *
 * @param x Physical point of the outer integration region.
 * @param labelx Label of the outer triangle.
 * @param y Physical point of the inner integration region.
 * @param labely Label of inner triangle.
 * @param sqdelta Squared delta.
 * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
 * of shape d x d in case of peridynamics.
 */
void kernelField_linearPrototypeMicroelastic(const double * x, long labelx, const double * y, long labely,
                                             double sqdelta, double * kernel_val);

void kernelField_constant(const double * x, long labelx, const double * y, long labely,
                          double sqdelta, double * kernel_val);

void (*model_f)(const double * x, double * forcing_out);
void f_constant(const double * x, double * forcing_out);
void fField_linear(const double * x, double * forcing_out);
void fField_constantDown(const double * x, double * forcing_out);
void f_linear(const double * x, double * forcing_out);
void f_linear3D(const double * x, double * forcing_out);


void model_basisFunction(const double * p, double *psi_vals);
void model_basisFunction(const double * p, int dim, double *psi_vals);
#endif //NONLOCAL_ASSEMBLY_MODEL_H
