//
// Created by klar on 29.10.20.
//

#ifndef NONLOCAL_ASSEMBLY_KERNEL_H
#define NONLOCAL_ASSEMBLY_KERNEL_H
#include "MeshTypes.h"

class Kernel {
public:
    const double delta;
    const double sqdelta;
    const double trdelta;
    const double qddelta;
    const double qtdelta;

    const long outdim;
    const bool isWeaklySingular;
    Kernel(long outdim_, double delta_, bool isSingular_):
    delta(delta_),
    sqdelta(pow(delta_, 2)),
    trdelta(pow(delta_, 3)),
    qddelta(pow(delta_, 4)),
    qtdelta(pow(delta_, 5)),
    outdim(outdim_),
    isWeaklySingular(isSingular_){};

    virtual double horizon(double * x, double * y, long labelx, long labely) { return 0.0; };
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) {};
};

class constant2D: public Kernel {
public:
    /**
     @brief Constant kernel in 2D case. The constant is chosen such that the operator is equivalent to the laplacian for
     * polynomials of degree less or equal to 2.
     *
     * @param interactionHorizon
     */
    const double sqsqdelta;
    explicit constant2D(double interactionHorizon): Kernel(1, interactionHorizon, false), sqsqdelta(pow(delta, 4)){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
     * @brief Evaluates kernel.
     *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion.
     * @return void
     */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};

class parabola2D: public Kernel {
public:
    /**
    * @brief This kernel is defined by \f$ \delta^2 - \|z\|^2 \f$. It is radial but not
    * singular or constant. The second moment is given by \f$ \pi \delta ^6 / 12 \f$.
     *
     * @param interactionHorizon
     *
    */
    explicit parabola2D(double interactionHorizon): Kernel(1, interactionHorizon, false){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
    * @param x Physical point of the outer integration region.
    * @param labelx Label of the outer triangle.
    * @param y Physical point of the inner integration region.
    * @param labely Label of inner triangle.
    * @param sqdelta Squared delta.
    * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion.
    * @return void
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};


class constant1D: public Kernel {
public:
    /**
     * @brief Constant kernel in 1D case. The constant is chosen such that the operator is equivalent to the laplacian for
     * polynomials of degree less or equal to 2.
     *
     * @param interactionHorizon
     *
    */
    explicit constant1D(double interactionHorizon): Kernel(1, interactionHorizon, false){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};

class constant3D: public Kernel {
public:
    /**
     * @brief Constant kernel in 3D case. The constant is chosen such that the operator is equivalent to the laplacian for
     * polynomials of degree less or equal to 2.
     *
     * @param interactionHorizon
     *
    */
    explicit constant3D(double interactionHorizon): Kernel(1, interactionHorizon, false){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};

class labeled2D: public Kernel {
public:
    /**
     * @brief Kernel depending on the triangle labels. Can be used to model nonlocal to nonlocal coupling.
     *
     * @param interactionHorizon
     *
    */
    explicit labeled2D(double interactionHorizon): Kernel(1, interactionHorizon, false){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};


class linearPrototypeMicroelastic2D: public Kernel {
public:
    /**
     * @brief Kernel for peridynamics diffusion model. The scalar valued weakly singular kernel reads as
     *
     *  \f[
     * \gamma(x,y) = \frac{1}{\| x - y \|}  \frac{3}{\pi \delta ^3}.
     *  \f]
     * The constant is chosen such that the operator is equivalent to the laplacian for
     * polynomials of degree less or equal to 2.
     * @param interactionHorizon
     *
    */
    explicit linearPrototypeMicroelastic2D(double interactionHorizon): Kernel(1, interactionHorizon, true) {};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};

class linearPrototypeMicroelastic2DField: public Kernel {
public:
    /**
     * @brief Kernel for the peridynamics model. The matrix valued weakly singular kernel reads as
     *
     *  \f[
     * \gamma(x,y) = (x-y) \otimes (x-y) \frac{1}{\| x - y \|^3}  \frac{12}{\pi \delta ^3}.
     *  \f]
     *
     * The constant is chosen such that the operator is equivalent to linear elasticity for
     * polynomials of degree less or equal to 2 and lame paramter
     * \f$\mu = \lambda = \frac{\pi}{4}\f$.
     *
    */
    explicit linearPrototypeMicroelastic2DField(double interactionHorizon): Kernel(2, interactionHorizon, true) {};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};

class constant2DField: public Kernel {
public:
    /**
     * @brief Kernel for the peridynamics model. The matrix valued weakly singular kernel reads as
     *
     *  \f[
     * \gamma(x,y) = (x-y) \otimes (x-y) \frac{1}{\| x - y \|^3}  \frac{12}{\pi \delta ^3}.
     *  \f]
     *
     * The constant is chosen such that the operator is equivalent to linear elasticity for
     * polynomials of degree less or equal to 2 and lame paramter
     * \f$\mu = \lambda = \frac{\pi}{4}\f$.
     *
    */
    explicit constant2DField(double interactionHorizon): Kernel(2, interactionHorizon, false){};
    double horizon(double * x, double * y, long labelx, long labely) override { return Kernel::delta; };
    /**
    * @brief Evaluates kernel.
    *
     * @param x Physical point of the outer integration region.
     * @param labelx Label of the outer triangle.
     * @param y Physical point of the inner integration region.
     * @param labely Label of inner triangle.
     * @param sqdelta Squared delta.
     * @param kernel_val Value of the the kernel. Pointer to double in case of diffusion. Pointer to a array
     * of shape d x d in case of peridynamics.
    */
    virtual void operator() (double * x, double * y, long labelx, long labely, double * kernel_val) override;
};


#endif //NONLOCAL_ASSEMBLY_KERNEL_H
