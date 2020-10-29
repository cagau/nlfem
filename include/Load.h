//
// Created by klar on 18.09.20.
//

#ifndef NONLOCAL_ASSEMBLY_MODEL_H
#define NONLOCAL_ASSEMBLY_MODEL_H

#include <cmath>
// Function pointer --------------------------------
extern void (*model_f)(const double * x, double * forcing_out);
/**
 * @brief Constant forcing function f = 1.
 * @param x
 * @param forcing_out
 */

// Implementation of load functions ----------------
void f_constant(const double * x, double * forcing_out);
/**
 * @brief Linear vector valued forcing function \f$f = (0, -2 (x_1 + 1))\f$.
 * @param x
 * @param forcing_out
 */
void fField_linear(const double * x, double * forcing_out);
/**
 * @brief Constant vector valued forcing function.
 * @param x
 * @param forcing_out
 */
void fField_constantRight(const double * x, double * forcing_out);
/**
 * @brief Constant vector valued forcing function.
 * @param x
 * @param forcing_out
 */
void fField_constantDown(const double * x, double * forcing_out);
/**
 * @brief Constant vector valued forcing function. Used for rates checks.
 * @param x
 * @param forcing_out
 */
void fField_constantBoth(const double * x, double * forcing_out);
/**
 * @brief Linear scalar valued forcing function \f$f = -2 (x_1 + 1)\f$. Used for computation of rates.
 * @param x
 * @param forcing_out
 */

void f_linear(const double * x, double * forcing_out);
/**
 * @brief Linear scalar valued forcing function \f$f = -2 (x+ 1)\f$. Used for computation of rates.
 * @param x
 * @param forcing_out
 */
void f_linear3D(const double * x, double * forcing_out);


#endif //NONLOCAL_ASSEMBLY_MODEL_H
