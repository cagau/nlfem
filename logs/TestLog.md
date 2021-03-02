
 - A with only termLocal is perfectly symmetric 
    <- norm(A + A.T) = 0 to machine precision
   
 - A with only termLocalPrime is perfectly symmetric
    <- norm(A + A.T) = 0 to machine precision
   
 - We can restore the rates and errors if we use 7 Quad Points in the inner integral
   and only use *termLocalPrime*

| h| dof| L2 Error| Rates| 
|---|---|---|---|
| 1.00e-01 | 8.10e+01 | 2.72e-03 | 0.00e+00 | 
| 5.00e-02 | 3.61e+02 | 5.36e-04 | 2.34e+00 | 
| 2.50e-02 | 1.52e+03 | 1.39e-04 | 1.95e+00 | 
| 1.25e-02 | 6.24e+03 | 3.34e-05 | 2.05e+00 | 
| 6.25e-03 | 2.53e+04 | 8.95e-06 | 1.90e+00 |

- We can restore the rates and errors if we use 7 Quad Points in the inner integral
  and only use termLocalPrime

| h| dof| L2 Error| Rates| 
|---|---|---|---|
| 1.00e-01 | 8.10e+01 | 2.51e-03 | 0.00e+00 | 
| 5.00e-02 | 3.61e+02 | 5.28e-04 | 2.25e+00 | 
| 2.50e-02 | 1.52e+03 | 1.38e-04 | 1.94e+00 | 
| 1.25e-02 | 6.24e+03 | 3.31e-05 | 2.06e+00 | 
| 6.25e-03 | 2.53e+04 | 8.94e-06 | 1.89e+00 | 

- The kernel of the symmetrified matrix 0.5(A + A^T) with 2*termLocal
  is also off, but the smallest Eigenvalue is -4.177e-06.
  
- The kernel of the symmetrified matrix 0.5(A+A^T) with different 
  termLocal + termLocalPrime is completely off. The smallest
 Eigenvalue is -2.204e-2
  
  
