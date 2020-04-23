Parallel Efficiency of C++ Code
===============================

Huge Mesh 
------
|Name  |Value |
|---|---|
|mesh_name|huge|
|delta| .1|
|ansatz | "CG"|
|N basis functs| 11527|
|aTdx|  13915|
|Neigs|  228|

**Assembly on Kronos**

|Num Kernels                 | Time needed (Sec) | Speedup (PP)  | Time needed (Sec)| Seedup (PP)|
|---                    |---                |----           | --- | --- |
| Compiling with Cython *|  |           | Static or Linked Library| |
|2      |   1.70e+00    |     -          |     7.78e+00     | -  | 
|4      |   9.80e-01    |     x1.73      |     4.17e+00     | x1.87  |
|8      |   5.50e-01    |     x1.78      |     2.26e+00     | x1.85  |
|16     |   2.88e-01    |     x1.9       |     1.24e+00     | x1.82  |
|32     |   2.67e-01    |     x1.08      |     8.85e-01     | x1.40  |
|64     |   1.94e-01    |     x1.38      |     6.04e-01     | x1.46  |

( * )  Compiling the hole code with Cython forces us to declare all functions as **static**, even those cimported
to Cython. Cython then compiles its own C-Code together with the functions in Cassemble.cpp and seems to take tremendous
advantage of those declarations. However, the project and its structure become inflexible and highly dependent on Cython. 
Therefore we choose the second option keeping in mind that there is a way to speed up the code by a factor 4 to 5.
All speed measurements in this file (except the one above) have been performed with the faster/odler version of the code.


Insane Mesh 
------
|Name  |Value |
|---|---|
|mesh_name|insane|
|delta| .1|
|ansatz | "CG"|
|N basis functs| 46228|
|aTdx|  55269|
|Neigs|  682|

**Assembly on Kronos**

|Num Kernels                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|2      |   2.00e+01    |     -      |
|4      |   1.08e+01    |     x1.85     |
|8      |   5.70e+00    |     x1.89    |
|16     |   3.05e+00    |     x1.86      |
|32     |   2.09e+00    |     x1.46     |
|64     |   1.24e+00    |     x1.68      |
