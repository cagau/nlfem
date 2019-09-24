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

|Num Kernels                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|2      |   1.70e+00    |     -      |
|4      |   9.80e-01    |     x1.73      |
|8      |   5.50e-01    |     x1.78      |
|16     |   2.88e-01    |     x1.9       |
|32     |   2.67e-01    |     x1.08      |
|64     |   1.94e-01    |     x1.38      |

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
