Speedup by Cython and parallel C++
=================

The assembly is done with **retriangulation of the inner triangle**, hence we can expect a slight speedup
with a better numerical algorithm. The initial Python code was very inefficient and
performed 8 unnecessary retriangulations in each step. Hence, the **Speedup factors
should be divided by 10 to get a realistic assessment** of the speedup by Cython or C++.

Large Mesh 
------
|Name  |Value |
|---|---|
|mesh_name|large|
|delta| .1|
|ansatz | "CG"|
|N basis functs| 2849|
|aTdx|  3391|
|Neigs|  70|


|Status                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|Pure Python (PP)       | 1.05e+03          |     x1        |
|PP Compiled            | 1.07e+03          |     x1        |
|+ cy clsInt.retriangulate| 4.69e+02          |     x2.2      |
|+ cy clsInt.compute_A (i. e. inner and outer Int) | 2.17e+01   |     x48.4   |
|+ cy inNbhd | 1.84e+01          |     x57.1    |
|full Cython | 2.90e-01 | x3620|
|parallel C++ | 4.80e-02 | x21875|
|kronos|6.88e-02||

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

|Status                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|Pure Python (PP)       | 8.54e+03       |     x1        |
|full Cython| 4.09e+00 | x2088|
|parallel C++| 5.47e-01 | x14680|
|kronos| 5.26e-01||

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

|Status                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|Pure Python (PP)       | -      |     -        |
|full Cython| 3.32e+01 | x1|
|parallel C++| 6.64e+00 | x5|
|kronos| 3.67e+00||