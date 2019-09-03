Speedup by Cython
=================
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
|parallel C | 9.77e-02 | x10747|
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
|parallel C| 7.15e-01 | x11944|

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
|parallel Cython| 8.13e+00 | x4|