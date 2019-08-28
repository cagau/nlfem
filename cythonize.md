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
|+ cy clsInt.A (i. e. inner and outer Int) | 2.17e+01   |     x48.4   |
|+ cy inNbhd | 1.84e+01          |     x57.1    |
|+ cy assemble | 8.7e+00 | x120,69|
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
|Cython Code| 1.32e+02 | x64,7|
