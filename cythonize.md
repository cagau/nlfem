Speedup by Cython
=================
Setting 
------
|Name  |Value |
|---|---|
|mesh_name|large|
|delta| .1|
|ansatz | "CG"|
|N basis functs| 16608|
|aTdx|  3391|
|Neigs|  70|


|Status                 | Time needed (Sec) | Speedup (PP)  |
|---                    |---                |----           |
|Pure Python (PP)       | 1.05e+03          |     x1        |
|PP Compiled            | 1.07e+03          |     x1        |
|cy clsInt.retriangulate| 4.69e+02          |     x2.2      |
|cy clsInt.A            | 1.42e+02          |     x7.4      |
