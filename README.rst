Build and Install
=================

In order to use this code you have to meet the following requirements.

-  The **basic requirements** for are the programs
   ``gcc, g++, python3-dev, python3-venv, libgmp-dev, libcgal-dev, metis,`` ``libmetis-dev, libarmadillo-dev``.
   You further need the Python 3 packages ``numpy`` and ``scipy``. If
   you want to change the Cython code, you require the package
   ``Cython``.
-  For **running the examples** you additionally need to install the
   Python packages ``matplotlib, meshzoo``. Make sure to install the
   correct versions by using the ``requirements.txt`` as given below.

-   You can clone the project and build and install the package via

      ``python3 setup.py build --force install``

    The ``--force`` option is required if you change code outside of
    ``nlcfem.pyx``. It might happen that recompilation or translation is
    skipped because Cython assumes that there have been no changes.


Step by Step Installation
------------------------

To prepare the basic requirements on Ubuntu do.

  ``sudo apt-get install git gcc g++ libarmadillo-dev liblapack-dev``

  ``sudo apt-get install python3-venv python3-dev libgmp-dev libcgal-dev``

  ``mkdir nlfemvenv``

  ``python3 -m venv nlfemvenv/``

  ``source nlfemvenv/bin/activate``

  ``(nlfemvenv) python3 -m pip install -r requirements.txt``

To clone the default branch (master) do

  ``git clone https://gitlab.uni-trier.de/klar/nonlocal-assembly.git path/to/nlfem``

To build and install the ``nlfem`` package do.

  ``(nlfemvenv) cd path/to/nlfem``

  ``(nlfemvenv) python3 setup.py build --force install``

Quick Start
===========

To test the rates for the constant kernel run

``(nlfemvenv) cd path/to/nlfem/examples/Test2D``

``(nlfemvenv) python3 computeRates2D.py -f testConfConstant -s 4``

Run a more extensive test via the option ``-f testConfFull -s 4``. Get
help via the option ``-h``.

Licence
=======

nlfem is published under MIT license. Copyright (c) 2021 Manuel Klar, Christian Vollmann

Documentation
=============

Find a documentation of the C++ code in ``docs/html`` and a
documentation of the Python interface in ``docs/build`` or on
http://klar.gitlab-pages.uni-trier.de/nonlocal-assembly/.

