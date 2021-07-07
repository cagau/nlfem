.. nlfem documentation master file, created by
   sphinx-quickstart on Thu May  6 17:59:33 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to nlfem's documentation!
=================================
.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. include:: ../../README.rst

Python interface
----------------

.. autofunction:: nlfem.stiffnessMatrix_fromArray
.. autofunction:: nlfem.stiffnessMatrix
.. autofunction:: nlfem.loadVector

Adding kernels or forcing functions
--------------------------------------------

If you want to change the model you have to alter the file
``src/model.cpp``. You find a several options for kernel and forcing
functions there.

-  The Kernel function is called model\_kernel() .
-  The right side is called model\_f() .

Step 1 Altering the model
~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to change the kernel implement your version
``kernel_myKernel()`` while exactly meeting the interface of the
function pointer ``mode_kernel``. Do the same thing for the right hand
side.

Step 2 Adding the option
~~~~~~~~~~~~~~~~~~~~~~~~

Open the file ``src/Cassemble.cpp`` and add an entry in the map inside of
``lookup_configuration()`` in order to make your kernel
available from the outside. The function where the assembly happens is
called ``par_system()`` and it is also to be found in this file.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
