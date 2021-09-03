Initial distribution
====================

When studying the directionality of excitations produced in dark matter scattering, down-conversion is only the second part of the calculation. The first, of course, is to sample the initial excitations in the first place!

The ``initial`` module provides tools to do exactly that, all the way from a halo velocity distribution to a material response function to an ensemble of initial excitations.


Dark matter distribution
------------------------
Before we can sample the initial excitations, we need to define the dark matter velocity distribution (speed and direction). A parent class for such distributions is provided in the ``halo`` module, and several important cases are implemented as subclasses.

+++++++++++++++
``halo`` module
+++++++++++++++

.. automodule:: scdc.initial.halo
    :members:


Dark matter interaction
-----------------------
Given the dark matter distribution, one still needs to supply the squared matrix element for the interaction with the electrons in the superconductor, and the material response function (i.e. the dielectric function and coherence factor). These are defined in the ``matrix_element`` and ``response`` modules, respectively.

+++++++++++++++++++++++++
``matrix_element`` module
+++++++++++++++++++++++++

.. automodule:: scdc.initial.matrix_element
    :members:

+++++++++++++++++++
``response`` module
+++++++++++++++++++

.. automodule:: scdc.initial.response
    :members:


Producing initial samples
-------------------------
The final step of the process is to generate actual samples given all of the above ingredients. This is performed by the ``distribution`` module. Previous versions of the code used different methods for this sampling problem, and the surviving method is implemented in the ``integral`` submodule.

+++++++++++++++++++
``integral`` module
+++++++++++++++++++

.. automodule:: scdc.initial.distribution.integral
    :members:
