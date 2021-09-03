

.. image:: _static/scdc.png
  :width: 400
  :alt: SCDC stylized as ACDC

Documentation
=============

`scdc` stands for `superconductor down-conversion`. This code simulates the relaxation of quasiparticle excitations produced by an energy deposit in a superconductor, with particular attention to energy deposits from dark matter scattering. The code was developed to study directional correlations between the initial and final states, and is bundled with some simple tools to quantify that relationship.

The basic structure of the calculation is as follows: an energy deposit produces initial quasiparticles and/or phonons. The quasiparticles can relax by emitting phonons, and sufficiently energetic phonons can decay to quasiparticle pairs. This process continues until all quasiparticles are too low-energy to emit a phonon, and all phonons are too low-energy to produce a quasiparticle pair. These excitations constitute the final state, and we say that they are `ballistic`. The full set of excitations connecting initial to final states is generated as a tree of python objects, so every step of the shower can be inspected directly.

The main non-triviality is that the kinematical equations governing phonons and quasiparticles cannot be solved analytically, so the code solves them numerically.

Contents
--------
.. toctree::
   :maxdepth: 2

   kinematics.rst
   code_structure.rst
   analysis.rst
   initial.rst
   using.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
