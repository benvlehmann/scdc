Kinematics
==========

We now review the kinematics of phonons and quasiparticles as treated by the code. Note that we work in the "clean" limit, where crystal momentum is conserved.

.. toctree::
   :maxdepth: 2


Units
-----
In the code, we work in "material" units, setting :math:`v_\ell = k_F = \Delta = 1`. This implies that :math:`\hbar = 2\gamma\sqrt z`, where :math:`z\equiv E_F/\Delta`.


Dispersion relations
--------------------
The phonon dispersion relation is :math:`E=\hbar q`. This is invertible: a phonon's energy uniquely determines its momentum. The same is not true for a quasiparticle, which has dispersion relation

.. math::
    E = \left[1 + \left(\frac{\hbar^2 k^2}{2m_e} - z\right)^2\right]^{1/2}.

The inverse is requires a choice of sign:

.. math::
    k = \frac{1}{\hbar}\left[
            2m_e\left(
                z \pm\sqrt{E^2 - 1}
            \right)
        \right]^{1/2}.

This corresponds to the fact that the energy is minimized (:math:`E=\Delta`) at the Fermi surface, :math:`k=k_F`, and increases in either direction. In the code, *if a quasiparticle is instantiated with only energy specified, we choose this sign randomly.*
