"""This module defines a `Material` class.

A note on units: at various places in the code, it is useful to have work with
non-dimensionalized quantities. However, picking these quantities requires
some care, because there are two sets of scales: those associated with
behavior near and far from the gap. We are mainly interested in near-gap
behavior, so we use the half-gap Delta as the energy scale. Now, excitations
with energy Delta have momentum k_F, so it makes sense to take k_F as our
preferred momentum scale. So far we have

    Delta = k_F = 1.

Having made this choice, we automatically obtain a velocity v_m and a mass m_m
as follows:

    v_m = Delta/k_F = 1,
    m_m = k_F/v_m = k_F^2/Delta = 1.

These parameters do NOT correspond to obvious scales in the material. In
particular, v_m is distinct from the sound speed c_s, and m_m is distinct from
the carrier mass m_star. In other words, c_s_m != 1 and m_star_m != 1.

We still need to be able to express a unit of length in order to put hbar into
these units. So far, however, time and length always appear in ratio. It
seems, then, that the natural thing to do is to simply set

    hbar = 1.

The natural units of time and length are then

    t_m = hbar/Delta = 1,
    l_m = hbar/Delta*v_m = hbar*k_F/Delta^2 = 1.

It is useful to distinguish quantities in ordinary units from those in
material units. To that end, we will append "_m" to the names of quantities
that are defined in material units.

For uniformity, ALL quantities not in material units are assumed to be in
units of an appropriate power of eV with c=hbar=kB=1.

"""


import numpy as np
import tensorflow as tf


OBJECT_KEYS = (
    'gamma', 'c_s', 'T_c', 'beta', 'cache_qp_energy', 'symbol', 'm_star_ratio'
)


class Material(object):
    """Container class for material properties.

    Args:
        symbol (str): chemical symbol.
        gamma (float): gamma parameter.
        c_s (float): speed of sound in m/s.
        T_c (float): critical temperature in K.
        Delta (float): half of the gap in eV.
        E_F (float): Fermi energy in eV.
        m_star (float): effective mass? in units of m_e.
        beta (float): beta.

    Attributes:
        symbol (str): chemical symbol.
        gamma (float): gamma parameter.
        c_s (float): speed of sound in m/s.
        T_c (float): critical temperature in K.
        Delta (float): half of the gap.
        E_F (float): Fermi energy in eV.
        m_star (float): effective mass? in units of m_e.
        z (float): Fermi energy in units of Delta.
        beta (float): beta.
        mcs2 (float): m*c_s^2 in units of eV.

    """

    def __init__(self, **kwargs):
        self._cache_qp_energy = None
        m_star_ratio = kwargs.get('m_star_ratio', 1)
        self.m_star = 511e3 * m_star_ratio
        setattr(self, 'E_F', kwargs.get('E_F', 11.7))
        setattr(self, 'Delta', kwargs.get('Delta', 1e-3))
        self.ep_cache = None
        for key in OBJECT_KEYS:
            setattr(self, key, kwargs.get(key))
        # Trivial quantities
        self.hbar = 1.
        self.hbar_m = 1.
        self.Delta_m = 1.
        self.k_F_m = 1.
        self.t_m = 1.
        self.v_m = 1.
        self.m_m = 1.
        self._update_derived()

    def _update_derived(self):
        """Set derived quantities in material and non-material units."""
        # Define base quantities in non-material units
        self.k_F = np.sqrt(2 * self.m_star * self.E_F)
        self.v_F = self.k_F / self.m_star
        self.c_s = self.gamma / np.sqrt(self.m_star/(2*self.Delta))
        self.t = self.hbar / self.Delta
        self.v = self.Delta / self.k_F
        self.m = self.k_F / self.v
        # Derived quantities
        self.z = self.E_F_m = self.E_F / self.Delta
        self.m_star_m = self.m_star / self.m
        self.v_F_m = self.v_F / self.v
        self.c_s_m = self.c_s / self.v
        # Momentum of an E=Delta excitation above the Fermi surface
        self.delta_k = np.sqrt(1 + 2*np.sqrt(3)*self.m_star_m) - 1

    def info(self):
        """Material data in a json-serializable format.

        Returns:
            dict: material info.

        """
        return dict(
            symbol=self.symbol,
            gamma=self.gamma,
            c_s=self.c_s,
            T_c=self.T_c,
            Delta=self.Delta,
            E_F=self.E_F,
            m_star=self.m_star,
            beta=self.beta
        )

    def epsilon_lindhard(self, q, omega):
        """Lindhard dielectric function."""
        k_F = self.k_F_m
        v_F = self.v_F_m
        e_charge = np.sqrt(1. / 137.)
        lambda_TF = e_charge / np.pi * (2*self.z*self.m_star_m**3)**(1./4.)
        omega_p = lambda_TF * v_F / np.sqrt(3)
        # Following Dressel and Gruner eq. 5.4.22
        # We break down the calculation into chunks for ease of debugging
        e1a = (lambda_TF**2 / q**2)
        e1b = k_F/(4*q) * (
            1 - (q/(2*k_F) - omega/(q*v_F))**2
        )
        e1c = tf.cast(
            tf.math.log(tf.abs(
                (q/(2*k_F) - omega/(q*v_F) + 1) /
                (q/(2*k_F) - omega/(q*v_F) - 1)
            )), tf.complex128
        )
        e1d = k_F/(4*q) * (
            1 - (q/(2*k_F) + omega/(q*v_F))**2
        )
        e1e = tf.cast(
            tf.math.log(tf.abs(
                (q/(2*k_F) + omega/(q*v_F) + 1) /
                (q/(2*k_F) + omega/(q*v_F) - 1)
            )), tf.complex128
        )
        e1 = 1 + e1a * (0.5 + e1b * e1c + e1d * e1e)
        condition_1 = tf.math.real(q/(2*k_F) + omega/(q*v_F)) < 1
        condition_2 = tf.math.logical_and(
            tf.abs(q/(2*k_F) - omega/(q*v_F)) < 1,
            1 < tf.math.real(q/(2*k_F) + omega/(q*v_F))
        )
        e2 = tf.where(
            condition_1,
            3*np.pi * omega_p**2 * omega / (2*q**3 * v_F**3),
            tf.where(
                condition_2,
                3*np.pi * omega_p**2 * k_F / (4*q**3 * v_F**2) * (
                    1 - (q/(2*k_F) - omega/(q*v_F))**2
                ),
                0
            )
        )
        return e1 + tf.cast(e2, tf.complex128)*1j

    def coherence_uvvu(self, s, k1, k2):
        """The coherence factor (uv' +/- vu')^2. The sign is `s`.

        Note that `k1` and `k2` are assumed to be in material units (i.e. in
        units of k_F).

        Args:
            s (int): sign that appears in the coherence factor (1 or -1).
            k1 (float): momentum of quasiparticle 1 in material units.
            k2 (float): momentum of quasiparticle 2 in material units.

        Returns:
            float: the coherence factor (uv' +/- vu')^2.

        """
        xi1 = (k1**2 - self.k_F_m**2) / (2*self.m_star_m) + 0j
        xi2 = (k2**2 - self.k_F_m**2) / (2*self.m_star_m) + 0j
        e1 = tf.sqrt(self.Delta_m**2 + xi1**2 + 0j)
        e2 = tf.sqrt(self.Delta_m**2 + xi2**2 + 0j)
        v1 = tf.sqrt((1 - xi1/e1)/2. + 0j)
        v2 = tf.sqrt((1 - xi2/e2)/2. + 0j)
        u1 = tf.sqrt(1 - v1**2 + 0j)
        u2 = tf.sqrt(1 - v2**2 + 0j)
        return (v1*u2 + s*u1*v2)**2

    def qpe(self, k):
        """Compute the energy of a quasiparticle of momentum :math:`k`.

        This is a convenience method that is identical to the ``Quasiparticle``
        dispersion relation. In fact, the two might be merged later on...

        Args:
            k: QP momentum

        Returns:
            float: QP energy in material units

        """
        return np.sqrt(
            1 + (
                (k*self.hbar_m)**2 / (2*self.m_star_m) - self.z
            )**2
        )
