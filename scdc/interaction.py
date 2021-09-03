"""This module defines the :obj:`Interaction` class and subclasses.

This is where most of the action lives. Particles propagate from interaction
to interaction, and the interaction objects determine the final states that
arise at each point. The differential rates, for instance, are used here.

"""

import numpy as np
from scipy.optimize import bisect

from .common import (
    NonPhysicalException, NotAllowedException, tolerate, cosine_add,
    Region, Interval
)
from .particle import Quasiparticle, Phonon, DarkMatter, ParticleCollection


# A "small number" that's not zero for numerical purposes
SMALL_NUMBER = 1e-10

# Sometimes it's useful to have all possible sign combinations as an array
SIGNS_1 = np.asarray([1, -1])
SIGNS_2 = np.asarray([
    [1,   1],
    [1,  -1],
    [-1,  1],
    [-1, -1]
])


class Interaction(object):
    """Parent class for interactions.

    An interaction combines an initial state, a material, and a final state.
    Subclasses must provide the means to go from the initial state to the final
    state.

    Interactions are inherently treelike and can be treated as such. The
    complication is that the initial and final states are `Particle` objects,
    not `Interaction` objects. This necessitates the introduction of a new
    container type, `InteractionNode`, which specifies a particle and
    the interaction in which it participates.

    Args:
        initial_state (:obj:`list` of :obj:`Particle`): particles in the
            initial state. The initial state is always a list, even if it
            contains only one object.
        material (obj:`Material`): the material in which this takes place.
        n_bins (int, optional): number of bins to use for certain discretized
            calculations. Defaults to 100.
        final_state (:obj:`list` of :obj:`Particle`, optional): particles in
            the final state. If supplied, the interaction is treated as trivial
            and the final state is never overwritten.
        final (bool, optional): if `True`, the interaction is treated as
            trivial and the final state is left unpopulated and unvalidated.
            Defaults to `False`.

    Attributes:
        initial_state (:obj:`list` of :obj:`Particle`): particles in the
            initial state. The initial state is always a list, even if it
            contains only one object.
        material (obj:`Material`): the material in which this takes place.
        final (bool, optional): if `True`, the interaction is treated as
            trivial and the final state is left unpopulated and unvalidated.
        n_initial (int): how many particles should be in the initial state.
        n_final (int): how many particles should be in the final state.
        initial (:obj:`tuple` of `type`): types of the initial particles,
            listed in order of their appearance.
        final (:obj:`tuple` of `type`): types of the final particles, listed in
            order of their appearance.
        ip (:obj:`Particle`): first of the initial particles.
        fixed_final_state (bool): `True` if the final state was supplied at
            initialization. In this case, the interaction is treated as trivial
            and the final state is never overwritten.

    """
    initial = ()
    final = ()

    def __init__(self, initial_state, material, *args, **kwargs):
        self.initial_state, self.material = initial_state, material
        self.final = kwargs.get('final', False)
        self.n_bins = kwargs.get('n_bins', 100)
        self.n_initial = len(self.__class__.initial)
        self.n_final = len(self.__class__.final)
        # Check that the initial state matches the specification
        if len(self.initial_state) != self.n_initial:
            raise ValueError("Wrong number of particles in initial state")
        for ptype, particle in zip(self.__class__.initial, self.initial_state):
            if not isinstance(particle, ptype):
                raise ValueError("Initial state does not match specification")
        # As a convenience for 1-particle initial states, we define:
        self.ip = self.initial_state[0]
        # Check if a final state is supplied
        self.final_state = kwargs.get('final_state')
        if self.final:
            self.final_state = []
            self.fixed_final_state = True
        elif self.final_state is not None:
            self._validate_final(self.final_state)
            self.fixed_final_state = True
        else:
            self.fixed_final_state = False

    @classmethod
    def valid_initial(cls, initial_state):
        """Check that an initial state contains the correct particle types.

        Args:
            initial_state (:obj:`list` of :obj:`Particle`): candidate initial
                state to validate.

        Returns:
            bool: `True` if the supplied initial state has the right particle
                types for this interaction. Otherwise `False`.

        """
        if len(initial_state) != len(cls.initial):
            return False
        for particle, PType in zip(initial_state, cls.initial):
            if not isinstance(particle, PType):
                return False
        return True

    def _validate_final(self, final_state):
        """Check that a final state contains the correct particle types.

        Unlike initial-state checking, the code should always fail if an
        invalid final state shows up. Thus, this does not return a boolean,
        but instead raises an error in the event of a mismatch.

        Args:
            final_state (:obj:`list` of :obj:`Particle`): candidate final
                state to validate.

        Raises:
            RuntimeError: the supplied final state does not match the
            specifications of this interaction.

        """
        if len(final_state) != self.n_final:
            raise RuntimeError("Wrong number of particles in final state")
        for ptype, particle in zip(self.__class__.final, final_state):
            if not isinstance(particle, ptype):
                raise RuntimeError("Final state does not match specification")

    def allowed(self):
        """Determine whether the process is allowed for this initial state.

        Returns:
            bool: `True` if the process can occur. Otherwise `False`.

        """
        raise NotImplementedError

    def _interact(self):
        """Compute and return the final state.

        Returns:
            :obj:`list` of :obj:`Particle`: the final-state particles.

        """
        raise NotImplementedError

    def interact(self):
        """Interact, validate outcome, and set final state.

        If the final state was supplied at initialization, do nothing.

        Returns:
            :obj:`list` of :obj:`Particle`: the final state.

        """
        if self.fixed_final_state:
            return self.final_state
        if not self.allowed():
            raise NotAllowedException
        final_state = self._interact()
        self._validate_final(final_state)
        self.final_state = final_state
        return self.final_state


class DarkMatterScatter(Interaction):
    """Represents an interaction in which DM scatters and creates two QPs.

    This is a container, in the sense that no DM dynamics are actually
    contained here. Rather, at initialization, two quasiparticles must be
    supplied in the final state.

    Args:
        qp1 (:obj:`Quasiparticle`): final state quasiparticle 1.
        qp2 (:obj:`Quasiparticle`): final state quasiparticle 2.

    """
    initial = (DarkMatter,)
    final = (DarkMatter, Quasiparticle, Quasiparticle)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.fixed_final_state:
            raise ValueError("Must supply final state for DM scatter")

    def allowed(self):
        return True

    def _interact(self):
        return self.final_state


class QuasiparticlePhononEmission(Interaction):
    """The `Interaction` for a quasiparticle to emit a phonon.

    Args:
        tolerance (float, optional): a numerical tolerance for root-finding.
            Defaults to 1e-3.

    Attributes:
        tolerance (float, optional): a numerical tolerance for root-finding.

    """
    initial = (Quasiparticle,)
    final = (Quasiparticle, Phonon)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = kwargs.get('tolerance', 1e-3)

    def min_cos_deflection(self):
        """Max angular deflection, expressed as minimum of cos(theta).

        For a given QP energy, there is a maximum amount of momentum it can
        shed via phonon emission. This produces an upper bound on the angular
        deflection of the QP. We express this in the form of a lower bound on
        the cosine of the deflection angle.

        Returns:
            float: min of cos(theta).

        """
        y = self.ip.energy
        z = self.material.z
        g = self.material.gamma
        rsy = np.sqrt(y**2 - 1)
        rsyz = np.sqrt(rsy + z)
        return (
            1 - y + 2*g*rsyz
        ) / np.sqrt(2) / np.sqrt(
            1 + y**2 - 2*g*rsyz + 2*g**2 * rsyz**2 + 2*y*(g*rsyz - 1)
        )

    def final_state_angles(self, phonon_energy, sp):
        """Cosine of angle of final state particles after QP scattering.

        Both angles are computed relative to the direction of the proximal
        quasiparticle just before scattering, not necessarily the initial
        quasiparticle.

        DEBUGGING NOTE: this follows section "Better" of `dispersion.nb`.

        Args:
            phonon_energy (float): phonon energy in units of Delta.
            qp_energy (float): Initial quasiparticle energy in units of Delta.
            sp (:obj:`list` of int): momentum sign for final quasiparticle.

        Returns:
            float: cosine of the angle of the emitted quasiparticle.
            float: cosine of the angle of the emitted phonon.

        """
        # For brevity and to match the notation in the note:
        x = phonon_energy
        y = self.ip.energy
        z = self.material.z
        rsy = self.ip.ksign * np.sqrt(y**2 - 1)
        """
        Sometimes, due to numerical precision problems, the following root
        can get a negative argument. This will only happen if that negative
        argument is exceedingly small, and in such a case, it can be safely
        set to zero, as follows.
        """
        root_arg = y**2 - 2*x*y + x**2 - 1
        if root_arg < 0 and root_arg > -1e-7:
            root_arg = 0
        elif root_arg < 0:
            # Danger:
            raise NotAllowedException
            raise RuntimeError(
                "root_arg appears to be negative (%e)" % root_arg
            )
        rsyz = sp * np.sqrt(root_arg)
        g2 = self.material.gamma**2
        cos_qp_angle = (-x**2 + 4*g2*(rsy + rsyz + 2*z)) / (
            8*g2*np.sqrt((rsy+z) * (rsyz+z))
        )
        cos_phonon_angle = (x**2 + 4*g2*(rsy-rsyz)) / (
            4*x*self.material.gamma*np.sqrt(rsy+z)
        )
        return cos_qp_angle, cos_phonon_angle

    def phonon_energy_region(self):
        """Min / max allowed phonon energy for emission by a quasiparticle.

        The min and max energies are determined by finding the energy range
        in which the final-state angles are physical, i.e.

            :math:`|\\cos(\\theta)| < 1`.

        This needs to be done for both the quasiparticle and phonon angle.
        Moreover, since there is a sign ambiguity in the quasiparticle
        momentum, this needs to be done for each sign independently to get
        self-consistent regions that work for both the quasiparticle angle and
        the phonon angle.

        Thus, we take the following approach. For each sign in turn, we find
        the physical region for each of the two angles. The intersection of
        these two regions is what we want. But then we have to do this for each
        sign, and take the union of the resulting intersections.

        In principle, this may be disjoint. For the moment, rather than
        significantly change the architecture of the code, I will simply throw
        an error if the regions are disjoint.

        Args:
            qp_energy (float): Initial quasiparticle energy in units of Delta.

        Returns:
            float: minimum emitted phonon energy in units of Delta.
            float: maximum emitted phonon energy in units of Delta.

        """
        # Test whether any values are allowed and exit if we can
        qpe = self.ip.energy
        if qpe <= 1:
            raise NotAllowedException
        # Define the allowed region for one function
        UPPER_EDGE = qpe - 1
        # We will neglect emissions with energy less than `SMALL_NUMBER`
        if UPPER_EDGE <= SMALL_NUMBER:
            raise NotAllowedException

        def _find_crossing(func, val):
            """Find the crossing of a function with a value.

            Args:
                func (function): a function of one variable which returns a
                    single value.
                val (float): the value with which to find a crossing.

            Returns:
                float: location of the crossing.

            Raises:
                ValueError: if there is no crossing.

            """
            # Set root-finding tolerance. These must be better than those used
            # to assess physicality of cosines.
            root_tol = 1e-1 * self.tolerance
            btol = 1e-2 * self.tolerance
            try:
                crossing = bisect(
                    lambda eph: func(eph) - val, SMALL_NUMBER, UPPER_EDGE,
                    xtol=btol, rtol=btol
                )
            except ValueError:
                raise ValueError("\tNo crossing")
            # Test to make sure the crossing is located within tolerance
            while np.abs(func(crossing) - val) > root_tol:
                # Reduce the bisection tolerance and try again
                btol /= 10.
                crossing = bisect(
                    lambda eph: func(eph) - val, SMALL_NUMBER, UPPER_EDGE,
                    xtol=btol, rtol=btol
                )
            return crossing

        def _allowed_region(func):
            """Find the region in which an angle is valid.

            Args:
                func (function): a function of one variable (the phonon energy)
                    which returns a single value, the angle in question.

            Returns:
                float: the lower bound of the region.
                float: the upper bound of the region.

            Raises:
                NotAllowedException: if no region is allowed.

            """
            try:
                upper_crossing = _find_crossing(func, 1)
            except ValueError:
                if func(UPPER_EDGE) < 1:
                    # The whole region is OK
                    upper_crossing = None
                else:
                    # The whole region is forbidden
                    raise NotAllowedException
            else:
                # Crossing at the upper edge is the same as no crossing
                if upper_crossing == UPPER_EDGE:
                    upper_crossing = None
            try:
                lower_crossing = _find_crossing(func, -1)
            except ValueError:
                if func(SMALL_NUMBER) > -1:
                    # The whole region is OK
                    lower_crossing = None
                else:
                    # The whole region is forbidden
                    raise NotAllowedException
            else:
                # Crossing at the lower edge is the same as no crossing
                if lower_crossing == 0:
                    lower_crossing = None
            # Now each of `lower_crossing` and `upper_crossing` can be `None`.
            if lower_crossing is None and upper_crossing is None:
                # If both are `None`, then everything is OK.
                return Interval(0, UPPER_EDGE)
            elif lower_crossing is None:
                # Only one side of `upper_crossing` is OK.
                if func(SMALL_NUMBER) < 1:
                    # Left side is OK
                    return Interval(0, upper_crossing)
                else:
                    # Right side is OK
                    return Interval(upper_crossing, UPPER_EDGE)
            elif upper_crossing is None:
                # Only one side of `lower_crossing` is OK.
                if func(SMALL_NUMBER) > -1:
                    # Left side is OK
                    return Interval(0, lower_crossing)
                else:
                    # Right side is OK
                    return Interval(lower_crossing, UPPER_EDGE)
            else:
                # The region between the two crossings is OK.
                if lower_crossing == upper_crossing:
                    # Crossings are equal! That's not allowed.
                    raise NotAllowedException
                return Interval(lower_crossing, upper_crossing)
        #
        # Find the allowed region for each sign and angle
        allowed_region = Region()
        for kp_sign in SIGNS_1:
            try:
                # Both the QP and phonon angle must be physical, or this sign
                # isn't viable
                interval_0 = _allowed_region(
                    lambda eph: self.final_state_angles(eph, kp_sign)[0]
                )
                interval_1 = _allowed_region(
                    lambda eph: self.final_state_angles(eph, kp_sign)[1]
                )
            except NotAllowedException:
                # At least one angle was non-physical
                continue
            # Find the intersection of these two regions
            try:
                intersection = interval_0.intersection(interval_1)
            except ValueError:
                # The regions are disjoint
                continue
            else:
                allowed_region.add(intersection)
        if allowed_region.n() == 0:
            # No regions were allowed
            raise NotAllowedException
        if allowed_region.measure() == 0:
            # raise RuntimeError("Something has gone wrong: zero-size region")
            # I think this is OK, just rare, and reflects that this process
            # is very nearly prohibited kinematically. TODO: check on this.
            raise NotAllowedException
        return allowed_region

    def phonon_energy_distribution(self):
        """Relative phonon emission probability by a quasiparticle.

        Computation follows dGamma/dx in eq. (21) of Noah's note. We normalize
        directly. If the distribution cannot be normalized (all values vanish),
        both return values are `None`.

        Returns:
            :obj:`ndarray` of :obj:`float`: phonon energy bin centers.
            :obj:`ndarray` of :obj:`float`: relative rate for each bin center,
                normalized to unity.

        Raises:
            NonPhysicalException: If the max phonon energy is not positive.

        """
        # Compute bins and centers
        qp_energy = self.ip.energy
        e_region = self.phonon_energy_region()
        e_bin_edge_groups = e_region.linspace(self.n_bins)
        # `e_bin_edge_groups` is a list of arrays, one for each interval
        energies, rates = [], []
        for e_bin_edges in e_bin_edge_groups:
            e_bins = np.vstack((e_bin_edges[:-1], e_bin_edges[1:])).T
            phonon_energy = np.mean(e_bins, axis=1)
            relative_rates = np.zeros_like(phonon_energy)
            e_diff = qp_energy - phonon_energy
            de_mask = e_diff > 1
            relative_rates[de_mask] = phonon_energy[de_mask]**2 * np.real(
                e_diff[de_mask] / np.sqrt(e_diff[de_mask]**2 - 1)
            ) * (
                1 - 1./(qp_energy*(e_diff[de_mask]))
            )
            # Get rid of numerical issues
            n_mask = np.isfinite(relative_rates) & (relative_rates > 0)
            mask = n_mask & de_mask
            if not np.count_nonzero(mask):
                continue
            # Normalize
            energies.extend(phonon_energy[mask])
            rates.extend(relative_rates[mask])
        rates = np.asarray(rates)
        rates /= np.sum(rates)
        if np.count_nonzero(rates) == 0:
            raise NonPhysicalException
        return np.asarray(energies), rates

    def allowed(self, exact=False):
        return True
        """
        if not exact:
            if 1 - self.min_cos_deflection() < self.tolerance:
                return True
        return False
        """

    def _interact(self):
        """Emit a phonon with energy sampled from the right distribution.

        Throughout this method, `qpf` stands for "final-state quasiparticle".

        Returns:
            :obj:`ParticleCollection`: the final-state particles.

        """
        # Sample the phonon energy
        try:
            e, p = self.phonon_energy_distribution()
        except NonPhysicalException:
            raise NotAllowedException
        phonon_energy = np.random.choice(e, p=p)
        qpf_energy = self.ip.energy - phonon_energy
        # "ctr" = cos theta relative
        # Compute this for each sign combination and choose a physical one
        qpf_ctr_candidates, phonon_ctr_candidates = \
            self.final_state_angles(phonon_energy, SIGNS_1)
        qpf_ctr_candidates = tolerate(
            qpf_ctr_candidates, self.tolerance
        )
        phonon_ctr_candidates = tolerate(
            phonon_ctr_candidates, self.tolerance
        )
        # Identify the sign pairs that result in physical angles
        mask = (-1 <= qpf_ctr_candidates) & (qpf_ctr_candidates <= 1) & \
               (-1 <= phonon_ctr_candidates) & (phonon_ctr_candidates <= 1)
        if not np.any(mask):
            raise NonPhysicalException("No candidate angles are physical")
        # Choose the sign for the outgoing quasiparticle
        index = np.random.choice(np.arange(len(mask))[mask])
        kpsign = SIGNS_1[index]
        qpf_ctr = qpf_ctr_candidates[index]
        phonon_ctr = phonon_ctr_candidates[index]
        # Choose azimuthal angle
        phi = 2*np.pi*np.random.random()
        # Convert relative to absolute angles
        phonon_cos_theta = cosine_add(self.ip.cos_theta, phonon_ctr, phi)
        qpf_cos_theta = cosine_add(self.ip.cos_theta, qpf_ctr, phi + np.pi)
        qpf_cos_theta = tolerate(qpf_cos_theta, self.tolerance)
        # Create the final-state phonon
        phonon = Phonon(
            energy=phonon_energy,
            cos_theta=phonon_cos_theta,
            material=self.material
        )
        if ~np.isfinite(qpf_cos_theta):
            raise NonPhysicalException
        # Create the final-state quasiparticle
        final_qp = Quasiparticle(
            energy=qpf_energy,
            cos_theta=qpf_cos_theta,
            material=self.material,
            ksign=kpsign
        )
        return ParticleCollection([final_qp, phonon])


class PhononDecayToQuasiparticles(Interaction):
    """The `Interaction` for a phonon to decay to two quasiparticles.

    Args:
        tolerance (float, optional): a numerical tolerance for root-finding.
            Defaults to 1e-3.

    Attributes:
        tolerance (float, optional): a numerical tolerance for root-finding.

    """
    initial = (Phonon,)
    final = (Quasiparticle, Quasiparticle)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tolerance = kwargs.get('tolerance', 1e-3)

    def decay_angles(self, Eqp1, Eqp2, s1, s2):
        """Computes the relative angles of two QPs created in phonon decay.

        Args:
            Eqp1 (float): energy of quasiparticle 1.
            Eqp2 (float): energy of quasiparticle 2.
            s1 (int): sign appearing in the inverse dispersion relation (QP1).
            s2 (int): sign appearing in the inverse dispersion relation (QP2).

        Returns:
            float: relative cos(theta) of quasiparticle 1.
            float: relative cos(theta) of quasiparticle 2.

        """
        Eq = Eqp1 + Eqp2
        z = self.material.z
        r1 = np.sqrt(
            2*self.material.m_star_m * (z + s1*np.sqrt(Eqp1**2 - 1))
        ) / self.material.hbar_m
        r2 = np.sqrt(
            2*self.material.m_star_m * (z + s2*np.sqrt(Eqp2**2 - 1))
        ) / self.material.hbar_m
        cos_theta_1 = (
            Eq**2 + 4*self.material.gamma**2 * (r1 - r2)
        ) / (
            4*Eq*self.material.gamma * np.sqrt(z + r1)
        )
        cos_theta_2 = (
            Eq**2 + 4*self.material.gamma**2 * (r2 - r1)
        ) / (
            4*Eq*self.material.gamma * np.sqrt(z + r2)
        )
        return cos_theta_1, cos_theta_2

    def qp_energy_distribution(self, eps=1e-10):
        """Relative probability for a quasiparticle energy in phonon decay.

        This is basically the integrand of eq. (27) of Kaplan+ 1976.

        Args:
            eps (float, optional): slight offset to prevent divide-by-zero
                errors. Defaults to 1e-10.

        Returns:
            :obj:`ndarray` of :obj:`float`: QP energy bin centers.
            :obj:`ndarray` of :obj:`float`: relative rate for each bin center,
                normalized to unity.

        """
        phonon_energy = self.ip.energy
        if phonon_energy < 2*(1+eps):
            raise NonPhysicalException(
                "A phonon with E < 2Delta cannot produce a quasiparticle pair"
            )
        qp_bin_edges = np.linspace(
            1 + eps, phonon_energy - 1 - eps, self.n_bins
        )
        qp_bins = np.vstack((
            qp_bin_edges[:-1], qp_bin_edges[1:]
        )).T
        qp_energy = np.mean(qp_bins, axis=1)
        # Here's the actual calculation
        relative_rates = 1. / np.sqrt(qp_energy**2 - 1) * (
            qp_energy * (phonon_energy - qp_energy) + 1
        ) / np.sqrt((phonon_energy - qp_energy)**2 - 1)
        # Normalize
        norm = np.sum(relative_rates)
        return qp_energy, relative_rates / norm

    def allowed(self):
        """Determine whether the process is allowed.

        Phonon decay can always take place if the energy of the phonon is at
        least 2*Delta.

        Returns:
            bool: `True` if phonon decay can occur. Otherwise `False`.

        """
        return self.ip.energy >= 2

    def _interact(self):
        """Produce final-state quasiparticles.

        Returns:
            :obj:`ParticleCollection`: final-state particles.

        """
        # Sample from the QP energy
        try:
            e, p = self.qp_energy_distribution()
        except NonPhysicalException:
            raise NotAllowedException
        qp1_energy = np.random.choice(e, p=p)
        qp2_energy = self.ip.energy - qp1_energy
        # Compute the final (relative) angles for each pair of signs s1, s2
        c12_candidates = np.zeros((len(SIGNS_2), 2))
        for i, (s1, s2) in enumerate(SIGNS_2):
            c12_candidates[i] = tolerate(
                self.decay_angles(qp1_energy, qp2_energy, s1, s2),
                100*self.tolerance  # TODO: that 100 shouldn't be there
            )
        # Identify the sign pairs that result in physical angles
        mask = np.all(
            (-1 <= c12_candidates) & (c12_candidates <= 1),
            axis=1
        )
        if not np.any(mask):
            # raise NonPhysicalException("No candidate angles are physical")
            # Danger:
            raise NotAllowedException
        # Choose the signs s1 and s2 for the quasiparticles
        index = np.random.choice(np.arange(len(mask))[mask])
        s1, s2 = SIGNS_2[index]
        c1_relative, c2_relative = c12_candidates[index]
        # Compute the momenta in terms of these signs
        z = self.material.z
        qp1_momentum = np.sqrt(1 + s1 * np.sqrt(qp1_energy**2 - 1) / z)
        qp2_momentum = np.sqrt(1 + s2 * np.sqrt(qp2_energy**2 - 1) / z)
        # Choose the azimuthal angle randomly
        phi = 2*np.pi*np.random.random()
        # Compute the final (absolute) angles
        cos_theta_1 = cosine_add(self.ip.cos_theta, c1_relative, phi)
        cos_theta_2 = cosine_add(self.ip.cos_theta, c2_relative, phi + np.pi)
        # Create and return the final-state particles
        return ParticleCollection([
            Quasiparticle(
                momentum=qp1_momentum,
                cos_theta=cos_theta_1,
                material=self.material
            ),
            Quasiparticle(
                momentum=qp2_momentum,
                cos_theta=cos_theta_2,
                material=self.material
            )
        ])
