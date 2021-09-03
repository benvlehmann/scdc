"""This module defines the distribution of the initial excitations in angles
and momenta, as a function of the scattering matrix element. Unlike prior
implementations using MCMC methods, here the sampling is carried by direct
integration of the distribution.

"""

from itertools import count

import numpy as np
import tensorflow as tf
from scipy.integrate import quad, fixed_quad
from scipy.interpolate import interp2d

from ..halo import SingleVelocityDistribution
from ...common import (
    cosine_add, NoDataException, NotAllowedException, sample_from_pdf,
    scalarize_method
)
from ...event import Event
from ...ensemble import Ensemble
from ...particle import DarkMatter, Quasiparticle


SIGNS = np.array([-1, 1])


class RateIntegrator(object):
    """Base class for integrating rates

    Args:
        m1 (float): mass of the incoming particle.
        matrix_element (callable): the matrix element for the scattering
            process, as a function of the magnitude of the 3-momentum transfer.
            A :obj:`ScatteringMatrixElement` object can be supplied here.
        material (:obj:`Material`): a material object.
        response (callable): a function of :math:`q` and :math:`\\omega`
            characterizing material response.
        vdf (callable): a probability distribution for the velocity as a
            function of zero, one, or two parameters. The number of arguments
            must match the length of the argument ``v_args`` to ``likelihood``.

    Attributes:
        m1 (float): mass of the incoming particle.
        matrix_element (callable): the matrix element for the scattering
            process, as a function of the magnitude of the 3-momentum transfer.
            A :obj:`ScatteringMatrixElement` object can be supplied here.
        material (:obj:`Material`): a material object.
        response (callable): a function of :math:`q` and :math:`\\omega`
            characterizing material response.
        vdf (callable): a probability distribution for the velocity as a
            function of zero, one, or two parameters. The number of arguments
            must match the length of the argument ``v_args`` to ``likelihood``.

    """
    def __init__(self, m1, matrix_element, material, response, vdf, **kwargs):
        self.m1 = m1
        self.matrix_element = matrix_element
        self.material = material
        self.response = response
        self.vdf = vdf
        self.omega_max = kwargs.get('omega_max')

    def _cq3(self, r1, r2, r3, rq, s):
        """Computes the cosine of the angle between :math:`q` and :math:`r_3`.

        Note that for the given arguments, :math:`\\cos\\theta(q,r_3)` has two
        values, differentiated by a sign :math:`s`. This sign must also be
        supplied as an argument.

        Args:
            r1: momentum of incoming particle.
            r2: momentum of outgoing particle.
            r3: momentum of one of the final-state quasiparticles.
            rq: momentum transfer.
            s: a sign determining which solution to use.

        Returns:
            float: :math:`\\cos\\theta(q,r_3)`.

        Raises:
            ValueError: if either :math:`r_3` or :math:`r_q` is zero.

        """
        if np.any(r3 == 0):
            raise ValueError("r3 is zero")
        if np.any(rq == 0):
            raise ValueError("rq is zero")
        E1 = r1**2/(2*self.m1)
        E2 = r2**2/(2*self.m1)
        E3 = np.sqrt(
            self.material.Delta_m**2 + (
                r3**2/(2*self.material.m_star_m) - self.material.E_F_m
            )**2
        )
        cq3 = (
            # -2*self.material.E_F_m * self.material.m_star_m + r3**2 +
            -self.material.k_F_m**2 + r3**2 +
            rq**2 + s * 2*self.material.m_star_m * np.sqrt(
                (E2 + E3 - E1)**2
                - self.material.Delta_m**2 + 0j
            )
        ) / (2*r3*rq)
        # assert not np.any(np.abs(cq3) > 1)
        """We actually have to tolerate such values: this method is used in a
        generic quadrature algorithm, which means that non-physical inputs may
        be supplied. We just need to give them pdf values of zero. The same is
        true for non-physical ``r3`` values that lead to complex ``cq3``."""
        return cq3

    def _omega(self, r1, rq, cq):
        """Compute the deposited energy.

        Args:
            r1 (float): :math:`r_1`.
            rq (float): :math:`r_q`.
            cq (float): :math:`c_q`.

        Returns:
            float: deposited energy.

        """
        return (2*r1*rq*cq - rq**2) / (2*self.m1)

    def _jac(self, rq, r3, cq3):
        """Compute the "Jacobian" part of the differential rate.

        This arises from evaluating one integral with a delta function in a
        related variable.

        Args:
            rq (float): :math:`r_q`.
            r3 (float): :math:`r_3`.
            cq3 (float): :math:`c_{q3}`.

        Returns:
            float: the Jacobian.

        """
        # x = -2*self.material.E_F_m*self.material.m_star_m \
        #    + r3**2 - 2*cq3*r3*rq + rq**2
        """The above is numerically dangerous. The first term is the Fermi
        momentum in material units, which is 1. The second term, r3**2, is very
        close to 1. That's a cancelation and it's asking for trouble. What we
        can do instead is the following: write

            r3 = k_F + dr3
            r3**2 = k_F**2 + dr3**2 + 2*k_F*dr3

        and therefore we have 

            -k_F**2 + r3**2 = dr3**2 + 2*k_F*dr3.

        """
        dr3 = r3 - self.material.k_F_m
        x = dr3**2 + 2*self.material.k_F_m*dr3 - 2*cq3*r3*rq + rq**2
        """Rarely, due to numerical precision problems, x is zero. We should
        give a result of zero for these points."""
        mask = (x == 0)
        # Set these bad values to 1 so they don't cause errors
        x[mask] = 1
        jac = np.abs(
            self.material.m_star_m * np.sqrt(
                x**2 + 4*self.material.m_star_m**2*self.material.Delta_m**2
                + 0j
            ) / (r3 * rq * x)
        )
        # Eliminate the bad values
        jac[mask] = 0
        return jac

    def pdf_fixed_sign(self, r1, rq, cq, r3, s):
        """Compute the pdf for fixed sign :math:`s`.

        Args:
            r1 (float): :math:`r_1`.
            rq (float): :math:`r_q`.
            cq (float): :math:`c_q`.
            r3 (float): :math:`r_3`.
            s (float): the sign :math:`s` selecting between the two solutions
                for the second quasiparticle's momentum.

        Returns:
            float: differential probability.

        Raises:
            ValueError: if :math:`r_q = 0`.

        """
        if np.any(rq == 0):
            raise ValueError
        r1 = np.asarray(r1).reshape((-1,))[:, None]
        rq = np.asarray(rq).reshape((-1,))[:, None]
        cq = np.asarray(cq).reshape((-1,))[:, None]
        r3 = np.asarray(r3).reshape((-1,))[:, None]
        # THE FOLLOWING LINE HAD A CRITICAL (DIMENSIONAL) ERROR
        r2 = np.sqrt(rq**2 + r1**2 - 2*rq*r1*cq + 0j)
        cq3 = self._cq3(r1, r2, r3, rq, s)
        r4 = np.sqrt(rq**2 + r3**2 - 2*rq*r3*cq3 + 0j)
        result = \
            self.matrix_element(rq) \
            * self.response(r3, r4, rq, self._omega(r1, rq, cq)) \
            * self._jac(rq, r3, cq3) \
            * r3**2
        deposit_condition = 2*r1*rq*cq - rq**2 > 0
        """While it is inefficient, we need to calculate c2 to check that our
        results are physical. c2x is found by solving the system

            p1 = p2 + pq <--> {
                0  = r2 s2 + rq sq,
                r1 = r2 c2 + rq cq
            }

        Be careful, though: here c2x and cq are both defined with respect to
        r1, not the wind axis.
        """
        c2x = (r1 - cq*rq) / r2
        angle_condition = tf.logical_and(
            tf.logical_and(
                tf.math.imag(cq3) == 0,
                tf.abs(cq3) <= 1
            ),
            tf.logical_and(
                tf.math.imag(cq) == 0,
                tf.abs(cq) <= 1
            ),
            tf.logical_and(
                tf.math.imag(c2x) == 0,
                tf.abs(c2x) <= 1
            )
        )
        condition = tf.logical_and(deposit_condition, angle_condition)
        result = tf.where(condition, result, 0)
        # Filter for complex or nan values
        result = tf.where(tf.math.imag(result) == 0, result, 0)
        result = tf.where(tf.math.is_finite(result), result, 0)
        return result

    def pdf(self, r1, rq, cq, r3):
        """Compute the differential rate summing over sign choices.

        Args:
            r1 (float): :math:`r_1`.
            rq (float): :math:`r_q`.
            cq (float): :math:`c_q`.
            r3 (float): :math:`r_3`.

        Returns:
            float: differential rate (non-normalized).

        """
        components = self.pdf_fixed_sign(r1, rq, cq, r3, SIGNS[None, :])
        return tf.abs(tf.reduce_sum(components, axis=-1))

    def r3_domain(self, r1, rq, cq):
        """Compute the minimum and maximum allowed values of :math:`r_3`.

        Args:
            r1 (float): :math:`r_1`.
            rq (float): :math:`r_q`.
            cq (float): :math:`c_q`.

        Returns:
            float: lower bound on :math:`r_3`.
            float: upper bound on :math:`r_3`.

        """
        om = self._omega(r1, rq, cq)
        mdep2 = om*(-2*self.material.Delta_m+om)
        if mdep2 < 0:
            return (0, 0)
        mdep = np.sqrt(mdep2)
        lo2, hi2 = (
            2*self.material.m_star_m*(self.material.E_F_m - mdep),
            2*self.material.m_star_m*(self.material.E_F_m + mdep)
        )
        if lo2 < 0:
            lo2 = 0
        lo, hi = np.sqrt(lo2), np.sqrt(hi2)
        return lo, hi

    def q_rate(self, r1, rq, cq, order=None):
        """Integrated rate at fixed :math:`r_1` and :math:`q`.

        Args:
            r1 (float): :math:`r_1`.
            rq (float): :math:`r_q`.
            cq (float): :math:`c_q`.

        Returns:
            float: differential rate (non-normalized).

        """
        # First check whether this violates a bound on omega if we have one
        if self.omega_max is not None:
            if self._omega(r1, rq, cq) > self.omega_max:
                return 0
        # Find the limit of integration in r3
        r3_min, r3_max = self.r3_domain(r1, rq, cq)
        if r3_min == r3_max:
            return 0
        if order is None:
            out = quad(
                lambda r3: self.pdf(r1, rq, cq, r3),
                r3_min, r3_max
            )
        else:
            out = fixed_quad(
                lambda r3: self.pdf(r1, rq, cq, r3),
                r3_min, r3_max,
                n=order
            )
        try:
            res, err = out
        except TypeError:
            res = out
        return res


class InitialSampler(RateIntegrator):
    """Sampler for parameter distribution of initial excitations.

    Here's the general strategy:
        1. Start by computing the total rate for each of many speeds.
             - Do this by computing the total rate on a sparse grid of speeds
               and interpolating.
             - The total rate calculation at each speed requires that we
               calculate the rate at many values in the (cq, rq) plane. Store
               these values for later.

        2. Use these together with the marginal distribution of speeds in the
           halo to sample a set of DM speeds.

        3. For each DM speed, interpolate the rates between different slices of
        the (cq, rq) plane, corresponding to different discrete speeds.
        This gives an interpolated set of rates in the (cq, rq) plane at
        the speed of interest.

        4. Interpolate these rates to sample (cq, rq).

        5. Determine two remaining kinematical variables.
             - The azimuthal angle of the final states with respect to the
               momentum transfer can be sampled uniformly.
             - The angle of the incoming DM with respect to the DM wind axis
               has a well-determined marginal distribution at fixed DM speed.

    All arguments required of ``RateIntegrator`` are also required here.

    Args:
        n_cq (int, optional): number of samples to take in cq. Defaults to 50.
        n_rq (int, optional): number of samples to take in rq. Defaults to 500.
        n_v1 (int, optional): number of samples to take in v1. Defaults to 20.
        n_cuts (int, optional): number of times to adaptively change the
            domain of sampling. Defaults to 3.
        n_spline (int, optional): number of samples to use in interpolations.
            Defaults to 200.
        support_threshold (float, optional): fraction of the total rate to
            retain when shrinking the sampling domain. Defaults to 0.9999.

    Attributes:
        n_cq (int): number of samples to take in cq.
        n_rq (int): number of samples to take in rq.
        n_v1 (int): number of samples to take in v1.
        n_cuts (int): number of times to adaptively change the domain of
            sampling. Defaults to 3.
        n_spline (int): number of samples to use in interpolations.
        support_threshold (float): fraction of the total rate to retain when
            shrinking the sampling domain.
        cq_vals (array of float): values at which cq is sampled.
        v1_min (float): minimum value of v1 compatible with scattering given
            the supplied vdf.
        v1_max (float): maximum value of v1 from the vdf.
        v1_vals (array of float): values at which v1 is sampled.
        r1_vals (array of float): momenta corresponding to ``v1_vals``.
        c1_vals (array of float): values at which c1 is sampled.
        rate_fixed_v1_cq_rq (array of float): array giving the rate at fixed
            v1, cq, and rq, with shape ``(n_v1, n_cq, n_rq)``.
        rate_fixed_v1 (array of float): array giving the rate at fixed v1,
            with shape ``(n_v1,)``.
        rq_by_v1 (array of float): array of values at which rq is sampled for
            each v1 value, with shape ``(n_v1, n_rq)``. Note that the sampled
            rq values may be different for each v1.

    Raises:
        NotAllowedException: DM mass is below threshold for scattering for all
            available velocities.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay_init = kwargs.get('delay_init', False)
        self.n_cq = kwargs.get('n_cq', 50)
        self.n_rq = kwargs.get('n_rq', 500)
        self.n_v1 = kwargs.get('n_v1', 20)
        self.n_cuts = kwargs.get('n_cuts', 3)
        self.n_spline = kwargs.get('n_spline', 200)
        self.cq_vals = np.linspace(0, 1, self.n_cq)
        self.support_threshold = kwargs.get('support_threshold')
        if self.support_threshold is None:
            self.support_threshold = 0.9999
        # Determine velocity values to use for sampling
        self.v1_min = max(
            2*np.sqrt(self.m1*self.material.Delta_m) / self.m1,
            self.vdf.v_min
        )
        self.v1_max = self.vdf.v_max
        if self.v1_min > self.v1_max:
            raise NotAllowedException("Scattering is kinematically prohibited "
                                      "at all available DM velocities")
        if self.v1_min == self.v1_max:
            # This is a single-velocity distribution
            self.v1_vals = np.array([self.v1_min])
        else:
            self.v1_vals = np.linspace(self.v1_min, self.v1_max, self.n_v1)
        self.r1_vals = self.m1 * self.v1_vals
        self.c1_vals = np.linspace(-1, 1, 100)
        # Allocate a list of fixed-velocity rate slices in the (cq, rq) plane
        self.rate_fixed_v1_cq_rq = \
            np.zeros((self.v1_vals.size, self.n_cq, self.n_rq))
        self.rate_fixed_v1_total = \
            np.zeros((self.v1_vals.size, self.n_cq, self.n_rq))
        # The sampled values of rq are different for each fixed velocity, so we
        # keep track of them as well.
        self.rq_by_v1 = np.zeros((self.v1_vals.size, self.n_rq))
        # Initialize by computing rates where we need them:
        if not self.delay_init:
            self._compute_rates()

    def _compute_rates(self):
        """Initialize the rate tables by computing at fixed (r1, cq, rq).

        This method populates the attributes ``rate_fixed_v1_cq_rq``,
        ``rq_by_v1``, and ``rate_fixed_v1_total``.

        """
        for i, r1 in enumerate(self.r1_vals):
            _, rq_vals, rates = self.q_rate_grid(r1)
            self.rate_fixed_v1_cq_rq[i] = rates
            self.rq_by_v1[i] = rq_vals
        self.rate_fixed_v1_total = np.sum(
            self.rate_fixed_v1_cq_rq, axis=(1, 2)
        )

    def _sample(self, n_samples):
        """Sample initial- and final-state momenta.

        Args:
            n_samples (int): number of samples to draw.

        Returns:
            ndarray: :math:`r_1` values.
            ndarray: :math:`r_2` values.
            ndarray: :math:`r_3` values.
            ndarray: :math:`r_4` values.
            ndarray: :math:`c_1` values.
            ndarray: :math:`c_2` values.
            ndarray: :math:`c_3` values.
            ndarray: :math:`c_4` values.

        """
        # First check if we have any non-zero points at all
        if np.sum(self.rate_fixed_v1_total) == 0:
            raise NoDataException("The total rate appears to be zero")
        v1s = self._sample_v1(n_samples)
        c1s = self._sample_c1(v1s)
        assert not np.any(np.abs(c1s) > 1)
        r1s = self.m1 * v1s
        cqs = np.zeros_like(v1s)
        assert not np.any(np.abs(cqs) > 1)
        rqs = np.zeros_like(v1s)
        r3s = np.zeros_like(v1s)
        for i, r1 in enumerate(r1s):
            # Try again a few times if we get a ``NoDataException``
            for _ in range(3):
                try:
                    cqs[i], rqs[i], r3s[i] = self._sample_q_fixed_r1(r1)
                except NoDataException:
                    cqs[i], rqs[i], r3s[i] = np.nan, np.nan, np.nan
                except ValueError:
                    cqs[i], rqs[i], r3s[i] = np.nan, np.nan, np.nan
                    break
                else:
                    break
        # Now we need to remove the nans
        mask = np.isfinite(cqs) & np.isfinite(rqs) & np.isfinite(r3s)
        r1s, rqs, r3s, c1s, cqs = \
            r1s[mask], rqs[mask], r3s[mask], c1s[mask], cqs[mask]
        r2s, r3s, r4s, c2s, c3s, c4s = self.q_to_p(r1s, rqs, r3s, c1s, cqs)
        momenta = np.vstack((r1s, r2s, r3s, r4s, c1s, c2s, c3s, c4s)).T
        # Just do one more nan-check to be on the safe side
        mask = ~np.any(np.imag(momenta) != 0, axis=1)
        for col in (4, 5, 6, 7):
            mask = mask & (np.abs(momenta[:, col]) <= 1)
        momenta = momenta[mask]
        return momenta.T

    def sample(self, n_samples, max_attempts=None):
        """Keeps drawing samples until we get the desired number.

        This method wraps ``_sample`` and has identical arguments and returns,
        except for one additional argument that specifies the max number of
        attempts.

        Args:
            n_samples (int): number of samples to draw.
            max_attempts (int): max number of calls to ``_sample``. If
                ``None``, never give up. Defaults to ``None``.

        Returns:
            ndarray: :math:`r_1` values.
            ndarray: :math:`r_2` values.
            ndarray: :math:`r_3` values.
            ndarray: :math:`r_4` values.
            ndarray: :math:`c_1` values.
            ndarray: :math:`c_2` values.
            ndarray: :math:`c_3` values.
            ndarray: :math:`c_4` values.

        """
        samples = self._sample(int(1.2*n_samples))
        if len(samples.T) >= n_samples:
            return samples[:, :n_samples]
        # Keep trying
        if max_attempts is None:
            attempts = count()
        else:
            attempts = range(max_attempts - 1)
        for attempt in attempts:
            samples = np.hstack((samples, self._sample(2*n_samples)))
            if len(samples.T) >= n_samples:
                return samples[:, :n_samples]

    def _sample_v1(self, n_samples):
        """Sample :math:`v_1` from the marginal distribution.

        Args:
            n_samples (int): number of samples to draw.

        Returns:
            ndarray: sampled :math:`v_1` values.

        """
        if self.vdf.v_min == self.vdf.v_max:
            arr = np.zeros(n_samples)
            arr[:] = self.vdf.v_min
            return arr
        else:
            p_v1 = self.rate_fixed_v1_total \
                * self.vdf.v1_marginal(self.v1_vals)
            mask = np.isfinite(p_v1) & (p_v1 > 0)
            if not np.any(mask):
                raise NoDataException("No velocity values")
            return sample_from_pdf(self.v1_vals[mask], p_v1[mask], n_samples)

    @scalarize_method
    def _sample_c1(self, v1s):
        """Sample :math:`c_1` at fixed :math:`v_1`.

        Args:
            v1s (ndarray): :math:`v_1` values.

        Returns:
            ndarray: :math:`c_1` values.

        """
        c1s = np.zeros_like(v1s)
        if isinstance(self.vdf, SingleVelocityDistribution):
            c1s[:] = self.vdf.cos_theta
            return c1s
        else:
            for i, v1 in enumerate(v1s):
                v1_vals = np.zeros_like(self.c1_vals)
                v1_vals[:] = v1
                p_c1 = self.vdf(v1_vals, self.c1_vals)
                c1s[i] = sample_from_pdf(self.c1_vals, p_c1, 1)[0]
                try:
                    assert np.abs(c1s[i]) <= 1
                except AssertionError:
                    raise AssertionError(
                        "got c1=%r for\nv1=%r,\nc1_vals=%r,\np_c1=%r" % (
                            v1, c1s[i], self.c1_vals, p_c1
                        )
                    )
            return c1s

    def q_to_p(self, r1, rq, r3, c1, cq):
        """Transform momentum-transfer samples to get the final-state momenta.

        Note that this method also samples azimuthal angles for the
        quasiparticles with respect to the momentum transfer. The three input
        arrays must have the same length.

        Args:
            r1 (ndarray): :math:`r_1` values.
            rq (ndarray): :math:`r_q` values.
            r3 (ndarray): :math:`r_3` values.
            c1 (ndarray): :math:`c_1` values.
            cq (ndarray): :math:`\\cos\\theta_q` values.

        Returns:
            ndarray: an array of samples of shape ``(len(cq), 4)``. These
                samples have the form :math:`(r_3, r_4, c_3, c_4)`.

        """
        # Compute r4. Note there are two choices, and we use the pdf to choose.
        # Compute the probability for each sign for each point
        assert not np.any(np.abs(cq) > 1)
        p_pairs = self.pdf_fixed_sign(r1, rq, cq, r3, SIGNS[None, :])
        # Normalize each pair
        p_pairs /= np.sum(p_pairs, axis=1)[:, None]
        # Choose a sign for each point
        s = np.array([
            np.random.choice(SIGNS, p=p_pair) for p_pair in p_pairs
        ])
        # Use this sign to compute r4
        em = self.material.E_F_m * self.material.m_star_m
        meo = self.material.m_star_m * self._omega(r1, rq, cq)
        med = self.material.m_star_m * self.material.Delta_m
        r4 = np.sqrt(
            2*em - s*np.sqrt(
                4*em**2 - 4*em*r3**2 + r3**4 + 4*meo*(
                    meo - np.sqrt(
                        4*em**2 - 4*em*r3**2 + r3**4 + 4*med**2
                    )
                ) + 0j
            ) + 0j
        )
        # Compute cq3, the cosine of angle between q and p3, and cq4
        cq3 = (rq**2 + r3**2 - r4**2) / (2*r3*rq)
        cq4 = (rq**2 - r3**2 + r4**2) / (2*r4*rq)
        assert not np.any(np.abs(cq3) > 1)
        assert not np.any(np.abs(cq4) > 1)
        # Generate azimuthal angles of the QPs wrt the momentum transfer
        phi3 = np.random.uniform(-np.pi, np.pi, size=len(cq))
        phi4 = phi3 + np.pi
        # Generate azimuthal angles of the DM around the DM axis
        phix = np.random.uniform(-np.pi, np.pi, size=len(cq))
        # Use cosine-addition to get c3 and c4 wrt the DM
        c3x = cosine_add(cq, cq3, phi3)
        c4x = cosine_add(cq, cq4, phi4)
        assert not np.any(np.abs(c3x) > 1)
        assert not np.any(np.abs(c4x) > 1)
        # Use cosine-addition to get c3 and c4 wrt the DM wind
        c3 = cosine_add(c1, c3x, phix)
        c4 = cosine_add(c1, c4x, phix)
        assert not np.any(np.abs(c3) > 1)
        assert not np.any(np.abs(c4) > 1)

        # Find the final-state DM momentum
        # We need E2 to get r2, and to get E2, we first need E1, E3, and E4.
        E1 = r1**2/(2*self.m1)
        E3 = np.sqrt(
            self.material.Delta_m**2 + (
                r3**2/(2*self.material.m_star_m) - self.material.E_F_m
            )**2
        )
        E4 = np.sqrt(
            self.material.Delta_m**2 + (
                r4**2/(2*self.material.m_star_m) - self.material.E_F_m
            )**2
        )
        E2 = E1 - E3 - E4
        r2 = np.sqrt(2*self.m1*E2)
        # Use this to compute the angle c2 wrt the DM
        # c2x = (r1**2 - r1*r3*c3 - r1*r4*c4)/(r1*r2)
        """I don't understand where the line above came from, and it looks
        wrong. One should find c2 as follows:

            p1 = p2 + q
            p1.p2 = r2^2 + q.p2
            c12 = [r2^2 + q.p2] / (r1 * r2)
                = [r2^2 + q.(p1 - p3 - p4)] / (r1 * r2)
                = [r2^2 + rq * (r1*cq1 - r3*cq3 - r4*cq4)] / (r1 * r2).

        """
        c2x = (r2**2 + rq * (r1*cq - r3*cq3 - r4*cq4)) / (r1 * r2)
        # and convert to the DM wind axis
        c2 = cosine_add(c1, c2x, phix)
        return np.real(np.vstack((r2, r3, r4, c2, c3, c4)))

    def samples_to_ensemble(self, r1, r2, r3, r4, c1, c2, c3, c4):
        """Convert sampled final-state momenta to an :obj:`Ensemble`.

        All input arrays must have the same shape.

        Args:
            r1 (ndarray): sampled :math:`r_1` values.
            r1 (ndarray): sampled :math:`r_2` values.
            r3 (ndarray): sampled :math:`r_3` values.
            r4 (ndarray): sampled :math:`r_4` values.
            c1 (ndarray): sampled :math:`c_1` values.
            c1 (ndarray): sampled :math:`c_2` values.
            c3 (ndarray): sampled :math:`c_3` values.
            c4 (ndarray): sampled :math:`c_4` values.

        Returns:
            :obj:`Ensemble`: an ensemble of :obj:`DarkMatterScatter` events.

        """
        # Build the ensemble
        events = [
            Event(
                [DarkMatter(mass=self.m1, momentum=r1p, cos_theta=c1p,
                            material=self.material)],
                self.material,
                final_state=[
                    DarkMatter(mass=self.m1, momentum=r2p, cos_theta=c2p,
                               material=self.material),
                    Quasiparticle(momentum=r3p, cos_theta=c3p,
                                  material=self.material),
                    Quasiparticle(momentum=r4p, cos_theta=c4p,
                                  material=self.material),
                ]
            )
            for r1p, r2p, r3p, r4p, c1p, c2p, c3p, c4p
            in zip(r1, r2, r3, r4, c1, c2, c3, c4)
        ]
        ensemble = Ensemble(events)
        # "act" once so that the quasiparticles become the out states
        for event in ensemble:
            event.act()
        return ensemble

    def ensemble(self, *args, **kwargs):
        """Sample an ensemble of DM scatters.

        Args:
            n_samples (int): number of samples to draw.

        Returns:
            :obj:`Ensemble`: an ensemble of :obj:`DarkMatterScatter` events.

        """
        return self.samples_to_ensemble(*self.sample(*args, **kwargs))

    def q_rate_grid(self, r1, order=20):
        """Compute a grid of rates at fixed :math:`(c_q,r_q)`.

        Args:
            r1 (float): fixed :math:`r_1` value.
            order (int, optional): fixed order to use for Gaussian quadrature.
            threshold (float, optional): the fraction of the integrated rate
                that should be preserved when doing cuts. Note that cuts are
                repeated ``n_cuts`` times. Defaults to 0.99.

        Returns:
            ndarray: 1d array of unique :math:`c_q` values.
            ndarray: 1d array of unique :math:`r_q` values.
            ndarray: 2d array of rate values.

        Raises:
            ValueError: if all rates are zero.
            ValueError: if :math:`r_1` is too small for any scattering to be
                kinematically allowed.

        """

        """We want to sample on a grid of ``self.n_cq`` by ``self.n_rq``, but
        the rate may have most of its support only in a small range of
        :math:`rq` values. Thus, we'll make a couple of passes to determine the
        best region to sample. To do that cleanly, we define the following
        helper method to do the actual rate calculation, and another helper
        method to trim the grid.
        """
        def _rate_grid_fixed_domain(rq_min, rq_max):
            """Compute points on a grid between two values of :math:`r_q`.

            Args:
                rq_min (float): lower bound on :math:`r_q`.
                rq_max (float): upper bound on :math:`r_q`.

            Returns:
                ndarray: 1d array of unique :math:`c_q` values.
                ndarray: 1d array of unique :math:`r_q` values.
                ndarray: 2d array of rate values.

            """
            cq_vals = np.linspace(0, 1, self.n_cq)
            rq_vals = np.linspace(rq_min, rq_max, self.n_rq)
            rates = np.zeros((len(cq_vals), len(rq_vals)))
            for i, cq in enumerate(cq_vals):
                for j, rq in enumerate(rq_vals):
                    rates[i, j] = self.q_rate(r1, rq, cq, order=order)
            return cq_vals, rq_vals, rates

        def _find_limits(rq_vals, max_attempts=10):
            """Search for a non-zero rate.

            Assuming all rates previously came up zero, keep adding points
            until a non-zero rate is found. Then return the rq values on either
            side, corresponding to zero rates that bracket the nonzero rate(s).
            This interval can then be sampled more densely.

            Args:
                rq_vals (ndarray): the previous array of :math:`r_q` values.
                max_attempts (int): number of shifts to try.

            Returns:
                float: the lower bound of the region of support of the rate.
                float: the upper bound of the region of support of the rate.

            Raises:
                RuntimeError: unable to locate any non-zero values.

            """
            cq_vals = np.linspace(0, 1, self.n_cq)
            delta = rq_vals[1] - rq_vals[0]
            shifted_rq_vals = rq_vals + delta/2
            shifted_rq_vals = np.insert(shifted_rq_vals, 0, rq_vals[0])[:-1]
            rates = np.zeros((len(cq_vals), len(shifted_rq_vals)))
            for i, cq in enumerate(cq_vals):
                for j, rq in enumerate(shifted_rq_vals):
                    rates[i, j] = self.q_rate(r1, rq, cq, order=order)
            # Sum out the cq axis
            rates = np.sum(rates, axis=0)
            if np.count_nonzero(rates > 0) == 0:
                if max_attempts > 1:
                    return _find_limits(
                        shifted_rq_vals,
                        max_attempts=(max_attempts - 1)
                    )
                else:
                    raise RuntimeError
            else:
                # Find the indices of non-zero values
                nonzero = np.where(rates > 0)[0]
                if nonzero[0] > 0:
                    lower = shifted_rq_vals[nonzero[0] - 1]
                else:
                    lower = shifted_rq_vals[nonzero[0]]
                if nonzero[-1] < len(shifted_rq_vals) - 1:
                    upper = shifted_rq_vals[nonzero[-1] + 1]
                else:
                    upper = shifted_rq_vals[nonzero[-1]]
                return lower, upper

        def _shrink_grid(rq_vals, rates):
            """Recompute the grid on the region with 99% of the support.

            More specifically, we presume that small :math:`r_q` values should
            always be included, but we move down the max :math:`r_q` value to
            lie above a threshold fraction of scatters (99% by default). As a
            practical matter, that last 1% can extend far above the rest, so
            this cut can make the sampling much more efficient while barely
            influencing the results.

            Args:
                rq_vals (ndarray): the previous array of :math:`r_q` values.
                rates (ndarray): the grid of rates computed with these values
                    of :math:`r_q`.

            Returns:
                ndarray: 1d array of unique :math:`c_q` values.
                ndarray: 1d array of unique :math:`r_q` values.
                ndarray: 2d array of rate values.

            Raises:
                ValueError: if all rates are zero.
                RuntimeError: the cdf never exceeds the threshold. This is a
                    sanity check that should never trigger unless there is an
                    error in the code.

            """
            # Compute the cdf of rq
            cdf = np.cumsum(np.sum(rates, axis=0))
            if cdf[-1] == 0:
                raise ValueError("No cdf: all zero")
            cdf /= cdf[-1]
            # Check whether the top bin contributes more than 1-threshold
            # If so, don't cut!
            try:
                if cdf[-1] - cdf[-2] > 1 - self.support_threshold:
                    cq_vals = np.linspace(0, 1, self.n_cq)
                    return cq_vals, rq_vals, rates
            except IndexError:
                pass
            # Compute the new upper limit where the cdf crosses the threshold
            try:
                rq_lim = rq_vals[
                    np.where(cdf > self.support_threshold)[0][0]
                ]
            except IndexError:
                raise RuntimeError
            # Recompute the rate grid with this new upper limit
            return _rate_grid_fixed_domain(np.amin(rq_vals), rq_lim)

        # Determine the physical limits of :math:`r_q`
        # The ``1e-10`` here is added for numerical stability, since there is
        # a multiplication by the DM mass in the middle.
        if r1 < 2*np.sqrt(self.m1*self.material.Delta_m) * (1 - 1e-10):
            raise ValueError("Insufficient energy for scattering")
        rq_min = min(
            np.sqrt(2*self.m1*self.material.Delta_m),  # TODO: check this!!!
            r1 - np.sqrt(r1**2 - 2*self.m1*self.material.Delta_m)
        )
        """Recall that omega and q are related by

            omega = q.v1 - q^2/(2*m1)
                  = (r1*rq*cq - rq^2/2) / m1

        With omega as above, we have

            rq = r1*cq  +/-  sqrt(cq^2*r1^2 - 2*m1*omega)

        and thus the following upper bound on rq applies:

                rq < r1 + sqrt(r1^2 - 2*m1*omega)

        Since omega is bounded below at 2Delta, inserting this value gives the
        absolute upper bound on rq.

        """
        rq_max = r1 + np.sqrt(r1**2 - 4*self.m1*self.material.Delta_m)

        # Carry out a first grid computation
        cq_vals, rq_vals, rates = _rate_grid_fixed_domain(rq_min, rq_max)

        if np.count_nonzero(rates) == 0:
            try:
                rq_min, rq_max = _find_limits(rq_vals)
            except RuntimeError:
                # We can't find any non-zero rates
                return cq_vals, rq_vals, rates
            cq_vals, rq_vals, rates = _rate_grid_fixed_domain(rq_min, rq_max)

        # Shrink this grid ``n_cuts`` times
        for _ in range(self.n_cuts):
            try:
                try:
                    cq_vals, rq_vals, rates = _shrink_grid(rq_vals, rates)
                except ValueError:
                    # Everything was zero! Try sampling more finely.
                    print("Failed to shrink")
                    break
            except RuntimeError:
                print("RuntimeError")
                break

        # We might now have an outlier point that drives the sampling. If so,
        # we haven't zoomed in far enough.

        def _outlier_ratio(matrix):
            # Ratio of largest value to second-largest value
            arr = np.copy(matrix)
            arr = arr.reshape(-1)
            arr.sort()
            return arr[-1] / arr[-2]

        n_shrink = 0
        while _outlier_ratio(rates) > 10:
            if n_shrink >= 10:
                raise RuntimeError(
                    "After 10 refinements, the outlier ratio is still %f" % (
                        _outlier_ratio(rates)
                    )
                )
            cq_vals, rq_vals, rates = _rate_grid_fixed_domain(
                np.amin(rq_vals), 0.99*np.amax(rq_vals)
            )
            n_shrink += 1

        return cq_vals, rq_vals, rates

    def _rate_grid_fixed_r1(self, r1):
        """Interpolate the fixed-:math:`q` rate grids at a given :math:`r_1`.

        We know some things about the structure of the data, so we can do this
        quite efficiently by hand. Basically, we first need to locate the
        slices above and below, and then establish a 1-to-1 correspondence
        between the rq values in the two slices. Then we can determine the rq
        values in the middle slice by interpolation.

        Args:
            r1 (float): :math:`r_1` value.

        Returns:
            float: :math:`c_q` values at this :math:`r_1`.
            float: :math:`r_q` values at this :math:`r_1`.
            float: rate grid at this :math:`r_1`.

        Raises:
            ValueError: the supplied :math:`r_1` is outside the computed range.

        """
        if r1 > np.amax(self.r1_vals) or r1 < np.amin(self.r1_vals):
            raise ValueError
        for i, sr1 in enumerate(self.r1_vals):
            if r1 == sr1:
                return (
                    self.cq_vals, self.rq_by_v1[i], self.rate_fixed_v1_cq_rq[i]
                )
        if self.r1_vals.size == 1:
            # We did not match any of the stored momenta
            raise ValueError("Invalid r1 for single-velocity distribution")
        # We can now assume that we have multiple slices and that the provided
        # r1 value lies between two of them.
        i_above = np.where(self.r1_vals > r1)[0][0]
        i_below = i_above - 1
        if i_below < 0:
            raise RuntimeError("Something has gone terribly wrong")
        # Get the two slices
        r1_above, r1_below = self.r1_vals[i_above], self.r1_vals[i_below]
        grid_above = self.rate_fixed_v1_cq_rq[i_above]
        grid_below = self.rate_fixed_v1_cq_rq[i_below]
        rq_above = self.rq_by_v1[i_above]
        rq_below = self.rq_by_v1[i_below]
        # Find 0 < t < 1 characterizing the linear interpolation
        t = (r1 - r1_below) / (r1_above - r1_below)
        # Interpolate the rq values
        rq_vals = rq_below + (rq_above - rq_below)*t
        # Interpolate the rate values
        grid_vals = grid_below + (grid_above - grid_below)*t
        return self.cq_vals, rq_vals, grid_vals

    def _sample_q_fixed_r1(self, r1, **kwargs):
        """Sample :math:`(c_q,r_q,r_3)`: at fixed :math:`r_1`.

        Args:
            r1 (float): fixed :math:`r_1` value.

        Returns:
            float: sampled :math:`c_q` value.
            float: sampled :math:`r_q` value.
            float: sampled :math:`r_3` value.

        """
        # Start by interpolating the fixed-q rate grids to this r1 value
        _, rq_vals, rates = self._rate_grid_fixed_r1(r1)
        # Normalize the rates for sampling
        total_rate = np.sum(rates)
        if total_rate == 0:
            raise ValueError("All integrated points zero")
        rates /= total_rate
        # Interpolate
        spline = interp2d(self.cq_vals, rq_vals, rates.T)
        x_int = np.linspace(0, 1, self.n_spline)
        y_int = np.linspace(np.amin(rq_vals), np.amax(rq_vals), self.n_spline)
        Z_int = spline(x_int, y_int)
        Z_int[Z_int < 0] = 0
        # Sample (cq, rq)
        X_int, Y_int = np.meshgrid(x_int, y_int)
        if self.omega_max is not None:
            # Apply a deposit cut
            omega = self._omega(r1, Y_int, X_int)
            Z_int[omega > self.omega_max] = 0
        Z_total = np.sum(Z_int)
        if Z_total == 0:
            raise ValueError("All interpolated points zero")
        Z_int /= Z_total
        X_line, Y_line = X_int.reshape(-1), Y_int.reshape(-1)
        index = np.random.choice(
            np.arange(X_int.size, dtype=int),
            p=Z_int.reshape(-1)
        )
        cq, rq = X_line[index], Y_line[index]
        if np.any(np.abs(cq) > 1):
            raise NoDataException("We got a non-physical cq value")
        # Sample r3
        lo, hi = self.r3_domain(r1, rq, cq)
        if lo == hi:
            raise NoDataException("This point appears to have no r3 values")
        r3s = np.linspace(lo, hi, self.n_spline)[1:-1]
        # We chop off the exact boundaries because they aren't physical
        ps = self.pdf(r1, rq, cq, r3s)
        mask = np.isfinite(ps)
        ps = ps[mask]
        r3s = r3s[mask]
        if len(ps) == 0:
            raise NoDataException("No probabilities were finite")
        tot = np.sum(ps)
        if tot == 0:
            raise NoDataException("Unable to find any non-zero points")
        ps /= tot
        r3 = np.random.choice(r3s, p=ps)
        return cq, rq, r3
