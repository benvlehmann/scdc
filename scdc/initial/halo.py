"""This module defines forms of the DM halo velocity distribution."""

import numpy as np
from scipy.special import erf
from scipy.interpolate import (
    InterpolatedUnivariateSpline, RectBivariateSpline
)
import tensorflow as tf

from ..common import sample_from_pdf, scalarize_method


class VelocityDistribution(object):
    """Represents a generic DM velocity distribution.

    This is a base class for velocity distributions. Such a distribution is
    defined by a pdf in the two variables ``v1`` and ``c1``, corresponding to
    speed and :math:`\cos\theta` with respect to the wind axis. The pdf is to
    be implemented in ``__call__``.

    During initialization, the pdf is sampled at a large number of points, and
    the marginal distributions of ``v1`` and ``c1`` are constructed from
    interpolation of those samples.

    Args:
        n_c1 (int, optional): number of sample points to use in ``c1`` when
            building the marginal distribution. Defaults to 50.
        n_v1 (int, optional): number of sample points to use in ``v1`` when
            building the marginal distribution. Defaults to 50.

    Attributes:
        n_c1 (int): number of sample points used in ``c1`` when building the
            marginal distribution.
        n_v1 (int): number of sample points used in ``v1`` when building the
            marginal distribution.
        v_min (float): minimum velocity.
        v_max (float): maximum velocity.
        c1_min (float): minimum :math:`\cos\theta`.
        c1_max (float): maximum :math:`\cos\theta`.
        c1_marginal_spline (function): spline interpolation of the marginal
            distribution of ``c1``.
        v1_marginal_spline (function): spline interpolation of the marginal
            distribution of ``v1``.


    """
    def __init__(self, *args, **kwargs):
        self.v_min = 0.
        self.v_max = np.inf
        self.c1_min = -1.
        self.c1_max = 1.
        self.c1_marginal_spline = None
        self.v1_marginal_spline = None
        self.n_c1 = kwargs.get('n_c1', 50)
        self.n_v1 = kwargs.get('n_v1', 50)

    def __call__(self, v1, c1):
        """Probability density function.

        Args:
            v1 (float): DM speed.
            c1 (float): cosine of angle with respect to DM wind.

        Returns:
            float: probability density.

        """
        raise NotImplementedError

    def _make_marginals(self):
        """Construct marginal interpolation splines."""
        c1_values = np.linspace(self.c1_min, self.c1_max, self.n_c1)
        v1_values = np.linspace(self.v_min, self.v_max, self.n_v1)
        c1_grid, v1_grid = np.meshgrid(c1_values, v1_values)
        p_values = self(v1_grid, c1_grid)
        # Sum probabilities to get interpolated marginal
        v1_totals = np.sum(p_values, axis=1)
        c1_totals = np.sum(p_values, axis=0)
        # Normalize
        v1_totals /= np.sum(v1_totals)
        c1_totals /= np.sum(c1_totals)
        # Interpolate
        self.v1_marginal_spline = \
            InterpolatedUnivariateSpline(v1_values, v1_totals)
        self.c1_marginal_spline = \
            InterpolatedUnivariateSpline(c1_values, c1_totals)

    @scalarize_method
    def v1_marginal(self, v1):
        """Marginal distribution of ``v1``.

        Args:
            v1 (float): ``v1`` value.

        Returns:
            float: probability density.

        """
        if self.v1_marginal_spline is None:
            self._make_marginals()
        result = self.v1_marginal_spline(v1)
        result[(v1 < self.v_min) | (v1 > self.v_max)] = 0.
        return result

    @scalarize_method
    def c1_marginal(self, c1):
        """Marginal distribution of ``c1``.

        Args:
            c1 (float): ``c1`` value.

        Returns:
            float: probability density.

        """
        if self.c1_marginal_spline is None:
            self._make_marginals()
        result = self.c1_marginal_spline(c1)
        result[(c1 < self.c1_min) | (c1 > self.c1_max)] = 0.
        return result


class SingleVelocityDistribution(VelocityDistribution):
    """Represents a velocity distribution with one speed in one direction.

    Args:
        speed (float): the fixed speed.
        cos_theta (float, optional): the direction, given as cos(theta) in the
            laboratory coordinate system. Defaults to 1.

    Attributes:
        speed (float): the fixed speed.
        cos_theta (float): the direction, given as cos(theta) in the
            laboratory coordinate system.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.speed = kwargs.get('speed')
        if self.speed is None:
            raise ValueError("Must supply `speed`")
        self.cos_theta = kwargs.get('cos_theta', 1.)
        self.v_min = self.speed
        self.v_max = self.speed
        self.c1_min = 1.
        self.c1_max = 1.

    def __call__(self, v1, c1):
        """Proportional to the pdf, here non-zero at only one point.

        Args:
            v1 (float): DM speed.
            c1 (float): DM direction, given as cos(theta) in the laboratory
                coordinate system.

        Returns:
            float: 1 if the speed and direction match the preset values.
                Otherwise 0.
        """
        return tf.where(
            tf.logical_and(v1 == self.speed, c1 == self.cos_theta),
            1, 0
        )

    def c1_marginal(self, c1):
        """Marginal distribution of ``c1``.

        For this artificial distribution, it is possible to write the marginal
        distribution explicitly.

        Args:
            c1 (float): ``c1`` value.

        Returns:
            float: probability density.

        """
        return tf.where(
            c1 == self.cos_theta,
            1, 0
        )

    def v1_marginal(self, v1):
        """Marginal distribution of ``v1``.

        For this artificial distribution, it is possible to write the marginal
        distribution explicitly.

        Args:
            v1 (float): ``v1`` value.

        Returns:
            float: probability density.

        """
        return tf.where(
            v1 == self.speed,
            1, 0
        )


class StandardHaloDistributionGalFrame(VelocityDistribution):
    """Represents the Standard Halo Model distribution in the halo frame.

    Args:
        v_esc (float): the escape velocity of the halo.
        v_0 (float): the rms velocity of the halo.

    Attributes:
        v_esc (float): the escape velocity of the halo.
        v_0 (float): the rms velocity of the halo. (NOTE not actually rms.
            This is exp(-v^2/v_0^2). TODO.)

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required_keys = ('v_esc', 'v_0')
        for key in required_keys:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Must supply `%s`" % key)
            setattr(self, key, value)
        # Precompute the normalization factor
        self.norm = 4*np.exp(self.v_esc**2 / self.v_0**2) / self.v_0**2 / (
            -2*self.v_esc + erf(self.v_esc / self.v_0) * (
                np.sqrt(np.pi)*self.v_0*np.exp(self.v_esc**2 / self.v_0**2)
            )
        )
        self.v_min = 0.
        self.v_max = self.v_esc
        self._make_marginals()

    def __call__(self, v1, c1):
        """Probability distribution for halo-frame velocities.

        Args:
            v1 (float): DM speed in the halo frame.
            c1 (float): DM direction, given as cos(theta) measured from the
                laboratory axis in the halo frame.

        Returns:
            float: pdf value for the specified DM velocity in the lab frame.

        """
        v1 = tf.cast(v1, tf.float32)
        return tf.where(v1 < self.v_esc, 1., 0.) * (
            self.norm * v1**2 * tf.exp(
                tf.cast(-v1**2 / self.v_0**2, dtype=tf.float32)
            )
        )


class StandardHaloDistribution(VelocityDistribution):
    """Represents the Standard Halo Model distribution in the lab frame.

    Args:
        v_esc (float): the escape velocity of the halo.
        v_0 (float): the rms velocity of the halo.
        v_wind (float): lab (Earth) velocity in the halo frame.

    Attributes:
        v_esc (float): the escape velocity of the halo.
        v_0 (float): the rms velocity of the halo. (NOTE not actually rms.
            This is exp(-v^2/v_0^2). TODO.)
        v_wind (float): lab (Earth) velocity in the halo frame.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        required_keys = ('v_esc', 'v_0', 'v_wind')
        for key in required_keys:
            value = kwargs.get(key)
            if value is None:
                raise ValueError("Must supply `%s`" % key)
            setattr(self, key, value)
        self.n_gal_samples = kwargs.get('n_gal_samples', int(1e6))
        # Instantiate an internal galactic-frame vdf
        self.gal_vdf = StandardHaloDistributionGalFrame(**kwargs)
        # Build the distribution by empirically sampling in the halo frame
        # and boosting to the lab frame. Assume halo frame vdf is isotropic.
        gal_v_vals = np.linspace(self.gal_vdf.v_min, self.gal_vdf.v_max, 1000)
        gal_v1_samples = sample_from_pdf(
            gal_v_vals,
            self.gal_vdf.v1_marginal(gal_v_vals),
            self.n_gal_samples
        )
        gal_c1_samples = np.random.uniform(-1, 1, self.n_gal_samples)
        gal_s1_samples = np.sqrt(1 - gal_c1_samples**2)
        gal_phi1_samples = np.random.uniform(0, 2*np.pi, self.n_gal_samples)
        # Convert to Cartesian
        gal_xyz = gal_v1_samples[:, None] * np.vstack((
            np.cos(gal_phi1_samples) * gal_s1_samples,
            np.sin(gal_phi1_samples) * gal_s1_samples,
            gal_c1_samples
        )).T
        # Add a DM wind in the z axis
        lab_xyz = gal_xyz + np.array([0, 0, self.v_wind])[None, :]
        # Go back to spherical coordinates
        lab_v1 = np.linalg.norm(lab_xyz, axis=1)
        lab_c1 = lab_xyz[:, 2] / lab_v1
        # Bin these velocities in 2d make a normalized density
        # Choose a number of bins so that the average bin has about 100 points
        n_bins = int(np.sqrt(self.n_gal_samples / 100))
        v_min = np.amin(lab_v1)
        v_max = np.amax(lab_v1)
        c_min = np.amin(lab_c1)
        c_max = np.amax(lab_c1)
        density, v1_edges, c1_edges = np.histogram2d(
            lab_v1, lab_c1, bins=[
                np.linspace(v_min, v_max, n_bins),
                np.linspace(c_min, c_max, n_bins)
            ],
            density=True
        )
        # Use only right edges. We're using a high sample density anyway.
        v1_right = v1_edges[1:]
        c1_right = c1_edges[1:]
        self.v_min = np.amin(v1_right)
        self.v_max = np.amax(v1_right)
        self.c1_min = np.amin(c1_right)
        self.c1_max = np.amax(c1_right)
        self.pdf = np.vectorize(
            RectBivariateSpline(v1_right, c1_right, density)
        )
        self._make_marginals()

    def __call__(self, v1, c1):
        """Probability distribution for halo-frame velocities.

        Args:
            v1 (float): DM speed in the halo frame.
            c1 (float): DM direction, given as cos(theta) measured from the
                laboratory axis in the halo frame.

        Returns:
            float: pdf value for the specified DM velocity in the lab frame.

        """
        return self.pdf(v1, c1)


class IsotropicDistribution(VelocityDistribution):
    """Speed distribution of the Standard Halo Model with isotropic DM.

    This is not a realistic distribution. It is meant to represent a null
    hypothesis in which the speed distribution is exactly the same as the
    Standard Halo Model, but the direction is isotropic. This is needed for
    comparative purposes to statistically assess directional reach.

    Args:
        v_esc (float): the escape velocity of the halo.
        v_0 (float): the rms velocity of the halo.
        v_wind (float): lab (Earth) velocity in the halo frame.

    Attributes:
        shm (:obj:`StandardHaloDistribution`): the corresponding Standard Halo
            Distribution object.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.shm = StandardHaloDistribution(*args, **kwargs)
        self.v_min = self.shm.v_min
        self.v_max = self.shm.v_max
        self.v1_marginal_spline = self.shm.v1_marginal_spline

    @scalarize_method
    def c1_marginal(self, c1):
        result = np.zeros_like(c1, dtype=float)
        result[:] = 0.5
        return result

    @scalarize_method
    def v1_marginal(self, v1):
        return self.shm.v1_marginal(v1)

    def __call__(self, v1, c1):
        """Probability distribution for isotropic velocities.

        Args:
            v1 (float): DM speed in the halo frame.
            c1 (float): DM direction, given as cos(theta) measured from the
                laboratory axis in the halo frame.

        Returns:
            float: pdf value for the specified DM velocity in the lab frame.

        """
        return self.v1_marginal(v1) * self.c1_marginal(c1)
