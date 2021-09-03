"""This module defines tools that are used throughout the rest of the code."""

import logging

import numpy as np
from scipy.interpolate import interp1d


class SimulationException(BaseException):
    """Base class for custom exceptions."""


class NonPhysicalException(SimulationException):
    """Raised if something non-physical happened."""


class NumericalException(SimulationException):
    """Raised if an exceptional numerical condition occurred."""


class NotAllowedException(SimulationException):
    """Raised if an interaction cannot proceed."""


class NoDataException(SimulationException):
    """Raised if no data is available for an operation."""


def scalarize(func, index=0):
    """Decorator used to allow vector functions to handle scalars.

    Functions designed to handle arrays often have to assume that the input is
    an array, and then return the output as an array. It is useful to have
    surrounding code to check whether the input is a scalar, and if so, to
    preprocess the scalar into an array and postprocess the output array back
    to a scalar.

    Note that this decorator coerces the target argument to a numpy array.

    In some sense this is the opposite of "vectorizing" a function.

    Args:
        func (function): the function to be decorated.
        index (int, optional): the index of the argument to be scalarized.
            Defaults to 0.

    Returns:
        function: the decorated function.

    """
    def wrapper(*args, **kwargs):
        """Scalarize in the first argument."""
        prev_args, s_arg = args[:index], args[index]
        if len(args) > index + 1:
            next_args = args[index+1:]
        else:
            next_args = []
        # Test whether the target arg is a scalar
        try:
            s_arg[0]
        except (TypeError, IndexError):
            scalar = True
            s_arg = np.asarray([s_arg])
        else:
            scalar = False
            s_arg = np.asarray(s_arg)
        all_args = []
        all_args.extend(prev_args)
        all_args.append(s_arg)
        all_args.extend(next_args)
        output = func(*all_args, **kwargs)
        if scalar:
            output = output[0]
        return output
    return wrapper


def scalarize_method(method, index=0):
    """Equivalent to `scalarize` but for methods.

    Args:
        method (function): the method to be decorated.
        index (int, optional): the index of the argument to be scalarized.
            Defaults to 0.

    Returns:
        function: the decorated method.

    """
    return scalarize(method, index=index+1)


def cosine_add(cos_theta_0, cos_theta_1, phi):
    """Find the polar angle of a vector given its relative angle to another.

    Args:
        cos_theta_0 (float): cosine of the polar angle of the base vector.
        cos_theta_1 (float): cosine of the angle between the base vector and
            the second vector.
        phi (float): azimuthal angle of the second vector around the base
            vector, defined so that phi=0 points towards the global z-axis.

    Returns:
        float: cosine of the polar angle of the second vector in the same
            coordinate system in which `cos_theta_0` is defined.

    """
    sin_theta_0 = np.sqrt(1 - cos_theta_0**2)
    sin_theta_1 = np.sqrt(1 - cos_theta_1**2)
    return cos_theta_0 * cos_theta_1 - \
        np.cos(phi) * sin_theta_0 * sin_theta_1


@scalarize
def tolerate(cosine, tolerance):
    """Chop off small excesses in cosines / sines.

    Sometimes, due to numerical tolerances in root-finding, we will end up with
    a cosine slightly above 1 or below -1. This function sets such values
    equal to exactly 1 or -1. Values that exceed the tolerance are left alone.

    Args:
        cosine (float): the value to trim.
        tolerance (float): allowed distance from +/-1.

    Returns:
        float: trimmed value.

    """
    trimmed = np.zeros_like(cosine)
    mask = (1 < np.abs(cosine)) & (np.abs(cosine) < 1 + tolerance)
    trimmed[mask] = np.sign(cosine[mask])
    trimmed[~mask] = cosine[~mask]
    return trimmed


def shorthash(obj):
    """Give the last five digits of the hash as a string identifier.

    Args:
        obj (object): hashable object.

    Returns:
        str: last five digits of the hash.

    """
    return str(hash(obj))[-5:]


def obj_logger(obj):
    """Create a standardized logger for an object.

    Args:
        obj (object): hashable object.

    Returns:
        :obj:`Logger`: a logger customized for this object.

    """
    logger = logging.getLogger(
        '%s:%s' % (
            shorthash(obj),
            obj.__class__.__name__
        )
    )
    ch = logging.StreamHandler()
    # To activate debug logging, uncomment:
    # ch.setLevel(logging.DEBUG)
    # logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(levelname)s - %(name)s - %(message)s'
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def sample_from_pdf(x, p, n_samples):
    """Draw samples from a 1d pdf represented by discrete samples.

    Args:
        x (ndarray): array of values where samples are drawn.
        p (ndarray): array of pdf values (not necessarily normalized).
        n_samples (int): number of samples to draw.

    Returns:
        ndarray: samples.

    """
    # Something can go wrong if the pdf vanishes somewhere, so that the cdf
    # is not invertible. Cure this by eliminating such points. We also remove
    # pathological negative probabilities.
    p_zero = p <= 0
    p = p[~p_zero]
    x = x[~p_zero]
    if len(p) == 0:
        raise ValueError("No non-zero values in the pdf")
    if len(p) == 1:
        # Only one non-zero value in the pdf, so we'll return only that
        return np.ones(n_samples) * x[0]
    cdf_samples = np.asarray(np.cumsum(p), dtype=float)
    cdf_samples /= cdf_samples[-1]
    icdf = interp1d(cdf_samples, x, fill_value='extrapolate')
    samples = icdf(np.random.uniform(size=n_samples))
    # It is possible that a bad spline would return values outside the range
    samples[samples < np.amin(x)] = np.amin(x)
    samples[samples > np.amax(x)] = np.amax(x)
    return samples


def intersection(interval_a, interval_b):
    """Find the intersection of two intervals.

    Args:
        interval_a (:obj:`tuple` of float): the first interval.
        interval_b (:obj:`tuple` of float): the second interval.

    Returns:
        :obj:`tuple` of float: the resulting intersection.

    Raises:
        ValueError: the intervals are disjoint.

    """
    # Determine whether the two are disjoint
    if np.amax(interval_a) < np.amin(interval_b) or \
            np.amax(interval_b) < np.amin(interval_a):
        raise ValueError("Disjoint intervals")
    # Max of the mins, min of the maxes
    return np.asarray([
        max(np.amin(interval_a), np.amin(interval_b)),
        min(np.amax(interval_a), np.amax(interval_b))
    ])


def union(interval_a, interval_b):
    """Find the union of two overlapping intervals.

    Args:
        interval_a (:obj:`tuple` of float): the first interval.
        interval_b (:obj:`tuple` of float): the second interval.

    Returns:
        :obj:`tuple` of float: the resulting union.

    Raises:
        ValueError: the intervals are disjoint.

    """
    # Determine whether the two are disjoint
    if np.amax(interval_a) < np.amin(interval_b) or \
            np.amax(interval_b) < np.amin(interval_a):
        raise ValueError("Disjoint intervals")
    return np.asarray([
        min(np.amin(interval_a), np.amin(interval_b)),
        max(np.amax(interval_a), np.amax(interval_b))
    ])


class Interval(object):
    """Represents an interval on the real line.

    This is part of an overengineered but highly generic solution to the
    problem of dealing with allowed regions of phase space. Multiple `Interval`
    can be combined together in one `Region` object.

    Args:
        a (float): endpoint a.
        b (float): endpoint b.

    Attrs:
        a (float): endpoint a.
        b (float): endpoint b.

    """
    def __init__(self, a, b):
        self.a, self.b = a, b

    def min(self):
        """Lower bound of the interval.

        Returns:
            float: lower bound.

        """
        return min(self.a, self.b)

    def max(self):
        """Upper bound of the interval.

        Returns:
            float: upper bound.

        """
        return max(self.a, self.b)

    def measure(self):
        """Measure of the interval.

        Returns:
            float: difference of upper and lower bounds.

        """
        return self.max() - self.min()

    def disjoint(self, other):
        """Test whether this interval is disjoint from another.

        Args:
            other (:obj:`Interval`): the other interval.

        Returns:
            bool: `True` if this interval and `other` are disjoint.

        """
        return self.max() < other.min() or self.min() > other.max()

    def union(self, other):
        """Union of this interval with another interval.

        Args:
            other (:obj:`Interval`): the other interval.

        Returns:
            :obj:`Interval`: an `Interval` representing the union of this
                `Interval` with `other`.

        """
        if self.disjoint(other):
            raise ValueError("Disjoint intervals")
        # Min of the mins, max of the maxes
        return Interval(
            min(self.min(), other.min()),
            max(self.max(), other.max())
        )

    def intersection(self, other):
        """Intersection of this interval with another interval.

        Args:
            other (:obj:`Interval`): the other interval.

        Returns:
            :obj:`Interval`: an `Interval` representing the Intersection of
                this `Interval` with `other`.

        """
        if self.disjoint(other):
            raise ValueError("Disjoint intervals")
        # Max of the mins, min of the maxes
        return Interval(
            max(self.min(), other.min()),
            min(self.max(), other.max())
        )

    def __repr__(self):
        return "<%s[%.3e, %.3e]>" % (
            self.__class__.__name__, self.min(), self.max()
        )


class Region(object):
    """Represents a union of real intervals.

    An overengineered but highly generic solution to the problem of dealing
    with allowed regions of phase space. A `Region` object contains multiple
    `Interval` objects, but curates them to keep all internal intervals
    disjoint. This means that adding an `Interval` to a `Region` need not mean
    that this exact

    Args:
        intervals (:obj:`list` of :obj:`tuple`, optional): starting intervals.

    """
    def __init__(self, *args, **kwargs):
        self.intervals = []
        if len(args) > 0:
            for interval in args[0]:
                self.add(interval)

    def _reduce(self):
        """Recusively replace non-disjoint interval pairs with their union.

        This method must be run after making any changes to the internal list
        of intervals. It ensures that the region is represented as a disjoint
        union at all times.

        """
        overlap = None  # Record whether we have found a non-disjoint pair
        # Iterate through all pairs
        for i, ia in enumerate(self.intervals):
            for j, ib in enumerate(self.intervals):
                if i == j:
                    continue
                if not ia.disjoint(ib):
                    # We have found a non-disjoint pair. Exit the loop.
                    overlap = (i, j)
                    break
            if overlap is not None:
                break
        if overlap is None:
            # All intervals are disjoint
            return
        # We found a non-disjoint pair. Merge the two intervals.
        i, j = overlap
        self.intervals[i] = self.intervals[i].union(self.intervals[j])
        self.intervals.pop(j)
        # There could be other non-disjoint pairs, so reduce again.
        self._reduce()

    def _sort(self):
        """Put intervals in order.

        Note that this method assumes that the intervals are disjoint, which is
        always the case if `_reduce` has been run.

        """
        mins = [interval.min() for interval in self.intervals]
        order = np.argsort(mins)
        self.intervals = [self.intervals[i] for i in order]

    def add(self, interval):
        """Add an interval by union.

        Since this is the only interface for modifying the region, it is
        important that this method curates the intervals by running `_reduce`
        and `_sort` after each modification.

        """
        self.intervals.append(interval)
        self._reduce()
        self._sort()

    def n(self):
        """Number of intervals.

        Returns:
            int: number of intervals.

        """
        return len(self.intervals)

    def measure(self):
        """Measure of the region.

        Returns:
            float: sum of the measures of the disjoint intervals.

        """
        return np.sum([i.measure() for i in self.intervals])

    def min(self):
        """Lower bound of the region.

        This method assumes that the intervals are stored in sorted order.

        Returns:
            float: lower bound.

        """
        return self.intervals[0].min()

    def max(self):
        """Upper bound of the region.

        This method assumes that the intervals are stored in sorted order.

        Returns:
            float: upper bound.

        """
        return self.intervals[-1].max()

    def linspace(self, n):
        """Generate `n` points equally separated within each subinterval.

        The logic is to imagine that all of the intervals were jammed together
        into one big interval, use `np.linspace` on that interval, and then put
        back the spaces between the intervals. Thus, the points generated by
        this method would be equally spaced if there were no space between the
        intervals.

        Returns:
            :obj:`ndarray` of float: points.

        """
        # Treat it as one long interval and then put space between points
        offsets = np.linspace(0, self.measure(), n).tolist()
        shift = self.min()
        points = []
        for i, interval in enumerate(self.intervals):
            interval_points = []
            # Take the next offset
            while len(offsets):
                offset = offsets.pop(0)
                point = offset + shift
                if point > interval.max():
                    # Put this back on the offset list and stop
                    offsets.insert(0, offset)
                    break
                else:
                    interval_points.append(point)
            if len(interval_points):
                points.append(np.asarray(interval_points))
            # If there is another interval, add to the shift
            if i + 1 < len(self.intervals):
                shift += self.intervals[i + 1].min() - interval.max()
        return [np.asarray(p) for p in points]
