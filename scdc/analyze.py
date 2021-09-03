"""This module defines functions for analysis of simulated ensembles.

Many of these are 'old', in that they were written at one time for a type of
analysis that has not been used since. Many are also undocumented. However,
they are preserved here for future applications.

"""

import numpy as np

from .particle import ParticleCollection


def fbo_asymmetry(cosines):
    fb_mask = np.abs(cosines) > 0.5
    return np.abs(2*np.count_nonzero(fb_mask)/fb_mask.size - 1)


def forward_mask(particles):
    angle = particles.cos_theta
    mask = np.isfinite(angle)
    mask[mask] = angle[mask] >= 0
    return mask


def forward_backward_count(particles):
    forward = forward_mask(particles)
    n_backward = np.count_nonzero(~forward)
    if n_backward == 0:
        return np.inf
    return np.count_nonzero(forward) / n_backward


def forward_backward_energy(particles):
    forward = forward_mask(particles)
    e_backward = np.sum(particles.energy[~forward])
    if e_backward == 0:
        return np.inf
    return np.sum(particles.energy[forward]) / e_backward


def orthogonal_mask(particles):
    angle = particles.cos_theta
    mask = np.isfinite(angle)
    mask[mask] = np.abs(angle[mask]) <= 0.5
    return mask


def orthogonal_excess_count(particles):
    orthogonal = orthogonal_mask(particles)
    return np.count_nonzero(orthogonal) / (0.5 * len(particles))


def orthogonal_excess_energy(particles):
    orthogonal = orthogonal_mask(particles)
    return np.sum(particles.energy[orthogonal]) \
        / (0.5 * np.sum(particles.energy))


def max_bin_asymmetry(particles, asym, bin_size):
    # slide a window of width `bin_size` in energy and find the max asymmetry
    if bin_size > len(particles):
        bin_size = len(particles) - 1
    # Sort the leaves by energy
    sort_index = np.argsort([p.energy for p in particles])
    sorted_leaves = [particles[i] for i in sort_index]
    nbins = len(sorted_leaves) - bin_size
    asym_values = np.zeros(nbins)
    for index in range(nbins):
        asym_values[index] = \
            asym(ParticleCollection(sorted_leaves[index:index+bin_size]))
    if any(asym_values == 0):
        result = np.inf
    elif any(asym_values == np.inf):
        result = np.inf
    else:
        result = asym_values[np.argmax(np.abs(np.log(asym_values)))]
        if result < asym(particles):
            raise RuntimeError("Something went wrong")
    return result


def plane_asymmetry(angles, n_bins=100, width=1):
    """Find the asymmetry in a sliding bin of fixed width in cos(theta).

    For cos(theta) = 1, this corresponds to forward-backward asymmetry.

    Args:
        angles (:obj:`ndarray`): cos(theta) values. A `ParticleCollection`
            object can be provided instead.
        width (float, optional): width of the sliding bin. Defaults to 1.

    """
    if isinstance(angles, ParticleCollection):
        angles = angles.cos_theta
    window_left = np.linspace(-1, 1, n_bins)
    cos_centers = window_left + width/2.
    cos_centers[cos_centers > 1] = -1 + cos_centers[cos_centers > 1] - 1
    order = np.argsort(cos_centers)
    cos_centers = cos_centers[order]
    counts = np.zeros_like(cos_centers)
    for i, left in enumerate(window_left):
        right = left + width
        if right > 1:
            right = -1 + (right - 1)
            counts[i] = np.count_nonzero((angles < right) | (angles > left))
        else:
            counts[i] = np.count_nonzero((left < angles) & (angles < right))
    counts = counts[order]
    asymmetries = counts / (width/2. * len(angles)) - 1
    return cos_centers, asymmetries


def p_dist(p):
    """Factory for L^p-norm integrands.

    Args:
        p (float): p for the L^p norm.

    Returns:
        function: L^p norm integrand as a function of one argument.

    """
    def _dist(x):
        """Actual norm integrand.

        Args:
            x: function value at a single point.

        Returns:
            float: the value of the integrand at this point.

        """
        return np.abs(x)**p
    return _dist


def norm_asymmetry(angles, distance_function=p_dist(1), n_bins=50):
    """Find the asymmetry as the distance from an isotropic distribution.

    For any norm, this will give zero for perfectly isotropic scattering. The
    maximum depends on the norm. For the default L^1 norm, if the scattering is
    purely directional, corresponding to a delta function, the difference will
    be 0.5 over the whole interval and then the delta function integrates to 1,
    so that'll be a maximum of 2. For the L^2 norm, the norm of the delta is
    not well defined.

    ** However, because of the default L^1 behavior, the result is divided by
    2 regardless of the distance function. **

    Because we're working with a histogram anyway, we don't want to use any
    actual quadrature. Instead, we want to use a p-norm of some kind. So the
    distance function here is not really a norm, but a single-point integrand
    thereof. For example, to use an L^2 norm, the distance function should be

        lambda x: np.abs(x)**2

    and the rest will be taken care of internally.

    Args:
        angles (:obj:`ndarray`): cos(theta) values. A `ParticleCollection`
            object can be provided instead.
        distance_function (function, optional): a function of one variable
            giving the integrand of the norm. Defaults to `p_dist(1)`.
        n_bins (int, optional): number of bins to use for the norm. Defaults
            to 50.

    Returns:
        float: norm/2 of the distance from an isotropic distribution.

    """
    if isinstance(angles, ParticleCollection):
        angles = angles.cos_theta
    # Find the distribution
    bin_edges = np.linspace(-1, 1, n_bins)
    densities, _ = np.histogram(angles, bins=bin_edges, density=True)
    # Integrate the distance function
    widths = np.diff(bin_edges)
    distances = np.asarray([distance_function(x - 0.5) for x in densities])
    return np.sum(distances * widths)/2.


def qp_angle_pairs(event):
    """Final-state QP angles in canonical pairs.

    Here 'canonical' pairing means the following. The number of quasiparticles
    produced in any event must be even, so we sort them by energy and then
    divide into a low-energy half and a high-energy half. The lowest low-energy
    QP is paired with the lowest high-energy QP, the second-lowest with the
    second-lowest, and so on. There is nothing important about this order
    except that it is well-defined.

    Args:
        event (Event): the event for which to find final-state QP pairs.

    Returns:
        ndarray: a 2d array in which each row is a pair.

    """
    qps = event.leaf_particles.quasiparticles
    qps = [qps[i] for i in np.argsort(qps.energy)]
    qps_lo = qps[:len(qps)//2]
    qps_hi = qps[len(qps)//2:]
    angles_lo = [qp.cos_theta for qp in qps_lo]
    angles_hi = [qp.cos_theta for qp in qps_hi]
    return np.asarray(list(zip(angles_lo, angles_hi)))


def get_dataset(hfile, m_dm, m_med, cf_sign=1):
    dataset = None
    for key in hfile:
        ds = hfile[key]
        try:
            if ds.attrs['failed']:
                continue
        except KeyError:
            pass
        try:
            if ds.attrs['cf_sign'] != cf_sign:
                continue
            if ds.attrs['m_dm'] != m_dm:
                # If it matches within 1%, accept it
                if np.abs(m_dm - ds.attrs['m_dm']) / m_dm > 0.01:
                    continue
            # Treat 'heavy' as anything non-zero
            if m_med == 'heavy':
                if ds.attrs['m_med'] == 0:
                    continue
            elif ds.attrs['m_med'] != m_med:
                continue
        except KeyError:
            continue
        dataset = ds
        break
    if dataset is None:
        raise ValueError("Could not find the specified dataset")
    return dataset


def total_deposit(ds, material):
    # Compute the total deposit for each
    dm_omega = {}
    for row in ds[ds['n1'] == b'QP']:
        try:
            dm_omega[row['i0']]
        except KeyError:
            dm_omega[row['i0']] = {}
        try:
            dm_omega[row['i0']][row['i1']]
        except KeyError:
            # This is a new QP
            pass
        else:
            # We have already counted this QP
            continue
        dm_omega[row['i0']][row['i1']] = material.qpe(row['p1'])
    for key, qped in dm_omega.items():
        dm_omega[key] = np.sum(list(dm_omega[key].values()))
    return dm_omega


def bin_by_deposit(rows, deposits, n_bins=None, log=True):
    omega_max = 1.01*np.amax(deposits)
    if n_bins is None:
        # Choose n_bins so that there are >=3 bins between 2 and 10
        n_bins = int(np.ceil(3 * (omega_max - 2) / (10 - 2)))
    if log:
        space = np.geomspace
    else:
        space = np.linspace
    bins = space(2, omega_max, n_bins)
    subsets = {}
    indexes = np.digitize(deposits, bins)
    for index in np.unique(indexes):
        subsets[index] = []
    for row, index in zip(rows, indexes):
        subsets[index].append(row)
    for index in subsets:
        subsets[index] = np.hstack(subsets[index])
    return bins, subsets


def binned_final(case, material, **kwargs):
    ds = case.fds_sub
    dm_deposit = total_deposit(ds, material)
    deposits = np.asarray([dm_deposit[row['i0']] for row in ds])
    return bin_by_deposit(ds, deposits, **kwargs)


def binned_initial(case, material, **kwargs):
    ds = case.ids_sub
    deposits = material.qpe(ds['p1']) + material.qpe(ds['p2'])
    return bin_by_deposit(ds, deposits, **kwargs)
