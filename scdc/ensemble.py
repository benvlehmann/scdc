"""This module defines the :obj:`Ensemble` class to facilitate simulations of
down-conversion with many copies of the initial state.

"""

from copy import copy
from itertools import count

import numpy as np

from .particle import ParticleCollection


class Ensemble(object):
    """Represents an ensemble of events.

    Args:
        initial (:obj:`list` of :obj:`Event`): initial events.
        copies (int, optional): number of copies of each initial event to use.
            Defaults to 1.
        params (dict, optional): arbitrary dictionary of parameters that may be
            used to label this ensemble. Must be json-serializable.
            Defaults to `{}`.

    Attributes:
        events (:obj:`list` of :obj:`Event`): initial events.
        initial_state (:obj:`list` of :obj:`Particle`): particles in the
            initial states of the events comprising the ensemble.
        out (:obj:`list` of :obj:`Particle`): particles in the final states of
            the events comprising the ensemble.

    """
    def __init__(self, initial, copies=1, params={}):
        self.base_initial = ParticleCollection()
        for e in initial:
            self.base_initial.extend(e.initial_state)
        self.copies = copies
        self.events = initial
        self.params = params
        self.out = ParticleCollection()
        self.initial_state = ParticleCollection()
        extra_copies = copies - 1
        if extra_copies:
            initial_events = copy(self.events)
            for _ in range(extra_copies):
                self.events.extend([copy(p) for p in initial_events])
        for event in self.events:
            self.initial_state.extend(event.initial_state)
        self._leaf_events = []  # Cache for leaf events
        self._leaf_particles = ParticleCollection()  # Cache for final states
        self._cache_ok = False  # Whether the caches are up-to-date

    def chain(self):
        """Run chains from all initial particles."""
        self.out = ParticleCollection()
        for e in self.events:
            e.chain()
            self.out.extend(e.out)
        self._cache_ok = False

    @property
    def leaf_events(self):
        """:obj:`Ensemble`: a new `Ensemble` with the leaves as events."""
        if self._cache_ok:
            return self._leaf_events
        self._leaf_events = []
        self._leaf_particles = ParticleCollection()
        for e in self.events:
            self._leaf_events.extend(e.leaf_events)
            self._leaf_particles.extend(e.leaf_particles)
        self._leaf_events = Ensemble(self._leaf_events)
        self._cache_ok = True
        return Ensemble(self._leaf_events)

    @property
    def leaf_particles(self):
        """:obj:`list` of :obj:`Particle`: final particles after relaxation."""
        if not self._cache_ok:
            # Access `self.leaf_events` to rebuild the cache
            self.leaf_events
        return self._leaf_particles

    def __len__(self):
        return len(self.events)

    def __iter__(self):
        return iter(self.events)

    def __getitem__(self, *args, **kwargs):
        return self.events.__getitem__(*args, **kwargs)

    def __add__(self, other):
        """Combine two ensembles.

        Args:
            other (:obj:`Ensemble`): an ensemble to combine with.

        """
        combined = Ensemble(self.events + other.events)
        return combined

    def __radd__(self, other):
        """Combination of ensembles is commutative."""
        return self.__add__(other)

    def info(self):
        """Ensemble data in json-serializable format.

        Presently, this includes only the material data and number of copies.

        Returns:
            dict: ensemble data.

        """
        info_dict = dict(params=self.params, copies=self.copies)
        try:
            info_dict['material'] = self.initial_state[0].material.info()
        except IndexError:
            info_dict['material'] = None
        return info_dict

    def to_npy(self, stats=None):
        """Produce a simplified numpy representation.

        The numpy format is a structured array that gives data for the initial
        excitation, all of the immediate children, and all of the leaves. All
        intermediate layers are omitted, so the full tree structure CANNOT be
        reconstructed from this output. The column structure is as follows:

            i0  n0  p0  c0  i1  n1  p1  c1  if  nf  pf  cf

        Here `_0` denotes the initial excitation and `_f` denotes a final
        (leaf) excitation. The letters `inpc` indicate `ID`, `name`,
        `momentum`, and `cos`, respectively. Note that the IDs and the
        corresponding data may not be unique. For example, particle 1 can also
        be a leaf, in which case `i_1 = i_f`.

        Additional columns can be added with ``stats``, with column headers
        's0', 's1', etc.

        Args:
            stats (:obj:`list` of :obj:`func`, optional): a list of statistics
                (i.e. functions) to be evaluated on each initial particle.
                Each function must return a single float. Defaults to None.

        Returns:
            :obj:`ndarray`: the array form of this collection.

        """
        rows = []
        id_iter = count()
        for p0 in self.initial_state:
            if p0.pid is None:
                p0.pid = next(id_iter)
            data_0 = [p0.pid, p0.shortname, p0.momentum, p0.cos_theta]
            data_s = []
            if stats is not None:
                for stat in stats:
                    data_s.append(stat(p0))
            if p0.dest.final:
                # Count it again as p1
                p0_out = ParticleCollection([p0])
            else:
                # Use the next outputs for p1
                p0_out = p0.dest.out
            for p1 in p0_out:
                if p1.pid is None:
                    p1.pid = next(id_iter)
                data_1 = [p1.pid, p1.shortname, p1.momentum, p1.cos_theta]
                for pf in p1.dest.leaf_particles:
                    if pf.pid is None:
                        pf.pid = next(id_iter)
                    data_f = [pf.pid, pf.shortname, pf.momentum, pf.cos_theta]
                    rows.append(tuple(data_0 + data_1 + data_f + data_s))
        dtype = [
            ('i0', '<i4'), ('n0', 'S5'), ('p0', '<f4'), ('c0', '<f4'),
            ('i1', '<i4'), ('n1', 'S5'), ('p1', '<f4'), ('c1', '<f4'),
            ('if', '<i4'), ('nf', 'S5'), ('pf', '<f4'), ('cf', '<f4'),
        ]
        if stats is not None:
            for i, stat in enumerate(stats):
                dtype.append(('s%d' % i, '<f4'))
        return np.asarray(rows, dtype=dtype)
