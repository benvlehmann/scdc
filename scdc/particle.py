"""This module defines the different types of (quasi)particle."""

import numpy as np

from .common import NonPhysicalException, tolerate

REQUIRED_KEYS = ('cos_theta', 'material')


class Particle(object):
    """Represents a single phonon or quasiparticle with a tree of children.

    `momentum` must be specified unless `dispersion_inverse` is defined, in
    which case either `momentum` or `energy` can be specified.

    All quantities are specified in material units.

    Args:
        energy (float, optional): energy in units of Delta.
        momentum (float, optional): momentum in canonical units.
        cos_theta (float): cosine of angle wrt global z.
        material (:obj:`Material`): material properties.
        pid (object, optional): optional ID label. Defaults to ``None``.
        tolerance (float, optional): tolerance for :math:`\\cos\\theta > 1`.
            Defaults to ``1e-2``.

    Attributes:
        energy (float): energy in units of Delta.
        momentum (float, optional): momentum in canonical units.
        cos_theta (float): cosine of angle wrt global z.
        material (:obj:`Material`): material properties.
        pid (object, optional): optional ID label.
        tolerance (float, optional): tolerance for :math:`\\cos\\theta > 1`.
        parent (:obj:`Particle`): parent particle.
        origin (:obj:`Event`): the originating `Event` for this particle.
        dest (:obj:`Event`): the destination `Event` for this particle.
        shortname (str): a short label for this particle type.

    """

    shortname = "Pa"

    def __init__(self, **kwargs):
        self._energy = None
        self._momentum = None
        self.origin = kwargs.get('origin')
        self.dest = kwargs.get('dest')
        self.pid = kwargs.get('pid')
        self.tolerance = kwargs.get('tolerance', 1e-1)
        for key in REQUIRED_KEYS:
            setattr(self, key, kwargs.get(key))
        try:
            self.momentum = kwargs['momentum']
        except KeyError:
            try:
                self.energy = kwargs['energy']
            except KeyError:
                raise ValueError("Must supply `momentum` or `energy`")
        if ~np.isfinite(self.cos_theta):
            raise NonPhysicalException(
                "Angle '%r' is non-numeric" % self.cos_theta
            )
        if np.imag(self.cos_theta) != 0:
            raise NonPhysicalException(
                "Angle '%r' is complex" % self.cos_theta
            )
        # If |cos(theta)| > 1, this will produce nan's in cosine addition later
        # Trim the cosine to 1 within tolerance
        self.cos_theta = tolerate(self.cos_theta, self.tolerance)
        # Check whether the resulting angle is OK
        if np.abs(self.cos_theta) > 1:
            raise NonPhysicalException(
                "Angle '%r' is outside tolerance" % self.cos_theta
            )

    def dispersion(self, k):
        """Disperion relation: wavenumber to energy."""
        raise NotImplementedError

    def dispersion_inverse(self, E):
        """Inverse dispersion relation: energy to wavenumber."""
        raise NotImplementedError

    @property
    def momentum(self):
        """float: momentum in material units."""
        if self._momentum is None:
            self._momentum = self.dispersion_inverse(self._energy)
        return self._momentum

    @momentum.setter
    def momentum(self, k):
        """Set the momentum and then set the energy accordingly."""
        self._momentum = k
        self._energy = self.dispersion(k)

    @property
    def energy(self):
        """float: energy in units of Delta."""
        if self._energy is None:
            self._energy = self.dispersion(self._momentum)
        return self._energy

    @energy.setter
    def energy(self, E):
        """Set the energy and then try to set the momentum accordingly."""
        self._energy = E
        try:
            self._momentum = self.dispersion_inverse(E)
        except NotImplementedError:
            raise NonPhysicalException(
                ("The dispersion relation is non-invertible, so I don't know "
                 "how to set the momentum from the energy. Please set the "
                 "momentum directly instead.")
            )

    def __copy__(self):
        kwargs = {}
        for key in REQUIRED_KEYS:
            kwargs[key] = getattr(self, key)
        kwargs['momentum'] = self.momentum
        try:
            kwargs['mass'] = self.mass
        except AttributeError:
            pass
        self_copy = self.__class__(**kwargs)
        return self_copy

    def __repr__(self):
        return "<%s E=%.3f, k/kF-1=%.3e, cos(theta)=%.3f>" % (
            self.shortname,
            self.energy,
            self.momentum - 1,
            self.cos_theta
        )


class DarkMatter(Particle):
    """A dark matter particle."""

    shortname = "DM"

    def __init__(self, **kwargs):
        try:
            self.mass = kwargs['mass']
        except KeyError:
            # raise ValueError("Must supply DM mass")
            # This is a bit of a workaround. We don't actually need the DM mass
            # to use this as a container class. Still, be aware that energies
            # and momenta cannot be computed without supplying the mass.
            self.mass = 1
        super().__init__(**kwargs)

    def dispersion(self, p):
        """Dispersion relation for dark matter.

        Be careful: there is a need for conversion to the material's own
        canonical units. (NOT currently implemented.)

        Args:
            p (float): momentum in who-knows-what units...to fix.

        Returns:
            float: dark matter energy in units of Delta.

        """
        return p**2 / (2*self.mass)


class Phonon(Particle):
    """A phonon."""

    shortname = "Ph"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decay_approx = kwargs.get('decay_approx', False)

    def dispersion(self, k):
        """Dispersion relation for phonons.

        Args:
            k (float): wavenumber in material units.

        Returns:
            float: phonon energy in material units.

        """
        return 2*self.material.gamma*np.sqrt(self.material.z)

    def dispersion_inverse(self, E):
        """Inverse dispersion relation for phonons.

        Args:
            E (float): phonon energy in material units.

        Returns:
            float: phonon wavenumber in material units.

        """
        return E / (self.material.hbar_m * self.material.c_s)


class Quasiparticle(Particle):
    """A Bogoliubov quasiparticle."""

    shortname = "QP"

    def __init__(self, **kwargs):
        self._ksign = kwargs.get('ksign')
        super().__init__(**kwargs)

    def dispersion(self, k):
        """Dispersion relation for quasiparticles.

        Args:
            k (float): wavenumber in material units.

        Returns:
            float: quasiparticle energy in material units.

        """
        return np.sqrt(
            1 + (
                (k*self.material.hbar_m)**2 / (2*self.material.m_star_m) -
                self.material.z
            )**2
        )

    def dispersion_inverse(self, E):
        """Inverse dispersion relation for quasiparticles.

        Warning: the dispersion relation is not actually invertible. There are
        two possible choices of a sign. If the sign `s` is not specified, this
        method selects it randomly!

        Args:
            E (float): quasiparticle energy in material units.
            s (int, optional): sign in the momentum solution.

        Returns:
            float: quasiparticle wavenumber in material units.

        """
        if self._ksign is None:
            self._ksign = np.random.choice([-1, 1])
        return np.sqrt(
            2*self.material.m_star_m * (
                self.material.z + self.ksign*np.sqrt(E**2 - 1)
            )
        ) / self.material.hbar_m

    @property
    def ksign(self):
        """int: the sign as determined from energy and momentum."""
        if self._ksign is None:
            self._ksign = np.sign(
                (-2*self.material.E_F_m * self.material.m_star_m
                    + self.momentum**2 * self.material.hbar_m**2
                 ) / (
                    2 * self.material.m_star_m * np.sqrt(
                        self.energy**2 - self.material.Delta_m**2
                    )
                )
            )
        return self._ksign

    @ksign.setter
    def ksign(self, s):
        self._ksign = s


SHORTNAME_TO_CLASS = dict(
    DM=DarkMatter,
    Ph=Phonon,
    QP=Quasiparticle
)


class ParticleCollection(object):
    """Represents a collection of particles.

    Args:
        particles (:obj:`list` of :obj:`Particle`): particles.

    Attributes:
        particles (:obj:`list` of :obj:`Particle`): particles.
        energy (:obj:`ndarray` of float): energy of each particle.
        momentum (:obj:`ndarray` of float): momentum of each particle.
        cos_theta (:obj:`ndarray` of float): cos_theta of each particle.

    """
    def __init__(self, *args, **kwargs):
        if len(args):
            self.particles = args[0]
        else:
            self.particles = []

    def _particle_attribute(self, key):
        """Get an attribute for each particle and return as a numpy array.

        Args:
            key (str): the key to retrieve.

        Returns:
            :obj:`ndarray`: a numpy array with one entry for each particle.

        """
        return np.asarray([getattr(p, key) for p in self.particles])

    @property
    def energy(self):
        """:obj:`ndarray`: array of particle energies."""
        return self._particle_attribute('energy')

    @property
    def momentum(self):
        """:obj:`ndarray`: array of particle energies."""
        return self._particle_attribute('momentum')

    @property
    def cos_theta(self):
        """:obj:`ndarray`: array of particle `cos_theta` values."""
        return self._particle_attribute('cos_theta')

    def select(self, test):
        """Select a subset of particles which satisfy a test.

        The argument `test` is a function with call signature

            `test(parent, particle)`

        which returns `True` if the leaf `particle` in the chain arising from
        the initial particle `parent` satisifes the test.

        Args:
            test (function): the test function.

        Returns:
            :obj:`ParticleCollection`: collection consisting of particles which
                satisfy the test.

        """
        survivors = []
        for particle in self.particles:
            # Find the originating particle
            parent = particle
            while parent.origin is not None:
                parent = parent.origin.initial_state[0]
            if test(parent, particle) is True:
                survivors.append(particle)
        return ParticleCollection(survivors)

    @property
    def phonons(self):
        """:obj:`ParticleCollection`: subcollection of phonons."""
        return self.select(lambda _, p: isinstance(p, Phonon))

    @property
    def quasiparticles(self):
        """:obj:`ParticleCollection`: subcollection of quasiparticles."""
        return self.select(lambda _, p: isinstance(p, Quasiparticle))

    @property
    def dm(self):
        """:obj:`ParticleCollection`: subcollection of DM particles."""
        return self.select(lambda _, p: isinstance(p, DarkMatter))

    @property
    def nondark(self):
        """:obj:`ParticleCollection`: subcollection of non-DM particles."""
        return self.select(lambda _, p: not isinstance(p, DarkMatter))

    def __len__(self):
        return len(self.particles)

    def __iter__(self):
        return iter(self.particles)

    def __getitem__(self, *args, **kwargs):
        # Use the `list` __getitem__
        subset = self.particles.__getitem__(*args, **kwargs)
        # If this is a list, wrap it in a ParticleCollection
        if isinstance(subset, list):
            subset = ParticleCollection(subset)
        return subset

    def __add__(self, other):
        """Combine two ParticleCollections.

        Args:
            other (:obj:`ParticleCollection`): a collection to combine with.

        """
        combined = ParticleCollection(self.particles + other.particles)
        return combined

    def __radd__(self, other):
        """Combination of collections is commutative."""
        return self.__add__(other)

    def extend(self, iterable):
        self.particles.extend(iterable)

    def to_npy(self):
        """Represent this particle collection in numpy format for export.

        The numpy format is a structured array with the following columns:

            shortname    momentum    cos_theta

        The numpy form REQUIRES momenta for all particles.

        Returns:
            :obj:`ndarray`: the array form of this collection.

        """
        return np.asarray(
            [(p.shortname, p.momentum, p.cos_theta) for p in self.particles],
            dtype=[
                ('shortname', 'S5'),
                ('momentum', '<f4'),
                ('cos_theta', '<f4')
            ]
        )

    @classmethod
    def from_npy(cls, npy, material):
        """Construct a `ParticleCollection` from the numpy representation.

        Args:
            npy (:obj:`ndarray`): a structured array with columns as specified
                in the `to_npy` method.
            material (:obj:`Material`): the material to associate with these
                particles. The material is not stored in the numpy format.

        Returns:
            :obj:`ParticleCollection`: a corresponding `ParticleCollection`.

        """
        particles = []
        for row in npy:
            ParticleClass = SHORTNAME_TO_CLASS[row['shortname']]
            particles.append(
                ParticleClass(
                    momentum=row['momentum'],
                    cos_theta=row['cos_theta'],
                    material=material
                )
            )
        return ParticleCollection(particles)
