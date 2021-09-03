"""The structure of the calculation is as follows.

A scattering event produces two quasiparticles. Each of those quasiparticles
can subsequently relax by emitting phonons. If sufficiently energetic, those
phonons can themselves decay to quasiparticle pairs. The process continues
until none of the products have enough energy to undergo another scatter or
decay.

This picture describes a tree of particles with interactions as the nodes of
the tree, and we use a very similar structure to implement the calculation
programmatically. Our tree consists of three types of object:

    - Event
    - Interaction
    - Particle

In tree language, :obj:`Event` objects are nodes and :obj:`Particle` objects
are edges. :obj:`Interaction` objects couple a single :obj:`Event` object to
a collection of :obj:`Particle` objects.

Why not just have :obj:`Interaction` and :obj:`Particle` objects? Because a
third type (:obj:`Event`) is required to specify the relationship between the
two. The :obj:`Interaction` types certainly need to know
about :obj:`Particle` types in order to create the final state of each
interaction. But then how would a :obj:`Particle` know about
the :obj:`Interaction` types for which it can be an initial state?
Alternatively, how would each :obj:`Interaction` type know about all the
others? At a technical level, this would require a circular import. At a
conceptual level, it indicates a mutual dependency between :obj:`Particle`
and :obj:`Interaction` types that deserves its own container.

So instead, the structure is as follows. The nodes in the tree
are :obj:`Event` objects. :obj:`Particle` objects are the edges
between :obj:`Event` objects. The :obj:`Event` objects "know"
what :obj:`Interaction` types are possible for a given :obj:`Particle`, and
they use such an :obj:`Interaction` object internally to produce a new final
state. The :obj:`Event` can then create new :obj:`Event` nodes at which to
terminate each final-state :obj:`Particle`.

Each :obj:`Interaction` can also be physically prohibited. When this happens,
the "initial" state of such an :obj:`Interaction` is a leaf of the tree.

"""

from copy import copy

from .common import NotAllowedException
from .particle import DarkMatter, ParticleCollection
from .interaction import (
    DarkMatterScatter,
    QuasiparticlePhononEmission,
    PhononDecayToQuasiparticles
)


INTERACTIONS = (
    DarkMatterScatter,
    QuasiparticlePhononEmission,
    PhononDecayToQuasiparticles
)


class Event(object):
    """Interaction tree node.

    Args:
        initial_state (:obj:`list` of :obj:`Particle`): particles in the
            initial state. The initial state is always listlike, even if it
            contains only one object.
        material (:obj:`Material`): the material in which this takes place.
        final_state (:obj:`list` of :obj:`Particle`, optional): particles in
            the final state. This is passed to the internal :obj:`Interaction`
            object if supplied.

    Attributes:
        initial_state (:obj:`list` of :obj:`Particle`): particles in the
            initial state. The initial state is always a list, even if it
            contains only one object.
        final_state (:obj:`list` of :obj:`Particle`, optional): particles in
            the final state if and only if supplied at initialization.
        material (obj:`Material`): the material in which this takes place.
        interaction (:obj:`Interaction`): the interaction which can take place
            given the supplied initial state.
        out (:obj:`list` of :obj:`Particle`): particle `Event` objects.
        final (bool): `True` if no further interaction can take place.

    """
    def __init__(self, initial_state, material, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.initial_state, self.material = initial_state, material
        self.final_state = kwargs.get('final_state')
        # Set the initial state's `dest` appropriately
        for particle in self.initial_state:
            particle.dest = self
        self._out = ParticleCollection()  # Final-state particles
        self._leaf_events = []  # Cache for leaf events
        self._leaf_particles = ParticleCollection()  # Cache for final states
        self._cache_ok = False  # Whether the caches are up-to-date
        self.final = kwargs.get('final', False)
        #
        # Find the appropriate interaction for the given initial state
        self.interaction = None
        if not self.final:
            for InteractionType in INTERACTIONS:
                if InteractionType.valid_initial(self.initial_state):
                    self.interaction = InteractionType(
                        initial_state, material, *args, **kwargs
                    )
            if self.interaction is None:
                raise ValueError(
                    ("The supplied initial state does not apppear to be a "
                     "valid initial state for any interactions")
                )

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, particle_list):
        """Hook up each particle's `origin` attribute to this event."""
        self._out = particle_list
        for particle in particle_list:
            particle.origin = self
        self._cache_ok = False

    @property
    def leaf_events(self):
        """:obj:`list` of :obj:`Event`: final events in the detector.

        These are the leaves of the event tree, corresponding to the final
        states that will be measured.

        """
        # Use cache if we can
        if self._cache_ok:
            return self._leaf_events
        # Clear and rebuild the caches
        self._leaf_events = []
        self._leaf_particles = ParticleCollection()
        if self.final:
            self._leaf_events.append(self)
            self._leaf_particles.extend(self.initial_state)
        else:
            for particle in self.out:
                self._leaf_events.extend(particle.dest.leaf_events)
                for event in particle.dest.leaf_events:
                    self._leaf_particles.extend(event.leaf_particles)
        # Mark the cache as up-to-date
        self._cache_ok = True
        return self._leaf_events

    @property
    def leaf_particles(self):
        """:obj:`list` of :obj:`Particle`: final particles after relaxation."""
        if not self._cache_ok:
            # Access `self.leaf_events` to rebuild the cache
            self.leaf_events
        return self._leaf_particles

    def act(self):
        """Run the interaction and generate a final state."""
        try:
            self.out = self.interaction.interact()
        except NotAllowedException:
            # Finalize
            self.final = True
            return
        for particle in self.out:
            # Each of these final states must terminate in an `Event`
            # Don't propagate the `final_state` arg
            kwargs = copy(self.kwargs)
            kwargs['final_state'] = None
            # Special case: for the initial DM scatter, the final-state DM
            # should not rescatter, so make the corresponding event `final`.
            if isinstance(particle, DarkMatter):
                kwargs['final'] = True
            particle.dest = Event(
                [particle], self.material, *self.args, **kwargs
            )
        self._cache_ok = False

    def chain(self):
        """Run the interaction and run particle interactions recursively."""
        if self.final:
            return
        self.act()
        for particle in self.out:
            particle.dest.chain()
        self._cache_ok = False

    def __repr__(self):
        template = "<%(name)s [%(initial)s]---%(action)s--->[%(final)s]>"
        data = dict(
            name=self.__class__.__name__,
            action=self.interaction.__class__.__name__,
            initial=(
                ','.join([p.shortname for p in self.initial_state])
            )
        )
        if self.final:
            data['final'] = 'final'
            data['action'] = ''
        elif self.out:
            data['final'] = (
                ','.join([p.shortname for p in self.out])
            )
        else:
            data['final'] = '??'
        return template % data

    def __copy__(self):
        copied_args = [copy(arg) for arg in self.args]
        copied_kwargs = {}
        for key in self.kwargs:
            copied_kwargs[key] = copy(self.kwargs[key])
        copied_initial_state = [copy(p) for p in self.initial_state]
        if 'final_state' in copied_kwargs:
            copied_kwargs['final_state'] = [
                copy(p) for p in copied_kwargs['final_state']
            ]
        return self.__class__(
            copied_initial_state, self.material, *copied_args, **copied_kwargs
        )
