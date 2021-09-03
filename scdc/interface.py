"""This module defines utilities for running ensembles through the command
line or batch queue.

Runs are defined by a configuration file in json format. The json file should
have the following keys:

    outfile: path to the output file. Extension will be appended if missing.
    copies: number of copies of the initial state to run. Defaults to 1.
    initial: initial state to use.

        - If a string, this is taken to be the path to a file. The file should
            have columns of the form

                pDM  cDM  p1  c1  p2  c2

            where `p` and `c` are the momentum and cos(theta) for the DM, QP1,
            and QP2, respectively.

        - If a dict, this is taken to specify the parameters of an initial
            state. In this case, the allowed keys are:
                - `momentum`
                - `energy`
                - `shortname`
                - `cos_theta`

            Exactly one of `energy` and `momentum` must be specified. Any of
            these keys may be given as a list of values. In this case, one
            enemble is produced for each value in the list (or for each
            combination of list values, if applicable).

    material: material data to use (dict).
        - If omitted or null, the default is `materials.Aluminum`.
        - Allowed keys are:
            - `gamma`
            - `c_s`
            - `T_c`
            - `Delta`
            - `E_F`
            - `m_star`
            - `beta`

        - Any keys omitted are taken from the default (`materials.Aluminum`).
        - As with the `initial` key, any values specified as lists will produce
            one ensemble for each value or combination of values.

    params: an arbitrary dict of additional labels with primitive values.

"""


from copy import copy
from itertools import product

import numpy as np
import h5py

from .material import Material
from .materials import ALUMINUM
from .particle import DarkMatter, Quasiparticle, SHORTNAME_TO_CLASS
from .event import Event


BASE_MATERIAL_KWARGS = {'symbol': 'Sy'}
for key in ('T_c', 'E_F', 'Delta', 'm_star', 'c_s', 'beta', 'gamma'):
    BASE_MATERIAL_KWARGS[key] = getattr(ALUMINUM, key)


class EnsembleTask(object):
    """Container class for single ensembles.

    Args:
        initial (:obj:`list` of :obj:`Event`): initial events.
        material (:obj:`Material`): material object.
        params (dict, optional): arbitrary parameter dict for labels.
            Defaults to `{}`.
        task_id (object, optional): a label for this task. Defaults to `None`.

    Attributes:
        initial (:obj:`list` of :obj:`Event`): initial events.
        material (:obj:`Material`): material object.
        params (dict): arbitrary parameter dict for labels.
        task_id (object): a label for this task.
        result (:obj:`ndarray`): results of running this task.

    """
    def __init__(self, **kwargs):
        self.initial = kwargs.get('initial')
        self.material = kwargs.get('material')
        self.task_id = kwargs.get('task_id')
        self.params = kwargs.get('params', {})
        self.statistics = kwargs.get('statistics')
        self.result = None


def expand_dict(d):
    """Expand a dict of lists to a list of dicts.

    The idea is to take a dict for which some values are iterable, and convert
    this to a single list of dicts, each of which has no listlike values.

    Args:
        d (dict): dict to expand.

    Returns:
        :obj:`list` of :obj:`dict`: expanded list of dicts.

    """
    # Make every value iterable, and also wrap strings in lists.
    for key, value in d.items():
        if not isinstance(value, list):
            d[key] = [value]
    # Take the product of all these iterables
    keys = list(d.keys())
    combinations = product(*[d[key] for key in keys])
    # Make one dict for each combination
    single_dicts = []
    for values in combinations:
        single_dict = {}
        for key, value in zip(keys, values):
            single_dict[key] = value
        single_dicts.append(single_dict)
    return single_dicts


def _parse_n_initial(n_initial, m_med, m_dm):
    """Parser for ``n_initial``.

    Args:
        n_initial (int or dict, optional): number of initial particles to
            select for downconversion. Selection is random with replacement. If
            ``None``, use all initial particles. A two-layer dict may be
            provided using the mediator mass and DM mass as keys, with an
            'other' entry as a fallback.
        m_med (float): mediator mass for which to determine ``n_initial``.
        m_dm (float): DM mass for which to determine ``n_initial``.

    Returns:
        int: number of initial particles to select for downconversion.

    """
    if n_initial is None:
        return None
    if isinstance(n_initial, int):
        return n_initial
    # Assume this is dict-like, with the first key being 'heavy' or 'light'
    if m_med == 0:
        med_key = 'light'
    else:
        med_key = 'heavy'
    n_cut = n_initial[med_key]
    if isinstance(n_cut, int):
        return n_cut
    # Assume that ``n_cut`` is a list of pairs (mass, number)
    for mass, number in n_cut:
        if mass == m_dm:
            return number
    # Try the 'other' fallback
    try:
        n_cut = n_initial['other']
    except KeyError:
        return None


class Configuration(object):
    """Configuration generator for multi-task downconversion runs.

    Args:
        outfile (str): path to the output file. Extension will be appended
            if missing.
        copies (int, optional): number of copies to run. Defaults to 1.
        initial (object): initial particle specification.
        n_initial (int or dict, optional): number of initial particles to
            select for downconversion. Selection is random with replacement. If
            ``None``, use all initial particles. A two-layer dict may be
            provided using the mediator mass and DM mass as keys, with an
            'other' entry as a fallback. Defaults to ``None``.
        material (object): material specification.
        params (dict, optional): arbitrary parameter dict for labels.
            Defaults to `{}`.


    Attributes:
        outfile (str): original outfile specification.
        copies (int): original copies specification.
        material (object): original material specification.
        params (dict): original params specification.
        materials (object): material specifications.
        ensemble_tasks (:obj:`list` of :obj:`EnsembleTask`): one task per
            ensemble.
        task_by_id (dict): a dictionary mapping ensemble task IDs to
            `EnsembleTask` objects.

    """
    def __init__(self, **kwargs):
        self.copies = kwargs.get('copies', 1)
        self.initial = kwargs.get('initial')
        self.n_initial = kwargs.get('n_initial')
        self.material = kwargs.get('material')
        self.params = kwargs.get('params', {})
        self.outfile = kwargs.get('outfile')
        self.omega_max = kwargs.get('omega_max')
        self.m_dm_max = kwargs.get('m_dm_max')
        self.cf_sign = kwargs.get('cf_sign')
        self.statistics = kwargs.get('statistics')
        self.ensemble_tasks = []
        try:
            extra_copies = int(self.copies) - 1
        except BaseException:
            raise ValueError
        if self.material is None:
            self.materials = [ALUMINUM]
        elif isinstance(self.material, dict):
            # Set defaults from the base material (aluminum)
            for key, value in BASE_MATERIAL_KWARGS.items():
                try:
                    self.material[key]
                except KeyError:
                    self.material[key] = value
            # Expand to a list of single material specifications
            material_dicts = expand_dict(self.material)
            # Instantiate `Material` objects
            self.materials = [Material(**md) for md in material_dicts]
        else:
            raise ValueError
        if isinstance(self.initial, str):
            if self.initial.endswith('csv'):
                # Create one task per material, loading particles from a file
                initial_data = [np.loadtxt(self.initial, delimiter=',')]
                initial_meta = [{}]
            elif self.initial.endswith('h5') or self.initial.endswith('hdf5'):
                # Assume each key is a separate set of initial conditions

                def _include(ds):
                    """Test whether we should include a given dataset"""
                    if ds.attrs['failed']:
                        return False
                    if self.m_dm_max is not None:
                        if ds.attrs['m_dm'] > self.m_dm_max:
                            return False
                    if self.cf_sign is not None:
                        if ds.attrs['cf_sign'] != self.cf_sign:
                            return False
                    return True

                with h5py.File(self.initial, 'r') as hdf:
                    initial_data = [
                        ds[:] for ds in hdf['initials'].values()
                        if _include(ds)
                    ]
                    initial_meta = [
                        {
                            'm_dm': ds.attrs['m_dm'],
                            'm_med': ds.attrs['m_med'],
                            'cf_sign': ds.attrs['cf_sign'],
                            'vdf': ds.attrs['vdf']
                        }
                        for ds in hdf['initials'].values()
                        if _include(ds)
                    ]
                    # Subsample if requested
                    for i, (data, meta) in enumerate(
                        zip(initial_data, initial_meta)
                    ):
                        n_cut = _parse_n_initial(
                            self.n_initial, meta['m_med'], meta['m_dm']
                        )
                        if n_cut is not None:
                            index = np.random.choice(
                                np.arange(len(data), dtype=int),
                                size=n_cut
                            )
                            initial_data[i] = data[index]
            # Create events
            for material, (initial, imeta) in product(
                self.materials, zip(initial_data, initial_meta)
            ):
                # Create one event per line in the file
                events = []
                for pdm, cdm, p1, c1, p2, c2 in initial:
                    # Past errors have generated unphysical angles. Check!
                    if np.abs(c1) > 1 or np.abs(c2) > 1:
                        raise ValueError("Data contains |cos(theta)|>1")
                    if np.imag(c1) != 0 or np.imag(c2) != 0:
                        raise ValueError("Data contains non-real angles")
                    E1 = np.sqrt(
                        material.Delta_m**2 + (
                            p1**2/(2*material.m_star_m)
                            - material.E_F_m
                        )**2
                    )
                    E2 = np.sqrt(
                        material.Delta_m**2 + (
                            p2**2/(2*material.m_star_m)
                            - material.E_F_m
                        )**2
                    )
                    omega = E1 + E2
                    # Filter
                    if self.omega_max is not None:
                        if omega > self.omega_max:
                            continue
                    # TODO: right now, DM mass is neglected if unavailable.
                    try:
                        m_dm = imeta['m_dm'] / material.m
                    except KeyError:
                        m_dm = 1
                    dmi = DarkMatter(momentum=pdm, cos_theta=cdm,
                                     mass=m_dm, material=material)
                    qp1 = Quasiparticle(momentum=p1, cos_theta=c1,
                                        material=material)
                    qp2 = Quasiparticle(momentum=p2, cos_theta=c2,
                                        material=material)
                    # TODO: compute the DM final state momentum
                    dmf = DarkMatter(momentum=pdm, cos_theta=cdm,
                                     mass=m_dm, material=material)
                    events.append(
                        Event([dmi], material, final_state=[dmf, qp1, qp2])
                    )
                
                # Copy if requested
                if extra_copies:
                    extra_events = []
                    for _ in range(extra_copies):
                        extra_events.extend([copy(e) for e in events])
                    events.extend(extra_events)
                params = {}
                params.update(self.params)
                params.update(imeta)
                self.ensemble_tasks.append(
                    EnsembleTask(
                        material=material,
                        initial=events,
                        params=params,
                        statistics=self.statistics
                    )
                )
        elif isinstance(self.initial, dict):
            initial_dicts = expand_dict(self.initial)
            for material in self.materials:
                # Expand initial particle specification into a list of dicts
                for initial_dict in initial_dicts:
                    initial_dict['material'] = material
                # Turn these into particles
                particles = [
                    SHORTNAME_TO_CLASS[
                        initial_dict['shortname']
                    ](**initial_dict)
                    for initial_dict in initial_dicts
                ]
                # One EnsembleTask for each of these
                for particle in particles:
                    ensemble_particles = [particle]
                    # Copy if requested
                    if extra_copies:
                        extra_particles = []
                        for _ in range(extra_copies):
                            extra_particles.extend(
                                [copy(p) for p in ensemble_particles]
                            )
                        ensemble_particles.extend(extra_particles)
                    # Make corresponding events
                    events = [Event([p], material) for p in ensemble_particles]
                    self.ensemble_tasks.append(
                        EnsembleTask(
                            material=material,
                            initial=events,
                            params=self.params,
                            statistics=self.statistics
                        )
                    )
        else:
            raise ValueError
        # Give these tasks unique IDs and dictionary access
        self.task_by_id = {}
        for i, task in enumerate(self.ensemble_tasks):
            task.task_id = i
            self.task_by_id[i] = task

    def save(self):
        """Save an HDF5 representation of this run and the output."""
        path = self.outfile
        if not path.endswith('.hdf5'):
            path += '.hdf5'
        with h5py.File(path, 'w') as hdf:
            # Store the original configuration data in json form
            config_group = hdf.create_group('config')
            config_group.attrs.update({
                'outfile': self.outfile,
                'copies': self.copies
            })
            if isinstance(self.initial, dict):
                initial_group = config_group.create_group('initial')
                initial_group.attrs.update(self.initial)
            else:
                config_group.attrs['initial'] = self.initial
            material_group = config_group.create_group('material')
            if self.material is not None:
                material_group.attrs.update(self.material)
            # Store each EnsembleTask as its own dataset
            ensemble_group = hdf.create_group('ensembles')
            for ensemble_task in self.ensemble_tasks:
                if ensemble_task.result is None:
                    # TODO: the fact that this sometimes happens suggests
                    # something is horribly wrong
                    continue
                dataset = ensemble_group.create_dataset(
                    str(ensemble_task.task_id),
                    ensemble_task.result.shape,
                    dtype=ensemble_task.result.dtype
                )
                # Store data
                dataset[:] = ensemble_task.result[:]
                # Store attributes
                dataset.attrs['task_id'] = ensemble_task.task_id
                dataset.attrs.update(ensemble_task.params)
                dataset.attrs.update(ensemble_task.material.info())
                # Store data for the first initial particle
                particle = ensemble_task.initial[0].initial_state[0]
                # TODO: store better event data
                dataset.attrs.update({
                    'shortname': particle.shortname,
                    'energy': particle.energy,
                    'momentum': particle.momentum,
                    'cos_theta': particle.cos_theta,
                })
                if self.statistics is not None:
                    for i, stat in enumerate(self.statistics):
                        dataset.attrs['s%d' % i] = stat


class InitialTask(object):
    """Container class for single initial-ensemble generation tasks.

    At present, it is assumed that only the DM and mediator mass vary. The
    velocity is fixed to 1e-3.

    """
    def __init__(self, *args, **kwargs):
        self.mmed = kwargs.get('m_med')
        self.m1 = kwargs.get('m_dm')
        self.n_samples = kwargs.get('n_samples')
        self.material = kwargs.get('material')
        self.task_id = kwargs.get('task_id')
        self.cf_sign = kwargs.get('cf_sign')
        self.vdf = kwargs.get('vdf')
        self.params = kwargs.get('params', {})
        self.result = None
        # Unit conversions from eV
        self.mmed_m = self.mmed / self.material.m
        self.m1_m = self.m1 / self.material.m
        self.r1 = (1e-3 / self.material.v) * self.m1_m
        self.omega_max = kwargs.get('omega_max')
        self.support_threshold = kwargs.get('support_threshold')


class InitialConfiguration():
    """Configuration generator for multi-task initial-particle runs.

    At present, it is assumed that only the DM and mediator mass vary. The
    velocity is fixed to 1e-3.

    """
    def __init__(self, **kwargs):
        self.material = kwargs.get('material')
        self.cf_sign = kwargs.get('cf_sign')
        self.vdf = kwargs.get('vdf')
        self.m_dm = kwargs.get('m_dm')
        self.m_med = kwargs.get('m_med')
        self.n_samples = kwargs.get('n_samples')
        self.params = kwargs.get('params', {})
        self.omega_max = kwargs.get('omega_max')
        self.outfile = kwargs.get('outfile')
        self.initial_tasks = []
        if self.material is None:
            self.materials = [ALUMINUM]
        elif isinstance(self.material, dict):
            # Set defaults from the base material (aluminum)
            for key, value in BASE_MATERIAL_KWARGS.items():
                try:
                    self.material[key]
                except KeyError:
                    self.material[key] = value
            # Expand to a list of single material specifications
            material_dicts = expand_dict(self.material)
            # Instantiate `Material` objects
            self.materials = [Material(**md) for md in material_dicts]
        else:
            raise ValueError
        # Build the tasks themselves
        for material, m1, m2, sign in product(
            self.materials, self.m_dm, self.m_med, self.cf_sign
        ):
            self.initial_tasks.append(
                InitialTask(
                    material=material,
                    n_samples=self.n_samples,
                    cf_sign=sign,
                    vdf=self.vdf,
                    m_dm=m1,
                    m_med=m2,
                    params=self.params,
                    omega_max=self.omega_max
                )
            )
        # Give these tasks unique IDs and dictionary access
        self.task_by_id = {}
        for i, task in enumerate(self.initial_tasks):
            task.task_id = i
            self.task_by_id[i] = task

    def save(self):
        """Save an HDF5 representation of this run and the output."""
        path = self.outfile
        if not path.endswith('.hdf5'):
            path += '.hdf5'
        with h5py.File(path, 'w') as hdf:
            # Store the original configuration data in json form
            config_group = hdf.create_group('config')
            config_group.attrs.update({
                'outfile': self.outfile
            })
            material_group = config_group.create_group('material')
            if self.material is not None:
                material_group.attrs.update(self.material)
            # Store each InitialTask as its own dataset
            ensemble_group = hdf.create_group('initials')
            for initial_task in self.initial_tasks:
                try:
                    shape = initial_task.result.shape
                    dtype = initial_task.result.dtype
                except AttributeError:
                    shape = ()
                    dtype = None
                dataset = ensemble_group.create_dataset(
                    str(initial_task.task_id),
                    shape=shape,
                    dtype=dtype
                )
                # Store data
                if initial_task.result is not None:
                    dataset[:] = initial_task.result[:]
                    dataset.attrs['failed'] = False
                else:
                    dataset.attrs['failed'] = True
                # Store attributes
                dataset.attrs['task_id'] = initial_task.task_id
                dataset.attrs.update(initial_task.params)
                dataset.attrs.update(initial_task.material.info())
                dataset.attrs.update({
                    'm_med': initial_task.mmed,
                    'm_dm': initial_task.m1,
                    'n_samples': initial_task.n_samples,
                    'cf_sign': initial_task.cf_sign,
                    'vdf': initial_task.vdf
                })
                if initial_task.omega_max is not None:
                    dataset.attrs.update({
                        'omega_max': initial_task.omega_max
                    })
