"""This module enables parallelized down-conversion using MPI."""

import json
import argparse
import logging
import gc

import numpy as np

from .base import DistributedTask
from ..interface import Configuration
from ..event import Event
from ..ensemble import Ensemble
from ..particle import SHORTNAME_TO_CLASS
from ..material import Material


logging.basicConfig()
LOGGER = logging.getLogger('MPI')
LOGGER.setLevel(logging.DEBUG)


def particle_as_dict(p):
    """Convert a :obj:`Particle` object to a lightweight dict.

    The dictionary form has keys 'shortname', 'momentum', and 'cos_theta'.

    Args:
        p (:obj:`Particle`): particle to convert to a dict.

    Returns:
        dict: a simple dictionary form of the particle.

    """
    return dict(
        shortname=p.shortname,
        momentum=p.momentum,
        cos_theta=p.cos_theta
    )


def particle_from_dict(d, material):
    """Convert the output of `particle_as_dict` back to an object.

    Args:
        d (dict): dictionary to convert to a particle.
        material (:obj:`Material`): the material to use for this particle.
            Material data is not included in the dict representation.

    Returns:
        :obj:`Particle`: an object form of the dict.

    """
    return SHORTNAME_TO_CLASS[d['shortname']](material=material, **d)


class DistributedEnsemble(DistributedTask):
    def _run_root(self, config_spec, **kwargs):
        """Create and run an ensemble.

        Note that the argument `config_spec` is only the config file as a dict,
        as read by `json`. This is because expanding the configuration itself
        takes some work, possibly including a lot of disk reading. We don't
        want to do this on every worker process when only the root process
        needs to know about it.

        Args:
            config_spec (dict): configuration data in the format specified by
                `interface.Configuration`.

        """
        # Expand the configuration
        config = Configuration(**config_spec)
        # Expand these into one task per event
        tasks = []
        for ensemble_task in config.ensemble_tasks:
            for e in ensemble_task.initial:
                if e.final_state is None:
                    final_state = None
                else:
                    final_state = [
                        particle_as_dict(p) for p in e.final_state
                    ]
                tasks.append(
                    dict(
                        ensemble_task_id=ensemble_task.task_id,
                        initial_state=[
                            particle_as_dict(p) for p in e.initial_state
                        ],
                        final_state=final_state,
                        material=e.initial_state[0].material.info(),
                        params=ensemble_task.params,
                        statistics=ensemble_task.statistics
                    )
                )
        # Make a results buffer and run the tasks
        results = []
        self._scatter(tasks, results)
        # The results are all same-shape numpy arrays, so jam them together.
        # But only combine results corresponding to the same EnsembleTask!
        results_by_id = {}
        for result in results:
            try:
                results_by_id[
                    result['ensemble_task_id']
                ].append(result['data'])
            except KeyError:
                results_by_id[
                    result['ensemble_task_id']
                ] = [result['data']]
        # To avoid messing with ID uniqueness, advance particle IDs
        for _, values in results_by_id.items():
            try:
                values[0].dtype
            except AttributeError:
                break
            offset = 0
            id_keys = ['i0', 'i1', 'if']
            id_keys = [
                key for key in id_keys if key in values[0].dtype.names
            ]
            for data in values:
                for key in id_keys:
                    data[key] += offset
                offset = max([np.amax(data[key]) for key in id_keys]) + 1
        # Combine the arrays for each EnsembleTask
        for key, values in results_by_id.items():
            """
            try:
                flat = (len(values[0].shape) == 1)
            except BaseException:
                # "BaseException" is aggressive, but it would be bad to crash
                # for a stupid reason at the output stage!
                flat = False
            if flat:
                # These need to be stacked vertically
                results_by_id[key] = np.vstack(tuple(values))
            else:
                results_by_id[key] = np.hstack(tuple(values))
            """
            # I don't know what went wrong with the above, but a non-stat run
            # was flagged as flat. To-do.
            results_by_id[key] = np.hstack(tuple(values))
        # Store the results in the corresponding EnsembleTask
        for key, result in results_by_id.items():
            config.task_by_id[key].result = result
        # Save to disk
        config.save()

    def _func(self, task, *args, **kwargs):
        """Actually run the chains.

        Args:
            task (dict): a dictionary with the needed parameters to instantiate
                an `Event`. This dictionary must contain two keys:
                    `particle`: a tuple `(shortname, momentum, cos_theta)`.
                    `material`: a dict with keyword arguments to instantiate a
                        `Material` object.

        Returns:
            ndarray: a structured array corresponding to the output of
                `Ensemble.to_npy`.

        """
        material = Material(**task['material'])
        # Build the `Event` from the serialized data
        initial = [
            particle_from_dict(d, material) for d in task['initial_state']
        ]
        if task['final_state'] is not None:
            final = [
                particle_from_dict(d, material) for d in task['final_state']
            ]
        else:
            final = None
        event = Event(initial, material, final_state=final)
        ensemble = Ensemble([event], params=task.get('params', {}))
        ensemble.chain()

        def total_deposit(p0):
            return np.sum(p0.dest.leaf_particles.nondark.energy)

        def qp_angle_mean(p0):
            return np.mean(p0.dest.leaf_particles.quasiparticles.cos_theta)

        stat_functions = {
            'total_deposit': total_deposit,
            'qp_angle_mean': qp_angle_mean
        }
        stats = []
        if task['statistics'] is not None:
            for stat_key in task['statistics']:
                stats.append(stat_functions[stat_key])

        ensemble_data = ensemble.to_npy(stats=stats)

        # If a statistic has been provided, return ONLY that statistic,
        # one per event. An easy way to do this is to subset to the rows with
        # a DM final state, since there is only one of these per event.
        if task['statistics'] is not None:
            ensemble_data = ensemble_data[ensemble_data['nf'] == b'DM']
            data = []
            for i in range(len(task['statistics'])):
                data.append(ensemble_data['s%d' % i][0])
            data = np.asarray(data)
        else:
            data = ensemble_data

        result = {
            'ensemble_task_id': task['ensemble_task_id'],
            'data': data
        }

        # Just make sure the garbage gets taken out
        del (
            task, initial, material, final, event, ensemble, ensemble_data,
            data
        )
        gc.collect()

        return result


if __name__ == '__main__':
    # Take only a config file path as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    # Read the config file as a json
    with open(args.config_file, 'r') as fh:
        config_spec = json.load(fh)
    # Initialize and run the ensemble
    ensemble = DistributedEnsemble()
    ensemble.run(config_spec)
