"""This module enables parallelized initial-QP sampling using MPI."""

import json
import argparse
import logging

import numpy as np

from .base import DistributedTask
from ..interface import InitialConfiguration
from ..material import Material
from ..common import NoDataException, NotAllowedException
from ..initial.halo import (
    SingleVelocityDistribution, StandardHaloDistribution, IsotropicDistribution
)
from ..initial.response import HybridResponseFunction
from ..initial.matrix_element import FiducialMatrixElement
from ..initial.distribution.integral import InitialSampler


logging.basicConfig()
LOGGER = logging.getLogger('MPI')
LOGGER.setLevel(logging.DEBUG)
KMS = 3.33564e-6  # km/s in natural units


class DistributedSampler(DistributedTask):
    def _run_root(self, config_spec, **kwargs):
        """Create and run a sampler for initial excitations.

        Note that the argument `config_spec` is only the config file as a dict,
        as read by `json`. This is because expanding the configuration itself
        takes some work, possibly including a lot of disk reading. We don't
        want to do this on every worker process when only the root process
        needs to know about it.

        Args:
            config_spec (dict): configuration data in the format specified by
                `interface.InitialConfiguration`.

        """
        # Expand the configuration
        config = InitialConfiguration(**config_spec)
        # Expand these into one task per event
        tasks = []
        for initial_task in config.initial_tasks:
            tasks.append(
                dict(
                    initial_task_id=initial_task.task_id,
                    m1=initial_task.m1_m,
                    mmed=initial_task.mmed_m,
                    r1=initial_task.r1,
                    cf_sign=initial_task.cf_sign,
                    vdf=initial_task.vdf,
                    n_samples=initial_task.n_samples,
                    material=initial_task.material.info(),
                    params=initial_task.params,
                    omega_max=initial_task.omega_max,
                    support_threshold=initial_task.support_threshold
                )
            )
        # Make a results buffer and run the tasks
        results = []
        self._scatter(tasks, results)
        # Store the results in the corresponding EnsembleTask
        results_by_id = {}
        for result in results:
            results_by_id[result['initial_task_id']] = result
        for key, result in results_by_id.items():
            config.task_by_id[key].result = result['data']
        # Save to disk
        config.save()

    def _func(self, task, *args, **kwargs):
        """

        Returns:
            ndarray: a structured array corresponding to the output of
                `Ensemble.to_npy`.

        """
        material = Material(**task['material'])
        me = FiducialMatrixElement(mediator_mass=task['mmed'])
        response = HybridResponseFunction(material, task['cf_sign'])
        if task['vdf'] == 'SHM':
            vdf = StandardHaloDistribution(
                v_0=220*KMS/material.v,
                v_esc=550*KMS/material.v,
                v_wind=230*KMS/material.v
            )
        elif task['vdf'] == 'isotropic':
            vdf = IsotropicDistribution(
                v_0=220*KMS/material.v,
                v_esc=550*KMS/material.v,
                v_wind=230*KMS/material.v
            )
        else:
            # Assume the vdf is a numerical speed in km/s
            vdf = SingleVelocityDistribution(speed=task['vdf']*KMS/material.v)
        try:
            sampler = InitialSampler(
                task['m1'], me, material, response, vdf,
                omega_max=task['omega_max'],
                support_threshold=task['support_threshold']
            )
            ensemble = sampler.ensemble(task['n_samples'])
        except (NoDataException, NotAllowedException):
            # We tried but could not sample from this. Return something null.
            return {
                'initial_task_id': task['initial_task_id'],
                'data': None
            }
        # Finalize the output state so this can be numpyified
        for e in ensemble:
            for p in e.out:
                p.dest.final = True
        # Reformat the output for initial data
        data = ensemble.to_npy()
        qp_rows = data[data['n1'] == b'QP']
        pair_rows = []
        qpi = iter(qp_rows)
        while True:
            try:
                row_1, row_2 = next(qpi), next(qpi)
            except StopIteration:
                break
            pair_rows.append((
                row_1['p0'], row_1['c0'],
                row_1['p1'], row_1['c1'],
                row_2['p1'], row_2['c1']
            ))
        pair_rows = np.asarray(
            pair_rows,
            dtype=[
                ('pDM', '<f4'), ('cDM', '<f4'),
                ('p1', '<f4'), ('c1', '<f4'),
                ('p2', '<f4'), ('c2', '<f4'),
            ]
        )
        return {
            'initial_task_id': task['initial_task_id'],
            'data': pair_rows
        }


if __name__ == '__main__':
    # Take only a config file path as an argument
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str)
    args = parser.parse_args()
    # Read the config file as a json
    with open(args.config_file, 'r') as fh:
        config_spec = json.load(fh)
    # Initialize and run the ensemble
    sampler = DistributedSampler()
    sampler.run(config_spec)
