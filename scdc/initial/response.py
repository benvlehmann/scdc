"""This module defines coherence factors for use in computing the initial
angular distribution of excitations produced by a hard scatter."""

import tensorflow as tf


class ResponseFunction(object):
    """Base class for response functions."""
    def __call__(self, r1, r2, q, omega):
        """Evaluate the response function.

        Args:
            r1 (float): magnitude of momentum of QP 1.
            r2 (float): magnitude of momentum of QP 2.
            q (float): magnitude of total momentum transfer.
            omega (float): energy deposit.

        Returns:
            float: value of the response function.

        """
        raise NotImplementedError


class HybridResponseFunction(ResponseFunction):
    """Approximate (BCS + Lindhard) response function.

    This response function approximates the numerator of the loss function,
    :math:`\mathrm{Im}(\epsilon_{\mathrm{BCS}})`, in terms of the BCS
    coherence factor, yielding a response function of the form

        :math:`F_{\mathrm{BCS}} / |\epsilon_{\mathrm{L}}|^2.`

    Args:
        material (:obj:`Material`): material object.
        coherence_sign (int): sign in the coherence factor (1 or -1).

    """
    def __init__(self, material, coherence_sign):
        self.material = material
        self.coherence_sign = coherence_sign

    def __call__(self, r1, r2, q, omega):
        return tf.abs(
            self.material.coherence_uvvu(
                self.coherence_sign, r1, r2
            )
        ) / tf.abs(self.material.epsilon_lindhard(q, omega))**2
