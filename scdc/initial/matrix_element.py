"""This module defines matrix elements for DM scattering."""


class ScatteringMatrixElement(object):
    """The **squared** matrix element for scattering.

    Matrix elements may accept arbitrary variables for initialization, but the
    evaluation must accept only the momentum transfer `rq`.

    """
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, rq):
        """Evaluate the matrix element.

        Args:
            rq: magnitude of the 3-momentum transfer.

        Returns:
            float: the squared matrix element.

        """
        raise NotImplementedError


class FiducialMatrixElement(ScatteringMatrixElement):
    """Fiducial squared matrix element for nonrelativistic DM scattering.

    This class represents a matrix element of the form 1/(q^2 + m_med^2)^2.

    Args:
        mediator_mass (float): mediator mass, units TBD.

    Attributes:
        mediator_mass (float): mediator mass, units TBD.

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mediator_mass = kwargs.get('mediator_mass')
        if self.mediator_mass is None:
            raise ValueError("Must supply `mediator_mass`")

    def __call__(self, rq):
        """Evaluate the matrix element (up to a constant).

        Args:
            rq: magnitude of the 3-momentum transfer.

        Returns:
            float: the squared matrix element.

        """
        return 1. / (rq**2 + self.mediator_mass**2)**2
