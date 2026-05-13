"""Custom exception hierarchy that exposes user-friendly error messages.

The exceptions defined here are tailored to the various subsystems of
TransitionListener. They encapsulate common failure modes (configuration,
phase reconstruction, tunnelling, percolation, hydrodynamics) and attach
error codes where the command-line interface needs to surface diagnostic
information to end users.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

# Terminal errors
class NoModelError(Exception):
    """Raised when the model file is not found or cannot be loaded."""

    def __init__(self, message="No model found. Please check the model file path."):
        """Initialise the error with a descriptive default message."""
        super().__init__(message)

class NoConfigError(Exception):
    """Raised when the configuration file is missing or unreadable."""

    def __init__(
        self,
        message="No config file found. Please check the config file path.",
    ):
        """Initialise the error with a descriptive default message."""
        super().__init__(message)

class NoLoggerError(Exception):
    """Raised when logging cannot be initialised."""

    def __init__(self, message="Could not create logger."):
        """Initialise the error with a descriptive default message."""
        super().__init__(message)

# Errors that are caught during the run
class TooSmalldVError(Exception):
    """
    Used when the potential is too flat to be integrated. This means that the
    bounce solution would be an extreme thin wall bubble with low latent heat,
    likely not giving rise to any gravitational waves.
    """
    pass

class InfiniteActionError(Exception):
    """
    This is a hot fix for the infinite action error: When this error is raised,
    the action is actually not infinite, but so large that the numerical
    integration of the bubble profile fails.
    """
    pass

class ZeroActionError(Exception):
    """
    This is a hot fix for the zero action error: When this error is raised,
    the action is actually not zero, but so small that the numerical
    integration of the bubble profile fails.
    """
    pass

class DeformationError(Exception):
    """Raised when the path deformation algorithm fails to converge."""

    def __init__(
        self,
        message=(
            "Deformation doesn't appear to be converging. Stopping at the point of best "
            "convergence."
        ),
    ):
        """Initialise the error with the convergence diagnostic message."""
        super().__init__(message)

class PotentialError(Exception):
    """
    Used when the potential does not have the expected characteristics.

    The error messages should be tuples, with the second item being one of
    ``(\"no barrier\", \"stable, not metastable\")``.
    """

    def __init__(
        self,
        message=(
            "There was an error with the potential. It could be that there is no barrier "
            "for instance."
        ),
        error_type="no barrier",
    ):
        """Store both the human-readable explanation and the machine-readable tag."""
        super().__init__(message)
        self.error_type = error_type

class ParamNamesError(Exception):
    """Raised when the parameter names do not match the model definition."""

    def __init__(self, message="Parameter names do not match the model parameters."):
        """Initialise the error with a descriptive default message."""
        super().__init__(message)

class GWParamsError(Exception):
    """Raised when the gravitational wave parameters are inconsistent or incomplete."""

    def __init__(self, message="The given GW params do not match the required ones."):
        """Initialise the error with a descriptive default message."""
        super().__init__(message)

# Errors that are raise during the run and are
# only caught in the main function together with the
# error code
class TachyonError(Exception):
    """Raised when a tachyonic mass is detected at the zero-temperature minimum."""

    def __init__(
        self,
        message="Tachyonic boson mass at zero temperature minimum.",
    ):
        """Attach the CLI error code identifying tachyonic ground states."""
        super().__init__(message)
        self.errorcode = 1


class NucleationError(Exception):
    """Raised when the nucleation temperature could not be identified."""

    def __init__(self, message="Nucleation temperature not found."):
        """Attach the CLI error code identifying nucleation failures."""
        super().__init__(message)
        self.errorcode = 2


class WrongT0MinimumError(Exception):
    """Raised when the expected T = 0 minimum does not match the last minimum."""

    def __init__(
        self,
        message="Expected T = 0 minimum does not match found last minimum.",
    ):
        """Attach the CLI error code identifying mis-matched zero-temperature minima."""
        super().__init__(message)
        self.errorcode = 3


class NoPhases(Exception):
    """Raised when the phase tracing routine cannot identify any phases or the user
    tries to compute transitions without tracing first."""

    def __init__(self, message=""):
        """Attach the CLI error code identifying missing phases."""
        super().__init__(message)
        self.errorcode = 4


class OnlyOnePhase(Exception):
    """Raised when phase tracing yields only a single phase. In that case,
    no transitions can be computed."""

    def __init__(self, message="Only one phase has been found."):
        """Attach the CLI error code identifying incomplete phase scans."""
        super().__init__(message)
        self.errorcode = 5


class NoTransitionFound(Exception):
    """Raised when no viable transition is found between identified phases."""

    def __init__(self, message="No transition was found."):
        """Attach the CLI error code identifying missing transitions."""
        super().__init__(message)
        self.errorcode = 6


class PercolationApproximation1Error(Exception):
    """Raised when the percolation temperature approximation fails to converge."""

    def __init__(
        self,
        message="Percolation temperature using approximation 1 could not be found.",
    ):
        """Attach the CLI error code identifying percolation-approximation failures."""
        super().__init__(message)
        self.errorcode = 7


class TooMuchSupercoolingError(Exception):
    """Raised when excessive supercooling prevents bubble nucleation within Hubble time."""

    def __init__(
        self,
        message=(
            "No transition found because the high-temperature phase is too supercooled "
            "to allow for the nucleation of one bubble per Hubble volume until today."
        ),
    ):
        """Attach the CLI error code identifying over-supercooled transitions."""
        super().__init__(message)
        self.errorcode = 8


class OnlySecondOrderTransitionsError(Exception):
    """Raised when the scan identifies only second-order transitions."""

    def __init__(
        self,
        message="Only second-order transition(s) found. No first-order transitions.",
    ):
        """Attach the CLI error code identifying missing first-order transitions."""
        super().__init__(message)
        self.errorcode = 9


class PercolationError(Exception):
    """Raised when the percolation temperature could not be determined."""

    def __init__(self, message="Percolation temperature could not be found."):
        """Attach the CLI error code identifying percolation failures."""
        super().__init__(message)
        self.errorcode = 10


class TunnelingError(Exception):
    """Raised when tunnelling fails to produce an action."""

    def __init__(self, message="Tunneling action not found."):
        """Attach the CLI error code identifying tunnelling failures."""
        super().__init__(message)
        self.errorcode = 11

class InitPotentialError(Exception):
    """
    Used when there is a problem creating the potential

    The error message should be something like ``\"potential unbounded from below\"``
    or ``\"unphysical Lagrangian parameters\"``.
    """

    def __init__(
        self,
        message="There was an error with the initialisation of the potential.",
        error_type="unbounded from below",
    ):
        """Attach both the human-readable explanation and a machine-readable tag."""
        super().__init__(message)
        self.error_type = error_type
        self.errorcode = 12


class SplineError(Exception):
    """Raised when constructing a spline along the tunnelling path fails."""

    def __init__(self, message="A spline error occurred."):
        """Attach the CLI error code identifying spline construction failures."""
        super().__init__(message)
        self.errorcode = 13


class WrongHighTPhaseError(Exception):
    """Raised when the high-temperature phase differs from the expected one."""

    def __init__(
        self,
        message="The high-temperature phase does not match the expected one.",
    ):
        """Attach the CLI error code identifying high-temperature phase mismatches."""
        super().__init__(message)
        self.errorcode = 14


class EternalInflationError(Exception):
    """Raised when the eternal-inflation criterion is fulfilled."""

    def __init__(self, message="Eternal inflation criterion is fulfilled."):
        """Attach the CLI error code identifying eternal-inflation issues."""
        super().__init__(message)
        self.errorcode = 15


class Timeout(Exception):
    """Raised when a long-running calculation exceeds the allowed runtime."""

    def __init__(self, message="The calculation timed out."):
        """Attach the CLI error code identifying timeout situations."""
        super().__init__(message)
        self.errorcode = 16


class ActionRateJitterError(Exception):
    """Raised when the sampled bounce action implies a non-smooth nucleation rate."""

    def __init__(
        self,
        message=(
            "Detected a large non-smooth jitter in Gamma/H^4 across the active "
            "percolation band. This indicates an instability of the bounce "
            "computation or a wrong tunnelling branch."
        ),
    ):
        """Attach the CLI error code identifying action/rate smoothness failures."""
        super().__init__(message)
        self.errorcode = 17


class UnexpectedError(Exception):
    """Raised when an unclassified failure occurs in the CLI."""

    def __init__(self, message="An unexpected error occurred."):
        """Attach the catch-all CLI error code."""
        super().__init__(message)
        self.errorcode = 999
