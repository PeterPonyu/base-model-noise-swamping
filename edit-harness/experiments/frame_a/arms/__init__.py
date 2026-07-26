"""arms/ — one adapter per knowledge-maintenance mechanism behind a common Arm interface.

DRYRUN paths are synthetic (no torch/GPU/network); real backends are lazy-imported and are
NEVER reached in a build-only pass. See base.Arm for the contract.
"""
from .base import Arm, ArmOutcome, make_arm, ALL_ARMS  # noqa: F401
