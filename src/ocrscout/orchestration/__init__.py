"""Runner orchestration: provisioning + model-major chunking, kept separate
from stage execution.

This package ``__init__`` is intentionally import-light: the ``autoscale``
decisions are a leaf the ``pipeline`` layer depends on, so eagerly importing
``Provisioner`` here (which imports ``pipeline``) would create a cycle. Import
``Provisioner`` / ``RunnerPlan`` from their submodules.
"""
