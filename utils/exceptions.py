# utils/exceptions.py
# Central place for small custom exceptions used across the codebase.

class GrowthExplosionError(RuntimeError):
    """
    Raised when a scenario diverges numerically (e.g. infinite growth in finite
    time). The runner catches it and aborts the batch early.
    """
    pass
