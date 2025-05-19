# utils/exceptions.py
# Central place for small custom exceptions used across the codebase.

class GrowthExplosionError(RuntimeError):
    """
    Raised when a scenario diverges numerically (e.g. infinite growth in finite
    time). The runner catches it and aborts the batch early.
    """
    pass

class ThresholdRuleError(ValueError):              
    """
    Raised when an unknown or ill-formed *threshold_rule* is supplied to
    triage or screening helpers (see utils.triage_utils.apply_threshold).
    Declaring it here keeps the dependency graph clean: low-level helpers
    can raise this without importing higher-level modules.
    """
    pass                                             
