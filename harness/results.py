"""Compatibility alias for vmr.evaluation.results."""
import sys
import vmr.evaluation.results as _implementation
sys.modules[__name__] = _implementation
