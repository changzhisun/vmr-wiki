"""Compatibility alias for vmr.evaluation.metrics."""
import sys
import vmr.evaluation.metrics as _implementation
sys.modules[__name__] = _implementation
