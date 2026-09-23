"""Compatibility alias; implementation lives in vmr.vlm.client."""
import sys
import vmr.vlm.client as _implementation

sys.modules[__name__] = _implementation
