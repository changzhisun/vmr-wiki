"""Compatibility alias; implementation lives in vmr.media.sampling."""
import sys
import vmr.media.sampling as _implementation

sys.modules[__name__] = _implementation
