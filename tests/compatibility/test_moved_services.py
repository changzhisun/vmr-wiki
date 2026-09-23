"""Legacy imports must share the real service module, including mutable state."""

import importlib

import pytest


@pytest.mark.parametrize(
    "legacy,current",
    [
        ("harness.progress", "vmr.core.progress"),
        ("harness.sampling", "vmr.media.sampling"),
        ("harness.vlm", "vmr.vlm.client"),
        ("harness.vlm_transport", "vmr.vlm.transport"),
        ("harness.ingest", "vmr.compat.ingest"),
    ],
)
def test_legacy_service_module_identity(legacy, current):
    assert importlib.import_module(legacy) is importlib.import_module(current)
