"""Dataset conversion and benchmark-specific evaluation."""

from importlib import import_module

from harness.common import HarnessError, identifier


def get_evaluator(name: str):
    """Adapters expose evaluate_predictions; core logic knows no benchmark names."""
    identifier(name, "evaluator")
    try:
        module = import_module(f"adapters.{name}")
    except ModuleNotFoundError as exc:
        if exc.name != f"adapters.{name}":
            raise
        raise HarnessError(f"Unknown evaluator adapter: {name}") from exc
    function = getattr(module, "evaluate_predictions", None)
    if not callable(function):
        raise HarnessError(f"Adapter {name!r} does not provide evaluate_predictions")
    return function
