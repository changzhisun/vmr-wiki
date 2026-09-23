"""Explicit registry; imports backends only when compilation is requested."""

from vmr.core.errors import HarnessError

COMPILERS = {}


def register(compiler):
    if compiler.name in COMPILERS:
        raise HarnessError(f"Compiler already registered: {compiler.name}")
    COMPILERS[compiler.name] = compiler


def _builtins():
    from .methods.simple.compiler import SimpleCompiler
    from .methods.dense.compiler import DenseCompiler
    from .methods.hierarchical.compiler import HierarchicalCompiler
    from .methods.bidirectional.compiler import BidirectionalCompiler
    from .methods.agentic.compiler import AgenticCompiler

    for cls in (
        SimpleCompiler,
        DenseCompiler,
        HierarchicalCompiler,
        BidirectionalCompiler,
        AgenticCompiler,
    ):
        COMPILERS.setdefault(cls.name, cls())


def get_compiler(name):
    _builtins()
    try:
        return COMPILERS[name]
    except KeyError as exc:
        raise HarnessError(f"Unknown compiler: {name}") from exc
