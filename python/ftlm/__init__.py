"""Python helpers for ftlm_extended_hubbard (bindings are optional)."""

try:
    from ._ftlm_core import HubbardParams, nsites, version_string
except ImportError:  # extension not built yet
    HubbardParams = None  # type: ignore[misc, assignment]
    nsites = None  # type: ignore[misc, assignment]
    version_string = None  # type: ignore[misc, assignment]

__all__ = ["HubbardParams", "nsites", "version_string"]
