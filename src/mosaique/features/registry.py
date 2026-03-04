"""Feature registry for built-in feature functions.

The ``@register_feature`` decorator stores metadata about each feature
(compatible transforms, the function itself) in a global registry.  This
registry is used by:

- The config loader to validate that a feature is wired to a compatible
  transform type.
- The test suite to auto-discover features for parametrized edge-case tests.
"""

import inspect
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureEntry:
    """Metadata about a registered feature function."""

    func: Any
    transforms: frozenset[str]


# Global registry: function name -> FeatureEntry
FEATURE_REGISTRY: dict[str, FeatureEntry] = {}


def register_feature(
    transform: str | list[str],
) -> Any:
    """Decorator to register a feature function.

    Parameters
    ----------
    transform : str or list[str]
        Transform type(s) this feature is compatible with.
        One of ``"simple"``, ``"tf_decomposition"``, ``"connectivity"``.

    Raises
    ------
    TypeError
        If the function does not accept ``**kwargs``.

    Example
    -------
    ::

        @register_feature(transform="simple")
        def my_feature(x, sfreq=200, **kwargs):
            return float(np.mean(x))
    """
    if isinstance(transform, str):
        transforms = frozenset([transform])
    else:
        transforms = frozenset(transform)

    def decorator(func: Any) -> Any:
        sig = inspect.signature(func)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            raise TypeError(
                f"{func.__name__!r} must accept **kwargs to be registered "
                f"as a feature function"
            )

        FEATURE_REGISTRY[func.__name__] = FeatureEntry(
            func=func,
            transforms=transforms,
        )
        return func

    return decorator
