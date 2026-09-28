"""Operation adapters enrich actual execution; they never replace PyTorch."""

from .registry import describe_operation

__all__ = ["describe_operation"]
