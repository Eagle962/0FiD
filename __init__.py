# ofd/__init__.py
"""
OFD - Overfitting Detector
"""

__version__ = "0.1.0"

from .adapters import get_adapter, BaseAdapter, SklearnAdapter

__all__ = [
    'get_adapter',
    'BaseAdapter', 
    'SklearnAdapter',
]