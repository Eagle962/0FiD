# ofd/__init__.py
"""
OFD - Overfitting Detector
"""

__version__ = "0.1.0"

from .adapters import get_adapter, BaseAdapter, SklearnAdapter
from .methods import calculate_performance_gap, detect_overfitting_basic

__all__ = [
    'get_adapter',
    'BaseAdapter', 
    'SklearnAdapter',
    'calculate_performance_gap',
    'detect_overfitting_basic',
]