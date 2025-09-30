"""
OFiD Methods Package
過擬合檢測方法模塊
"""

from .performance_gap import calculate_performance_gap, detect_overfitting_basic

__all__ = [
    'calculate_performance_gap',
    'detect_overfitting_basic',
]
