"""
OFiD Adapters - Base Adapter Module
定義所有適配器的統一接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np


class BaseAdapter(ABC):
    """
    所有模型適配器的抽象基類
    
    設計原則：
    1. 統一接口：所有框架通過相同方法訪問
    2. 最小依賴：只依賴 numpy，各框架在子類中引入
    3. 信息豐富：提供足夠的模型信息給檢測器使用
    """
    
    def __init__(self, model: Any):
        """
        初始化適配器
        
        Args:
            model: 任何機器學習模型對象
        """
        self.model = model
        self._model_type = None
        self._task_type = None  # 'classification' or 'regression'
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseAdapter':
        """
        訓練模型
        
        Args:
            X: 訓練特徵
            y: 訓練標籤
            **kwargs: 框架特定的參數
            
        Returns:
            self: 支持鏈式調用
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Args:
            X: 輸入特徵
            
        Returns:
            predictions: 預測結果
        """
        pass
    
    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        計算模型得分（準確率或 R²）
        
        Args:
            X: 測試特徵
            y: 測試標籤
            
        Returns:
            score: 性能分數（越高越好）
        """
        pass
    
    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """
        獲取模型超參數
        
        Returns:
            params: 超參數字典
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型詳細信息
        
        Returns:
            info: 包含模型類型、任務類型、複雜度等信息
        """
        pass
    
    def get_task_type(self) -> str:
        """
        獲取任務類型
        
        Returns:
            task_type: 'classification' 或 'regression'
        """
        if self._task_type is None:
            self._task_type = self._infer_task_type()
        return self._task_type
    
    def _infer_task_type(self) -> str:
        """
        推斷任務類型（子類可以覆蓋）
        
        Returns:
            task_type: 'classification' 或 'regression'
        """
        # 默認實現，子類應該覆蓋
        return 'unknown'
    
    def is_fitted(self) -> bool:
        """
        檢查模型是否已訓練
        
        Returns:
            fitted: True 如果模型已訓練
        """
        # 默認實現，子類可以覆蓋
        return hasattr(self.model, 'predict')
    
    def get_complexity_metrics(self) -> Dict[str, Any]:
        """
        獲取模型複雜度指標
        
        Returns:
            metrics: 複雜度相關指標（參數數量、深度等）
        """
        # 默認返回基礎信息，子類可以擴展
        return {
            'model_class': self.model.__class__.__name__,
            'framework': self._model_type
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model.__class__.__name__})"


class AdapterFactory:
    """
    適配器工廠類
    自動根據模型類型創建對應的適配器
    """
    
    _adapters = {}
    
    @classmethod
    def register(cls, model_type: str, adapter_class: type):
        """
        註冊適配器
        
        Args:
            model_type: 模型類型標識（如 'sklearn'）
            adapter_class: 適配器類
        """
        cls._adapters[model_type] = adapter_class
    
    @classmethod
    def create(cls, model: Any) -> BaseAdapter:
        """
        自動創建適配器
        
        Args:
            model: 模型對象
            
        Returns:
            adapter: 對應的適配器實例
            
        Raises:
            ValueError: 如果無法識別模型類型
        """
        model_type = cls._detect_model_type(model)
        
        if model_type not in cls._adapters:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Supported types: {list(cls._adapters.keys())}"
            )
        
        adapter_class = cls._adapters[model_type]
        return adapter_class(model)
    
    @staticmethod
    def _detect_model_type(model: Any) -> str:
        """
        檢測模型類型
        
        Args:
            model: 模型對象
            
        Returns:
            model_type: 模型類型字符串
        """
        module_name = model.__class__.__module__
        
        if 'sklearn' in module_name:
            return 'sklearn'
        elif 'torch' in module_name:
            return 'pytorch'
        elif 'tensorflow' in module_name or 'keras' in module_name:
            return 'tensorflow'
        elif 'xgboost' in module_name:
            return 'xgboost'
        elif 'lightgbm' in module_name:
            return 'lightgbm'
        else:
            return 'unknown'
    
    @classmethod
    def list_supported_frameworks(cls) -> list:
        """
        列出所有支持的框架
        
        Returns:
            frameworks: 框架名稱列表
        """
        return list(cls._adapters.keys())