"""
OFiD Adapters - Scikit-learn Adapter
支持所有 Scikit-learn 模型
"""

from typing import Any, Dict
import numpy as np
from .base import BaseAdapter, AdapterFactory


class SklearnAdapter(BaseAdapter):
    """
    Scikit-learn 模型適配器
    
    支持：
    - 分類器 (ClassifierMixin)
    - 回歸器 (RegressorMixin)
    - 所有 sklearn estimators
    """
    
    def __init__(self, model: Any):
        super().__init__(model)
        self._model_type = 'sklearn'
        self._validate_model()
    
    def _validate_model(self):
        """驗證模型是否為有效的 sklearn 模型"""
        if not hasattr(self.model, 'fit'):
            raise ValueError(
                f"Model {self.model.__class__.__name__} does not have 'fit' method. "
                "Not a valid sklearn estimator."
            )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SklearnAdapter':
        """
        訓練模型
        
        Args:
            X: 訓練特徵
            y: 訓練標籤
            **kwargs: sklearn 特定參數（如 sample_weight）
            
        Returns:
            self: 支持鏈式調用
        """
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        預測
        
        Args:
            X: 輸入特徵
            
        Returns:
            predictions: 預測結果
        """
        if not self.is_fitted():
            raise RuntimeError("Model must be fitted before prediction")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        預測概率（僅分類器）
        
        Args:
            X: 輸入特徵
            
        Returns:
            probabilities: 預測概率
        """
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(
                f"Model {self.model.__class__.__name__} does not support predict_proba"
            )
        return self.model.predict_proba(X)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        計算模型得分
        
        對於分類器：返回準確率
        對於回歸器：返回 R² score
        
        Args:
            X: 測試特徵
            y: 測試標籤
            
        Returns:
            score: 性能分數
        """
        if not self.is_fitted():
            raise RuntimeError("Model must be fitted before scoring")
        return self.model.score(X, y)
    
    def get_params(self) -> Dict[str, Any]:
        """
        獲取模型超參數
        
        Returns:
            params: 超參數字典
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params(deep=False)
        return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        獲取模型詳細信息
        
        Returns:
            info: 模型信息字典
        """
        info = {
            'framework': 'sklearn',
            'model_class': self.model.__class__.__name__,
            'model_module': self.model.__class__.__module__,
            'task_type': self.get_task_type(),
            'is_fitted': self.is_fitted(),
            'params': self.get_params()
        }
        
        # 添加複雜度信息
        info.update(self.get_complexity_metrics())
        
        return info
    
    def _infer_task_type(self) -> str:
        """
        推斷任務類型
        
        Returns:
            task_type: 'classification' 或 'regression'
        """
        from sklearn.base import ClassifierMixin, RegressorMixin
        
        if isinstance(self.model, ClassifierMixin):
            return 'classification'
        elif isinstance(self.model, RegressorMixin):
            return 'regression'
        else:
            # 嘗試從方法推斷
            if hasattr(self.model, 'predict_proba'):
                return 'classification'
            return 'unknown'
    
    def is_fitted(self) -> bool:
        """
        檢查模型是否已訓練
        
        Returns:
            fitted: True 如果模型已訓練
        """
        # sklearn 的標準檢查方式
        from sklearn.utils.validation import check_is_fitted
        try:
            check_is_fitted(self.model)
            return True
        except:
            return False
    
    def get_complexity_metrics(self) -> Dict[str, Any]:
        """
        獲取模型複雜度指標
        
        Returns:
            metrics: 複雜度指標字典
        """
        metrics = super().get_complexity_metrics()
        params = self.get_params()
        
        # 決策樹類模型
        if 'max_depth' in params:
            metrics['max_depth'] = params.get('max_depth')
            metrics['min_samples_leaf'] = params.get('min_samples_leaf')
            metrics['min_samples_split'] = params.get('min_samples_split')
        
        # 隨機森林類模型
        if 'n_estimators' in params:
            metrics['n_estimators'] = params.get('n_estimators')
        
        # 線性模型
        if hasattr(self.model, 'coef_'):
            metrics['n_features'] = self.model.coef_.shape[-1]
            if hasattr(self.model, 'alpha'):
                metrics['regularization'] = self.model.alpha
        
        # 神經網路（MLPClassifier/MLPRegressor）
        if 'hidden_layer_sizes' in params:
            metrics['hidden_layer_sizes'] = params['hidden_layer_sizes']
            metrics['n_layers'] = len(params['hidden_layer_sizes']) + 1
        
        # SVM
        if 'kernel' in params:
            metrics['kernel'] = params['kernel']
            metrics['C'] = params.get('C')
        
        return metrics
    
    def get_feature_importance(self) -> np.ndarray:
        """
        獲取特徵重要性（如果可用）
        
        Returns:
            importance: 特徵重要性數組
            
        Raises:
            AttributeError: 如果模型不支持特徵重要性
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # 線性模型使用係數的絕對值
            return np.abs(self.model.coef_).flatten()
        else:
            raise AttributeError(
                f"Model {self.model.__class__.__name__} does not provide feature importance"
            )


# 自動註冊 Scikit-learn 適配器
AdapterFactory.register('sklearn', SklearnAdapter)