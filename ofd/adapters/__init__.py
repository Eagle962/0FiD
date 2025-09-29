"""
OFiD Adapters Package
模型適配器模塊
"""

from .base import BaseAdapter, AdapterFactory
from .sklearn_adapter import SklearnAdapter

__all__ = [
    'BaseAdapter',
    'AdapterFactory',
    'SklearnAdapter',
]


def get_adapter(model):
    """
    To shrub 反正這個就是你要用模型的時候用這個函數
    就是取得模型資訊的東西 你不需要知道是哪種模型 因為我做完處理了
    然後底下有你可以用這個取得的資訊
    再來就是 你可能會需要分兩種狀況 就是輸入的模型有沒有被訓練過
    你可以用is_fitted()檢測 因為可能用戶只給模型配置
    到時候使用他會需要輸入資料集和訓練集
    
    
    
    Args:
        model: 任何機器學習模型
        
    Returns:
        adapter: 對應的適配器實例
        
    Example:
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> model = RandomForestClassifier()
        >>> adapter = get_adapter(model)
        >>> print(adapter)
        SklearnAdapter(model=RandomForestClassifier)
    """
    return AdapterFactory.create(model)

"""
Adapter 可用方法

| 方法                          | 返回類型      | 用途                              |
|-------------------------------|--------------|-----------------------------------|
| `.get_task_type()`            | `str`        | 'classification' 或 'regression'  |
| `.is_fitted()`                | `bool`       | 是否已訓練                         |
| `.get_params()`               | `dict`       | 所有超參數                         |
| `.get_model_info()`           | `dict`       | 完整模型資訊                       |
| `.get_complexity_metrics()`   | `dict`       | 複雜度相關指標                     |
| `.predict(X)`                 | `ndarray`    | 預測結果                          |
| `.predict_proba(X)`           | `ndarray`    | 預測概率（僅分類器）               |
| `.score(X, y)`                | `float`      | 性能分數（準確率或 R²）            |
| `.get_feature_importance()`   | `ndarray`    | 特徵重要性（如果支持）             |
| `.model`                      | `object`     | 訪問原始模型對象                   |
| `.fit(X, y)`                  | `self`       | 訓練模型                          |

Example:
    >>> adapter = get_adapter(model)
    >>> adapter.fit(X_train, y_train)
    >>> score = adapter.score(X_val, y_val)
"""