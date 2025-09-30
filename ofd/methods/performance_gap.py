"""
OFiD Methods - Performance Gap Calculator
計算訓練-驗證性能差距的核心方法
"""

from typing import Dict, Tuple, Optional, List
import numpy as np
from ..adapters.base import BaseAdapter
from ..utils.data_utils import DataValidator


def calculate_performance_gap(adapter: BaseAdapter,
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_val: np.ndarray, 
                            y_val: np.ndarray,
                            metrics: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
    """
    計算訓練-驗證性能差距的主要函數
    
    Args:
        adapter: 模型適配器
        X_train: 訓練特徵
        y_train: 訓練標籤
        X_val: 驗證特徵
        y_val: 驗證標籤
        metrics: 要計算的指標列表，None則使用默認指標
        
    Returns:
        results: 包含各指標訓練分數、驗證分數和差距的字典
        
    Example:
        >>> from ofd.adapters import get_adapter
        >>> from ofd.methods.performance_gap import calculate_performance_gap
        >>> 
        >>> model = RandomForestClassifier()
        >>> adapter = get_adapter(model)
        >>> adapter.fit(X_train, y_train)
        >>> 
        >>> gap_results = calculate_performance_gap(
        ...     adapter, X_train, y_train, X_val, y_val, ['accuracy', 'f1']
        ... )
        >>> print(f"Accuracy gap: {gap_results['accuracy']['gap']:.4f}")
    """
    # 數據驗證
    X_train, y_train = DataValidator.validate_X_y(X_train, y_train)
    X_val, y_val = DataValidator.validate_X_y(X_val, y_val)
    DataValidator.validate_train_val_split(X_train, X_val, y_train, y_val)
    
    # 確保模型已訓練
    if not adapter.is_fitted():
        raise RuntimeError("Model must be fitted before calculating performance gap")
    
    # 獲取任務類型
    task_type = adapter.get_task_type()
    
    # 選擇要計算的指標
    if metrics is None:
        metrics = _get_default_metrics(task_type)
    
    # 驗證指標
    _validate_metrics(metrics, task_type)
    
    # 計算各指標的分數
    results = {}
    for metric in metrics:
        train_score = _calculate_metric(adapter, X_train, y_train, metric, task_type)
        val_score = _calculate_metric(adapter, X_val, y_val, metric, task_type)
        
        # 計算差距
        gap = _calculate_gap_value(train_score, val_score, metric)
        
        results[metric] = {
            'train_score': train_score,
            'val_score': val_score,
            'gap': gap,
            'gap_absolute': abs(gap),
            'gap_percentage': _calculate_percentage_gap(train_score, val_score, metric)
        }
    
    return results


def detect_overfitting_basic(adapter: BaseAdapter,
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray,
                           gap_threshold: float = 0.1,
                           metric: str = None) -> Dict[str, any]:
    """
    基礎過擬合檢測函數
    
    Args:
        adapter: 模型適配器
        X_train: 訓練特徵
        y_train: 訓練標籤
        X_val: 驗證特徵
        y_val: 驗證標籤
        gap_threshold: 性能差距閾值（超過此值認為可能過擬合）
        metric: 用於檢測的指標，None則使用默認指標
        
    Returns:
        result: 包含檢測結果的字典
        
    Example:
        >>> from ofd.methods.performance_gap import detect_overfitting_basic
        >>> 
        >>> result = detect_overfitting_basic(
        ...     adapter, X_train, y_train, X_val, y_val, gap_threshold=0.05
        ... )
        >>> print(f"Is overfitting: {result['is_overfitting']}")
        >>> print(f"Gap: {result['gap']:.4f}")
    """
    # 獲取任務類型
    task_type = adapter.get_task_type()
    
    # 選擇檢測指標
    if metric is None:
        metric = 'accuracy' if task_type == 'classification' else 'r2'
    
    # 計算性能差距
    gap_results = calculate_performance_gap(adapter, X_train, y_train, X_val, y_val, [metric])
    gap_info = gap_results[metric]
    
    # 判斷是否過擬合
    is_overfitting = gap_info['gap_absolute'] > gap_threshold
    
    # 計算嚴重程度
    severity = _calculate_overfitting_severity(gap_info['gap_absolute'], gap_threshold)
    
    return {
        'is_overfitting': is_overfitting,
        'metric': metric,
        'gap': gap_info['gap'],
        'gap_absolute': gap_info['gap_absolute'],
        'gap_percentage': gap_info['gap_percentage'],
        'train_score': gap_info['train_score'],
        'val_score': gap_info['val_score'],
        'threshold': gap_threshold,
        'severity': severity,
        'recommendation': _get_overfitting_recommendation(severity)
    }


# 輔助函數
def _get_default_metrics(task_type: str) -> List[str]:
    """獲取默認指標"""
    if task_type == 'classification':
        return ['accuracy']
    elif task_type == 'regression':
        return ['r2']
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def _validate_metrics(metrics: List[str], task_type: str) -> None:
    """驗證指標是否支持"""
    supported_metrics = {
        'classification': ['accuracy', 'precision', 'recall', 'f1', 'auc'],
        'regression': ['mse', 'rmse', 'mae', 'r2', 'mape']
    }
    
    valid_metrics = supported_metrics.get(task_type, [])
    invalid_metrics = [m for m in metrics if m not in valid_metrics]
    
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metrics for {task_type}: {invalid_metrics}. "
            f"Supported metrics: {valid_metrics}"
        )


def _calculate_metric(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray, metric: str, task_type: str) -> float:
    """計算單個指標"""
    if metric == 'accuracy':
        return _calculate_accuracy(adapter, X, y)
    elif metric == 'precision':
        return _calculate_precision(adapter, X, y)
    elif metric == 'recall':
        return _calculate_recall(adapter, X, y)
    elif metric == 'f1':
        return _calculate_f1(adapter, X, y)
    elif metric == 'auc':
        return _calculate_auc(adapter, X, y)
    elif metric == 'mse':
        return _calculate_mse(adapter, X, y)
    elif metric == 'rmse':
        return _calculate_rmse(adapter, X, y)
    elif metric == 'mae':
        return _calculate_mae(adapter, X, y)
    elif metric == 'r2':
        return adapter.score(X, y)  # sklearn 的 score 方法
    elif metric == 'mape':
        return _calculate_mape(adapter, X, y)
    else:
        raise ValueError(f"Unknown metric: {metric}")


def _calculate_gap_value(train_score: float, val_score: float, metric: str) -> float:
    """
    計算差距值
    
    對於準確率、R²等越高越好的指標：train_score - val_score
    對於MSE、MAE等越低越好的指標：val_score - train_score
    """
    higher_is_better = metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'r2']
    
    if higher_is_better:
        return train_score - val_score
    else:
        return val_score - train_score


def _calculate_percentage_gap(train_score: float, val_score: float, metric: str) -> float:
    """計算百分比差距"""
    higher_is_better = metric in ['accuracy', 'precision', 'recall', 'f1', 'auc', 'r2']
    
    if higher_is_better:
        if train_score == 0:
            return float('inf') if val_score != 0 else 0.0
        return (train_score - val_score) / train_score * 100
    else:
        if val_score == 0:
            return float('inf') if train_score != 0 else 0.0
        return (val_score - train_score) / val_score * 100


def _calculate_overfitting_severity(gap_absolute: float, threshold: float) -> str:
    """計算過擬合嚴重程度"""
    if gap_absolute <= threshold:
        return 'none'
    elif gap_absolute <= threshold * 2:
        return 'mild'
    elif gap_absolute <= threshold * 3:
        return 'moderate'
    else:
        return 'severe'


def _get_overfitting_recommendation(severity: str) -> str:
    """獲取過擬合處理建議"""
    recommendations = {
        'none': '模型泛化能力良好，無需調整',
        'mild': '建議增加正則化或減少模型複雜度',
        'moderate': '建議增加正則化、減少模型複雜度或增加訓練數據',
        'severe': '建議大幅調整模型架構、增加正則化或收集更多訓練數據'
    }
    return recommendations.get(severity, '未知嚴重程度')


# 分類指標計算方法
def _calculate_accuracy(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算準確率"""
    predictions = adapter.predict(X)
    return np.mean(predictions == y)


def _calculate_precision(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算精確率（宏平均）"""
    predictions = adapter.predict(X)
    unique_labels = np.unique(y)
    precisions = []
    
    for label in unique_labels:
        tp = np.sum((predictions == label) & (y == label))
        fp = np.sum((predictions == label) & (y != label))
        if tp + fp > 0:
            precisions.append(tp / (tp + fp))
        else:
            precisions.append(0.0)
    
    return np.mean(precisions)


def _calculate_recall(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算召回率（宏平均）"""
    predictions = adapter.predict(X)
    unique_labels = np.unique(y)
    recalls = []
    
    for label in unique_labels:
        tp = np.sum((predictions == label) & (y == label))
        fn = np.sum((predictions != label) & (y == label))
        if tp + fn > 0:
            recalls.append(tp / (tp + fn))
        else:
            recalls.append(0.0)
    
    return np.mean(recalls)


def _calculate_f1(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算F1分數（宏平均）"""
    precision = _calculate_precision(adapter, X, y)
    recall = _calculate_recall(adapter, X, y)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def _calculate_auc(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算AUC（僅支持二分類）"""
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError("AUC calculation only supports binary classification")
    
    if not hasattr(adapter.model, 'predict_proba'):
        raise ValueError("Model must support predict_proba for AUC calculation")
    
    try:
        from sklearn.metrics import roc_auc_score
        probabilities = adapter.predict_proba(X)
        # 使用正類的概率
        return roc_auc_score(y, probabilities[:, 1])
    except ImportError:
        raise ImportError("scikit-learn is required for AUC calculation")


# 回歸指標計算方法
def _calculate_mse(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算均方誤差"""
    predictions = adapter.predict(X)
    return np.mean((y - predictions) ** 2)


def _calculate_rmse(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算均方根誤差"""
    return np.sqrt(_calculate_mse(adapter, X, y))


def _calculate_mae(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算平均絕對誤差"""
    predictions = adapter.predict(X)
    return np.mean(np.abs(y - predictions))


def _calculate_mape(adapter: BaseAdapter, X: np.ndarray, y: np.ndarray) -> float:
    """計算平均絕對百分比誤差"""
    predictions = adapter.predict(X)
    # 避免除零錯誤
    mask = y != 0
    if np.sum(mask) == 0:
        return float('inf')
    
    return np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
