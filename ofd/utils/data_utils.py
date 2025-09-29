"""
OFi Utils - Data Utilities
數據驗證和預處理工具
"""

from typing import Tuple, Optional, Union
import numpy as np


class DataValidator:
    """
    數據驗證器
    確保輸入數據格式正確，適合模型訓練和檢測
    """
    
    @staticmethod
    def validate_X_y(X: np.ndarray, y: np.ndarray, 
                     allow_none: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        驗證 X 和 y 的格式
        
        Args:
            X: 特徵矩陣
            y: 標籤向量
            allow_none: 是否允許 None 值
            
        Returns:
            X, y: 驗證後的數據
            
        Raises:
            ValueError: 如果數據格式不正確
        """
        if X is None or y is None:
            if allow_none:
                return X, y
            raise ValueError("X and y cannot be None")
        
        # 轉換為 numpy 數組
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 檢查維度
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got {X.ndim}D")
        
        if y.ndim not in [1, 2]:
            raise ValueError(f"y must be 1D or 2D array, got {y.ndim}D")
        
        # 檢查樣本數量一致
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {len(X)}, y: {len(y)}"
            )
        
        # 檢查是否有 NaN
        if np.any(np.isnan(X)):
            raise ValueError("X contains NaN values")
        
        if np.any(np.isnan(y)):
            raise ValueError("y contains NaN values")
        
        # 檢查樣本數量
        if len(X) == 0:
            raise ValueError("X and y cannot be empty")
        
        return X, y
    
    @staticmethod
    def validate_train_val_split(X_train: np.ndarray, X_val: np.ndarray,
                                 y_train: np.ndarray, y_val: np.ndarray) -> None:
        """
        驗證訓練集和驗證集
        
        Args:
            X_train: 訓練特徵
            X_val: 驗證特徵
            y_train: 訓練標籤
            y_val: 驗證標籤
            
        Raises:
            ValueError: 如果數據格式不正確
        """
        # 驗證各自格式
        X_train, y_train = DataValidator.validate_X_y(X_train, y_train)
        X_val, y_val = DataValidator.validate_X_y(X_val, y_val)
        
        # 檢查特徵數量一致
        if X_train.shape[1] != X_val.shape[1]:
            raise ValueError(
                f"X_train and X_val must have same number of features. "
                f"Got X_train: {X_train.shape[1]}, X_val: {X_val.shape[1]}"
            )
        
        # 檢查驗證集不能太小
        min_val_samples = 10
        if len(X_val) < min_val_samples:
            raise ValueError(
                f"Validation set too small. "
                f"Got {len(X_val)} samples, need at least {min_val_samples}"
            )
        
        # 檢查訓練集不能太小
        min_train_samples = 20
        if len(X_train) < min_train_samples:
            raise ValueError(
                f"Training set too small. "
                f"Got {len(X_train)} samples, need at least {min_train_samples}"
            )
    
    @staticmethod
    def check_data_distribution(y_train: np.ndarray, y_val: np.ndarray,
                               task_type: str = 'classification') -> dict:
        """
        檢查訓練集和驗證集的分佈
        
        Args:
            y_train: 訓練標籤
            y_val: 驗證標籤
            task_type: 任務類型
            
        Returns:
            info: 分佈信息字典
        """
        info = {}
        
        if task_type == 'classification':
            # 分類任務：檢查類別分佈
            train_classes, train_counts = np.unique(y_train, return_counts=True)
            val_classes, val_counts = np.unique(y_val, return_counts=True)
            
            info['train_classes'] = train_classes.tolist()
            info['val_classes'] = val_classes.tolist()
            info['train_class_counts'] = dict(zip(train_classes, train_counts))
            info['val_class_counts'] = dict(zip(val_classes, val_counts))
            
            # 檢查類別不平衡
            train_imbalance = max(train_counts) / min(train_counts) if len(train_counts) > 1 else 1.0
            val_imbalance = max(val_counts) / min(val_counts) if len(val_counts) > 1 else 1.0
            
            info['train_imbalance_ratio'] = float(train_imbalance)
            info['val_imbalance_ratio'] = float(val_imbalance)
            
            # 警告：驗證集缺少某些類別
            missing_classes = set(train_classes) - set(val_classes)
            if missing_classes:
                info['warning'] = f"Validation set missing classes: {missing_classes}"
        
        else:
            # 回歸任務：檢查數值分佈
            info['train_mean'] = float(np.mean(y_train))
            info['train_std'] = float(np.std(y_train))
            info['train_min'] = float(np.min(y_train))
            info['train_max'] = float(np.max(y_train))
            
            info['val_mean'] = float(np.mean(y_val))
            info['val_std'] = float(np.std(y_val))
            info['val_min'] = float(np.min(y_val))
            info['val_max'] = float(np.max(y_val))
            
            # 檢查分佈差異
            mean_diff = abs(info['train_mean'] - info['val_mean'])
            std_diff = abs(info['train_std'] - info['val_std'])
            
            if mean_diff > info['train_std'] * 0.5:
                info['warning'] = "Train and validation sets have different mean values"
        
        return info


class DataSplitter:
    """
    數據分割工具
    """
    
    @staticmethod
    def train_val_split(X: np.ndarray, y: np.ndarray,
                       val_size: float = 0.2,
                       random_state: Optional[int] = None,
                       stratify: bool = True) -> Tuple[np.ndarray, np.ndarray, 
                                                        np.ndarray, np.ndarray]:
        """
        分割訓練集和驗證集
        
        Args:
            X: 特徵矩陣
            y: 標籤向量
            val_size: 驗證集比例
            random_state: 隨機種子
            stratify: 是否分層抽樣（分類任務推薦）
            
        Returns:
            X_train, X_val, y_train, y_val: 分割後的數據
        """
        # 驗證數據
        X, y = DataValidator.validate_X_y(X, y)
        
        # 檢查驗證集大小
        if not 0 < val_size < 1:
            raise ValueError(f"val_size must be between 0 and 1, got {val_size}")
        
        n_samples = len(X)
        n_val = int(n_samples * val_size)
        
        if n_val < 10:
            raise ValueError(
                f"Validation set would be too small ({n_val} samples). "
                f"Need at least 10 samples."
            )
        
        # 設置隨機種子
        if random_state is not None:
            np.random.seed(random_state)
        
        # 生成索引
        indices = np.arange(n_samples)
        
        if stratify:
            # 分層抽樣（保持類別比例）
            from collections import defaultdict
            class_indices = defaultdict(list)
            
            for idx, label in enumerate(y):
                class_indices[label].append(idx)
            
            train_idx = []
            val_idx = []
            
            for label, idx_list in class_indices.items():
                n_class_val = max(1, int(len(idx_list) * val_size))
                np.random.shuffle(idx_list)
                val_idx.extend(idx_list[:n_class_val])
                train_idx.extend(idx_list[n_class_val:])
        else:
            # 隨機抽樣
            np.random.shuffle(indices)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        return X_train, X_val, y_train, y_val
    
    @staticmethod
    def create_learning_curve_splits(X: np.ndarray, y: np.ndarray,
                                    train_sizes: list = None) -> list:
        """
        為學習曲線創建不同大小的訓練集
        
        Args:
            X: 特徵矩陣
            y: 標籤向量
            train_sizes: 訓練集大小列表（可以是比例或絕對數量）
            
        Returns:
            splits: [(X_train, y_train), ...] 列表
        """
        if train_sizes is None:
            train_sizes = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        X, y = DataValidator.validate_X_y(X, y)
        n_samples = len(X)
        
        splits = []
        for size in train_sizes:
            if size <= 1.0:
                # 比例
                n_train = int(n_samples * size)
            else:
                # 絕對數量
                n_train = int(size)
            
            n_train = min(n_train, n_samples)
            n_train = max(n_train, 10)  # 至少 10 個樣本
            
            indices = np.random.choice(n_samples, n_train, replace=False)
            splits.append((X[indices], y[indices]))
        
        return splits


def get_dataset_info(X: np.ndarray, y: np.ndarray) -> dict:
    """
    獲取數據集基本信息
    
    Args:
        X: 特徵矩陣
        y: 標籤向量
        
    Returns:
        info: 數據集信息字典
    """
    X, y = DataValidator.validate_X_y(X, y)
    
    info = {
        'n_samples': len(X),
        'n_features': X.shape[1],
        'feature_names': [f'feature_{i}' for i in range(X.shape[1])],
        'X_dtype': str(X.dtype),
        'y_dtype': str(y.dtype),
    }
    
    # 檢查是否為分類任務
    unique_y = np.unique(y)
    if len(unique_y) < 20 and y.dtype in [np.int32, np.int64, object]:
        info['task_type'] = 'classification'
        info['n_classes'] = len(unique_y)
        info['classes'] = unique_y.tolist()
    else:
        info['task_type'] = 'regression'
        info['y_range'] = [float(np.min(y)), float(np.max(y))]
    
    return info