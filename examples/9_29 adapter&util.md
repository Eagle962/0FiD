## Adapter 核心方法

| 函數名稱 | 返回類型 | 用途 |
|---------|---------|------|
| `get_adapter(model)` | `Adapter` | 自動創建適配器 |
| `adapter.fit(X, y)` | `self` | 訓練模型 |
| `adapter.predict(X)` | `ndarray` | 預測結果 |
| `adapter.score(X, y)` | `float` | 計算性能分數（準確率或R²） |
| `adapter.is_fitted()` | `bool` | 檢查是否已訓練 |
| `adapter.get_task_type()` | `str` | 獲取任務類型（'classification' 或 'regression'） |
| `adapter.get_params()` | `dict` | 獲取所有超參數 |
| `adapter.get_model_info()` | `dict` | 獲取完整模型信息 |
| `adapter.get_complexity_metrics()` | `dict` | 獲取複雜度指標（max_depth, n_estimators 等） |

## 數據工具方法

| 函數名稱 | 返回類型 | 用途 |
|---------|---------|------|
| `DataValidator.validate_X_y(X, y)` | `tuple` | 驗證數據格式 |
| `DataValidator.validate_train_val_split(...)` | `None` | 驗證訓練-驗證集 |
| `DataValidator.check_data_distribution(...)` | `dict` | 檢查數據分佈 |
| `DataSplitter.train_val_split(X, y, ...)` | `tuple` | 分割訓練-驗證集 |
| `DataSplitter.create_learning_curve_splits(...)` | `list` | 創建學習曲線數據 |
| `get_dataset_info(X, y)` | `dict` | 獲取數據集基本信息 |