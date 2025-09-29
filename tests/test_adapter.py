"""
逐步測試 - 一步一步確認每個部分
放在 tests/step_by_step_test.py
"""

print("步驟 1: 測試 Python 路徑")
print("-" * 50)
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(f"✅ 當前工作目錄: {os.getcwd()}")
print(f"✅ Python 路徑已設置")

print("\n步驟 2: 測試基本導入")
print("-" * 50)
try:
    import numpy as np
    print("✅ numpy 已安裝")
except:
    print("❌ numpy 未安裝，請運行: pip install numpy")
    exit(1)

try:
    import sklearn
    print("✅ sklearn 已安裝")
except:
    print("❌ sklearn 未安裝，請運行: pip install scikit-learn")
    exit(1)

print("\n步驟 3: 測試導入 base.py")
print("-" * 50)
try:
    from ofd.adapters.base import BaseAdapter, AdapterFactory
    print("✅ 成功導入 BaseAdapter")
    print("✅ 成功導入 AdapterFactory")
except Exception as e:
    print(f"❌ 導入失敗: {e}")
    print("\n請檢查:")
    print("1. ofd/adapters/base.py 文件是否存在")
    print("2. 文件中是否有語法錯誤")
    exit(1)

print("\n步驟 4: 測試導入 sklearn_adapter.py")
print("-" * 50)
try:
    from ofd.adapters.sklearn_adapter import SklearnAdapter
    print("✅ 成功導入 SklearnAdapter")
except Exception as e:
    print(f"❌ 導入失敗: {e}")
    print("\n請檢查:")
    print("1. ofd/adapters/sklearn_adapter.py 文件是否存在")
    print("2. 文件中是否有語法錯誤")
    print("3. 是否正確導入了 base.py")
    exit(1)

print("\n步驟 5: 測試導入 get_adapter")
print("-" * 50)
try:
    from ofd.adapters import get_adapter
    print("✅ 成功導入 get_adapter")
except Exception as e:
    print(f"❌ 導入失敗: {e}")
    print("\n請檢查:")
    print("1. ofd/adapters/__init__.py 文件是否存在")
    print("2. __init__.py 中是否正確導出了 get_adapter")
    exit(1)

print("\n步驟 6: 測試創建 sklearn 模型")
print("-" * 50)
try:
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=5, max_depth=3)
    print(f"✅ 成功創建模型: {model.__class__.__name__}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    exit(1)

print("\n步驟 7: 測試創建 adapter")
print("-" * 50)
try:
    adapter = get_adapter(model)
    print(f"✅ 成功創建 adapter: {adapter.__class__.__name__}")
    print(f"✅ Adapter 類型: {type(adapter)}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n步驟 8: 測試 adapter 基本方法")
print("-" * 50)
try:
    task_type = adapter.get_task_type()
    print(f"✅ get_task_type(): {task_type}")
    
    is_fitted = adapter.is_fitted()
    print(f"✅ is_fitted(): {is_fitted}")
    
    params = adapter.get_params()
    print(f"✅ get_params(): 獲得 {len(params)} 個參數")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n步驟 9: 測試創建數據")
print("-" * 50)
try:
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    print(f"✅ 創建數據: X shape={X.shape}, y shape={y.shape}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    exit(1)

print("\n步驟 10: 測試訓練模型")
print("-" * 50)
try:
    adapter.fit(X, y)
    print("✅ 模型訓練成功")
    print(f"✅ 訓練後 is_fitted(): {adapter.is_fitted()}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n步驟 11: 測試預測")
print("-" * 50)
try:
    predictions = adapter.predict(X[:10])
    print(f"✅ 預測成功: {predictions}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n步驟 12: 測試評分")
print("-" * 50)
try:
    score = adapter.score(X, y)
    print(f"✅ 評分成功: {score:.4f}")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n步驟 13: 測試數據工具")
print("-" * 50)
try:
    from ofd.utils.data_utils import DataValidator, DataSplitter
    print("✅ 成功導入 DataValidator")
    print("✅ 成功導入 DataSplitter")
    
    X_valid, y_valid = DataValidator.validate_X_y(X, y)
    print("✅ DataValidator.validate_X_y() 正常")
    
    X_train, X_val, y_train, y_val = DataSplitter.train_val_split(X, y)
    print(f"✅ DataSplitter.train_val_split() 正常")
    print(f"   訓練集: {len(X_train)} 樣本, 驗證集: {len(X_val)} 樣本")
except Exception as e:
    print(f"❌ 失敗: {e}")
    import traceback
    traceback.print_exc()
    print("\n這個失敗不影響 adapter 的基本功能")

print("\n" + "="*60)
print("🎉 所有基本測試通過！你的代碼可以使用了！")
print("="*60)
print("\n下一步：運行完整測試")
print("python quick_test.py")