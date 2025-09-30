"""
Performance Gap Calculation Example
展示如何使用性能差距計算和過擬合檢測方法
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 導入 OFiD 方法
from ofd import get_adapter, calculate_performance_gap, detect_overfitting_basic


def main():
    print("🌸 OFiD Performance Gap Example - Senpai's Overfitting Detector! ✨")
    print("=" * 60)
    
    # 1. 創建數據
    print("📊 Step 1: Creating dataset...")
    X, y = make_classification(
        n_samples=1000, 
        n_features=20, 
        n_informative=10,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    # 分割數據
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"✅ Training set: {X_train.shape[0]} samples")
    print(f"✅ Validation set: {X_val.shape[0]} samples")
    
    # 2. 創建和訓練模型
    print("\n🤖 Step 2: Creating and training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,  # 故意設深一點來製造過擬合
        random_state=42
    )
    
    adapter = get_adapter(model)
    adapter.fit(X_train, y_train)
    print(f"✅ Model trained: {model.__class__.__name__}")
    
    # 3. 計算性能差距
    print("\n📈 Step 3: Calculating performance gap...")
    gap_results = calculate_performance_gap(
        adapter, X_train, y_train, X_val, y_val, 
        metrics=['accuracy', 'f1']
    )
    
    print("🎯 Performance Gap Results:")
    for metric, results in gap_results.items():
        print(f"  {metric.upper()}:")
        print(f"    Training Score: {results['train_score']:.4f}")
        print(f"    Validation Score: {results['val_score']:.4f}")
        print(f"    Gap: {results['gap']:.4f}")
        print(f"    Gap %: {results['gap_percentage']:.2f}%")
        print()
    
    # 4. 基礎過擬合檢測
    print("🔍 Step 4: Basic overfitting detection...")
    detection_result = detect_overfitting_basic(
        adapter, X_train, y_train, X_val, y_val,
        gap_threshold=0.05,  # 5% 差距閾值
        metric='accuracy'
    )
    
    print("🚨 Overfitting Detection Results:")
    print(f"  Is Overfitting: {detection_result['is_overfitting']}")
    print(f"  Metric: {detection_result['metric']}")
    print(f"  Gap: {detection_result['gap']:.4f}")
    print(f"  Gap %: {detection_result['gap_percentage']:.2f}%")
    print(f"  Severity: {detection_result['severity']}")
    print(f"  Recommendation: {detection_result['recommendation']}")
    
    # 5. 嘗試不同的閾值
    print("\n🎚️ Step 5: Testing different thresholds...")
    thresholds = [0.01, 0.05, 0.1, 0.2]
    
    for threshold in thresholds:
        result = detect_overfitting_basic(
            adapter, X_train, y_train, X_val, y_val,
            gap_threshold=threshold
        )
        print(f"  Threshold {threshold:.2f}: {result['is_overfitting']} ({result['severity']})")
    
    print("\n🎉 Example completed! Your overfitting detector is working perfectly! ✨")


if __name__ == "__main__":
    main()
