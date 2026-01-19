"""
Скрипт для генерации синтетических данных
для экспериментов с градиентным спуском
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification

# 1. Генерация данных для регрессии
print("Генерация данных для регрессии...")
X_reg, y_reg = make_regression(
    n_samples=500,      # 500 примеров
    n_features=1,       # 1 признак
    noise=15,           # уровень шума
    random_state=42     # для воспроизводимости
)

# Сохранение в CSV
df_reg = pd.DataFrame({
    'feature': X_reg.flatten(),
    'target': y_reg
})
df_reg.to_csv('regression_data.csv', index=False)
print(f"  Сохранено: regression_data.csv ({len(df_reg)} строк)")

# 2. Генерация данных для классификации
print("\nГенерация данных для классификации...")
X_clf, y_clf = make_classification(
    n_samples=500,      # 500 примеров
    n_features=2,       # 2 признака
    n_classes=2,        # бинарная классификация
    n_redundant=0,      # без избыточных признаков
    n_clusters_per_class=1,
    random_state=42
)

# Сохранение в CSV
df_clf = pd.DataFrame({
    'feature1': X_clf[:, 0],
    'feature2': X_clf[:, 1],
    'target': y_clf
})
df_clf.to_csv('classification_data.csv', index=False)
print(f"  Сохранено: classification_data.csv ({len(df_clf)} строк)")

print("\nГенерация данных завершена!")
print("Файлы сохранены в папке data/")
