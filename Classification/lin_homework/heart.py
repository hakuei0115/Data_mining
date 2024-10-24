import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv('heart.csv')

# 切分特徵與目標變數
X = data.drop('output', axis=1)
y = data['output']

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分訓練集與測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立隨機森林模型
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# 預測
y_pred_rf = rf_model.predict(X_test)

# 評估模型
accuracy_rf = accuracy_score(y_test, y_pred_rf)
classification_report_rf = classification_report(y_test, y_pred_rf)

# print(accuracy_rf, classification_report_rf)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# 混淆矩陣
conf_matrix = confusion_matrix(y_test, y_pred_rf)

# 繪製混淆矩陣圖表
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 繪製特徵重要性圖表
importances = rf_model.feature_importances_
features = data.columns[:-1]

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="skyblue")
plt.title("Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
