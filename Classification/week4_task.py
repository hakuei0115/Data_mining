import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score

file_path = 'nasa.csv'
nasa_data = pd.read_csv(file_path)

# 先進行資料預處理，將 "Hazardous" 作為標籤（y），其餘的數值特徵作為特徵集（X）
# 我們只會使用數值特徵進行分類，移除非數值列，例如 'Name' 和 'Neo Reference ID'

# 選擇數值特徵並移除無關特徵
X = nasa_data.drop(columns=['Neo Reference ID', 'Name', 'Hazardous', 'Equinox'])
y = nasa_data['Hazardous'].apply(lambda x: 1 if x == True else 0)  # 將 Hazardous 轉換為二進制數據

# 找出並移除數據中的非數值列
X_numeric = X.select_dtypes(include=['float64', 'int64'])

# 將數據集分割為訓練集和測試集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, random_state=42)

# 構建隨機森林分類器並訓練模型
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 預測測試集
y_pred = rf_model.predict(X_test)

# 評估模型表現
from sklearn.metrics import classification_report, accuracy_score

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}', classification_rep)
