import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = 'database.csv'
df = pd.read_csv(file_path)

data_selected = df[['Cause Of Death', 'Nature Of Death', 'Duty', 'Activity', 'Emergency']]

data_selected_clean = data_selected.dropna()

# 將數據轉換為列表
transactions = data_selected_clean.values.tolist()

# 使用 TransactionEncoder 函數轉換數據格式
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# 使用 Apriori 演算法，設定支持度門檻為 0.1
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# 產生信心度超過 0.6 的關聯規則
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

rules.to_csv('analysis_result.csv', index=False)

# 顯示關聯規則
print(rules)
