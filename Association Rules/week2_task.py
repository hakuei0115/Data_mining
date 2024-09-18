import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# 1. 讀取 Groceries 資料集
df = pd.read_csv('Groceries_dataset.csv')

# 2. 將資料轉換為交易格式
transactions = df.groupby('Member_number')['itemDescription'].apply(list).tolist()

# 3. 將資料轉換為 one-hot encoding 格式
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# 4. 應用 Apriori 演算法
frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)

# 5. 生成關聯規則
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=2.4) # 找出最強關聯規則

print(rules)