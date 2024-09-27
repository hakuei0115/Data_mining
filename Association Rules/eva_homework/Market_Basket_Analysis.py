import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

file_path = 'Assignment-1_Data.xlsx'
df = pd.read_excel(file_path, sheet_name="retaildata")

df['Itemname'] = df['Itemname'].astype(str)

transactions = df.groupby(['Country', 'BillNo'])['Itemname'].apply(list).tolist()

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

frequent_itemsets = apriori(df_encoded, min_support=0.02, use_colnames=True)

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

rules.to_csv('association_rules.csv', index=False) # 將生成的關聯規則存儲到一個 CSV 文件中，方便日後分析。index=False 表示不將索引存入文件。

print(rules)