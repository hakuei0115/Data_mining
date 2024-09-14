# 導入所需模組
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# 交易資料
transactions = [
    ["牛奶", "麵包", "奶酪"],  # 交易1
    ["牛奶", "麵包"],          # 交易2
    ["牛奶", "咖啡"],          # 交易3
    ["麵包", "奶酪"],          # 交易4
    ["牛奶", "麵包", "咖啡"],  # 交易5
]

# 交易編碼（將交易轉換為二進位表示）
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)


# 使用 apriori 算法來找到頻繁項集，設定最低支持度為 0.6
frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)

# 產生關聯規則，並計算信心度和提升度
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

print(rules)
