from itertools import combinations

transactions = [
    {"牛奶", "麵包", "奶酪"},  # 交易1
    {"牛奶", "麵包"},          # 交易2
    {"牛奶", "咖啡"},          # 交易3
    {"麵包", "奶酪"},          # 交易4
    {"牛奶", "麵包", "咖啡"},  # 交易5
]

# Step 2: 提取單項集
def get_single_itemsets(transactions):
    itemsets = set()
    for transaction in transactions:
        itemsets.update(transaction)
    return [{item} for item in itemsets]

single_itemsets = get_single_itemsets(transactions)
# print(single_itemsets)

# Step 3: 計算支持度
def calculate_support(itemset, transactions):
    count = sum([1 for transaction in transactions if itemset.issubset(transaction)])
    return count / len(transactions)

# 計算單項集的支持度
min_support = 0.6  # 設定最低支持度閾值
frequent_itemsets = []

for itemset in single_itemsets:
    support = calculate_support(itemset, transactions)
    if support >= min_support:
        frequent_itemsets.append((itemset, support))

# print(frequent_itemsets)

# Step 4: 合併生成更大的項集
def generate_candidate_itemsets(frequent_itemsets, k):
    # 合併項集，生成k項集
    items = set()
    for itemset, _ in frequent_itemsets:
        items.update(itemset)
    return [set(combo) for combo in combinations(items, k)]

# 生成二項集
candidate_2_itemsets = generate_candidate_itemsets(frequent_itemsets, 2)
# print(candidate_2_itemsets)

# Step 5: 計算二項集的支持度
frequent_2_itemsets = []

for itemset in candidate_2_itemsets:
    support = calculate_support(itemset, transactions)
    if support >= min_support:
        frequent_2_itemsets.append((itemset, support))

# print(frequent_2_itemsets)

# Step 6: 計算信心度和提升度
def calculate_confidence(itemset_A, itemset_B, transactions):
    support_A = calculate_support(itemset_A, transactions)
    support_AB = calculate_support(itemset_A.union(itemset_B), transactions)
    return support_AB / support_A

def calculate_lift(itemset_A, itemset_B, transactions):
    support_A = calculate_support(itemset_A, transactions)
    support_B = calculate_support(itemset_B, transactions)
    support_AB = calculate_support(itemset_A.union(itemset_B), transactions)
    return support_AB / (support_A * support_B)

# 對於每個頻繁的二項集，生成規則並計算指標
for itemset, support in frequent_2_itemsets:
    itemset_list = list(itemset)
    for i in range(len(itemset_list)):
        itemset_A = {itemset_list[i]}
        itemset_B = itemset.difference(itemset_A)
        confidence = calculate_confidence(itemset_A, itemset_B, transactions)
        lift = calculate_lift(itemset_A, itemset_B, transactions)
        print(f"規則: {itemset_A} -> {itemset_B}, 信心度: {confidence:.2f}, 提升度: {lift:.2f}")
