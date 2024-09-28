import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

file_path = 'database.csv'
df = pd.read_csv(file_path)

# Selecting relevant columns for association rule mining
columns_of_interest = ['Cause Of Death', 'Nature Of Death', 'Duty', 'Activity', 'Property Type']

# Dropping any rows with missing values in the selected columns
cleaned_data = df[columns_of_interest].dropna()

# Assuming 'cleaned_data' contains your relevant columns
transactions = cleaned_data.applymap(lambda x: str(x))
transactions_list = transactions.values.tolist()

# Convert transactions to one-hot encoding format
te = TransactionEncoder()
te_ary = te.fit(transactions_list).transform(transactions_list)
transaction_df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm
frequent_itemsets = apriori(transaction_df, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

rules.to_csv('association_rules.csv', index=False)

# Display the results
print(rules)
