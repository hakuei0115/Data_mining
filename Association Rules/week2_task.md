# Find a dataset you want to use to find associations

* 這個數據集來自一個德國雜貨店是在kaggle上找到的，該數據集記錄了大量的購物交易。每一行數據代表一次交易，包含了顧客的編號、交易的日期以及購買的商品。這些數據非常適合用來進行購物籃分析和關聯規則挖掘，因為它提供了顧客的購物行為詳細記錄，有助於理解哪些商品經常一起被購買。
* This dataset, sourced from a German grocery store and found on Kaggle, records a large number of shopping transactions. Each row represents a transaction and includes the customer's ID, the transaction date, and the purchased items. This data is well-suited for basket analysis and association rule mining because it provides detailed records of customer shopping behavior, helping to understand which items are frequently purchased together.

# Describe how to generate a transaction database

To generate a transaction database from the dataset, follow these steps:

1. Read the data from the CSV file using a tool like pandas.
2. Data Transformation: Convert the data into transaction lists by grouping transactions by customer ID (Member_number). Since we are not considering date factors, combine all transactions together to focus on overall purchasing patterns.
   like:
   
   | Member_number | Date | itemDescription |
   | ---- | ---- | ---- |
   |   1   | 2008-12-01 | whole milk |
   |   1   | 2008-12-01 | yogurt |
   |   2   | 2008-12-01 | rolls/buns |
   |   2   | 2008-12-01 | whole milk |
   
   to:
    ```
    [
        ['whole milk', 'yogurt'],
        ['rolls/buns', 'whole milk']
    ]
    ```

# Give an example to show the importance of the discovered rules

當顧客同時購買了牛奶、香腸和其他蔬菜時，他們也有較高的機會購買酸奶和麵包。這條規則是通過關聯規則挖掘過程發現的。我使用了Apriori算法來分析交易數據集，並設置了適當的支持度和置信度閾值。最終，我找到了一條具有顯著提升度（Lift）的規則，表明這些商品之間有較強的購買關聯性。

根據這條規則，商店可以採取以下策略來提升銷售：

1. 搭配銷售：設計促銷活動，如「購買牛奶、香腸和其他蔬菜，酸奶和麵包享折扣」。這樣可以激勵顧客在購買主要商品時，也同時選購促銷商品。
2. 捆綁包裝：將酸奶和麵包與牛奶、香腸和其他蔬菜打包銷售，提供優惠價格。這樣可以促使顧客一次性購買更多商品。
3. 改變商品擺放：將酸奶和麵包擺放在牛奶、香腸和其他蔬菜的附近，以便顧客在選擇主要商品時，自然地注意到促銷商品，增加購買機會。
4. 設計購物路徑：在店鋪內設計顧客的購物路徑，使他們在選擇主要商品後，自然經過促銷商品區域，從而增加對酸奶和麵包的接觸機會。

When customers purchase milk, sausages, and other vegetables together, they also have a higher likelihood of buying yogurt and bread. This rule was discovered through the association rule mining process. I used the Apriori algorithm to analyze the transaction dataset, setting appropriate support and confidence thresholds. Ultimately, I identified a rule with a significant lift, indicating a strong association between these items.

Based on this rule, the store can adopt the following strategies to boost sales:
1. Bundled Promotions: Design promotional activities such as "Buy milk, sausages, and other vegetables, and get a discount on yogurt and bread." This can encourage customers to purchase promotional items along with their main products.
2. Bundled Packaging: Package yogurt and bread with milk, sausages, and other vegetables, offering a special price. This can incentivize customers to buy more items in one go.
3. Product Placement: Position yogurt and bread near milk, sausages, and other vegetables, so that customers naturally notice these promotional items while selecting their main products, increasing the chances of additional purchases.
4. Shopping Path Design: Design the shopping path in the store to guide customers past the promotional items after they have selected their main products, thereby increasing their exposure to yogurt and bread.