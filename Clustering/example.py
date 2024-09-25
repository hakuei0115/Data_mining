# 導入必要的庫
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成一個簡單的 2D 資料集，含有 300 個樣本和 3 個群組
X, y = make_blobs(n_samples=300, centers=3, random_state=42)

# 使用 K-means 演算法進行分群，這裡我們指定 K=3
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 取得分群的結果
y_kmeans = kmeans.predict(X)

# 畫出分群結果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

# 畫出群組的中心（質心）
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='x')

plt.title("K-means Clustering Example")
plt.show()