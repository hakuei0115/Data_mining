import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# 讀取資料
file_path = 'Mall_Customers.csv'
df = pd.read_csv(file_path)

# 選取 年齡、年收入與消費評分 作為聚類依據
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# 對數據進行標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用 Elbow method 確定最佳的聚類數量
inertia = []
for n in range(1, 11):
    kmeans = KMeans(n_clusters=n, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# # 繪製 Elbow method 的圖表
# plt.plot(range(1, 11), inertia, marker='o')
# plt.title('Elbow Method for Optimal Clusters')
# plt.xlabel('Number of clusters')
# plt.ylabel('Inertia')
# plt.show()

# 使用 6 個叢集進行 K-means 聚類
kmeans_6 = KMeans(n_clusters=6, random_state=42)
df['Cluster_6'] = kmeans_6.fit_predict(X_scaled)
centroids_6 = kmeans_6.cluster_centers_

# 繪製 3D 散佈圖 (6 個叢集)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'], 
                c=df['Cluster_6'], cmap='plasma', s=50)

# 繪製 6 個叢集的中心點
ax.scatter(centroids_6[:, 0], centroids_6[:, 1], centroids_6[:, 2], 
           c='blue', s=200, alpha=0.75, label='Centroids')

ax.set_title('3D Scatter Plot of K-means Clustering (6 clusters)')
ax.set_xlabel('Age')
ax.set_ylabel('Annual Income (k$)')
ax.set_zlabel('Spending Score (1-100)')

# Add a color bar to show cluster groupings
plt.colorbar(sc)
plt.show()
