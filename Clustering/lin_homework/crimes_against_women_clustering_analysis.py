import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

def draw_Elbow(data, numerical_columns):
    sse = []

    k_values = range(1, 11)
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data[numerical_columns])
        sse.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 6))
    plt.plot(k_values, sse, marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.grid(True)
    plt.show()
    
def kmeans_clustering(data, numerical_columns, k):
    kmeans = KMeans(n_clusters=k, random_state=0)
    data['Cluster'] = kmeans.fit_predict(data[numerical_columns])
    return data

def visualize_clusters(data, numerical_columns):
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data[numerical_columns])

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c=data['Cluster'], cmap='viridis', marker='o')
    plt.title('K-means Clustering (K=3) with PCA Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    plt.grid(True)
    plt.show()

def draw_boxplot(data, numerical_columns):
    cluster_melted = data.melt(id_vars=['Cluster'], value_vars=numerical_columns, var_name='Crime Type', value_name='Crime Rate')
    plt.figure(figsize=(12, 6))

    sns.boxplot(x='Crime Type', y='Crime Rate', hue='Cluster', data=cluster_melted)

    plt.title('Crime Type Distribution by Cluster')
    plt.xticks(rotation=45)
    plt.ylabel('Normalized Crime Rate')
    plt.xlabel('Crime Type')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'CrimesOnWomenData.csv'
    data = pd.read_csv(file_path)
    
    data = data.drop(columns=['Unnamed: 0'])
    
    scaler = MinMaxScaler()
    numerical_columns = ['Rape', 'K&A', 'DD', 'AoW', 'AoM', 'DV', 'WT']
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    draw_Elbow(data, numerical_columns)
    data = kmeans_clustering(data, numerical_columns, 3)
    # visualize_clusters(data, numerical_columns)
    # draw_boxplot(data, numerical_columns)
    
    cluster_state_distribution = data.groupby('Cluster')['State'].value_counts()
    print(cluster_state_distribution)