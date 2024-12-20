import pandas as pd
import torch
from transformers import BertTokenizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pickle

device = torch.device('mps') if torch.has_mps else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

with open('data/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters, random_state=42)

cluster= kmeans.fit_predict(embeddings)
with open('data/kmeans_clusters_k_4.pkl', 'wb') as f:
    pickle.dump(cluster, f)

print("Cluster labels saved to 'kmeans_clusters_k_4.pkl'")

data.to_csv('data/clustered_github_issues.csv', index=False)

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

k_values = [5,4,3]
plt.figure(figsize=(20, 5))
for i, k in enumerate(k_values):
    k_means_clusters = KMeans(n_clusters=k, random_state=42)
    k_means = k_means_clusters.fit_predict(embeddings)
    plt.subplot(1, 3, i + 1)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=k_means)
    plt.title(f'k={k}')
    plt.legend()
plt.tight_layout()
plt.savefig('kmeans.png')
plt.show()
