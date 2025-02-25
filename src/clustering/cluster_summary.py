import os
import json
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from collections import Counter

# New imports for hierarchical clustering and dendrogram visualization
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt

from matplotlib import font_manager
import matplotlib as mpl
import matplotlib
matplotlib.use('Agg')

NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"

# Matplotlib font settings
nanum_path = "/usr/share/fonts/truetype/nanum/NanumGothicCoding.ttf"
if os.path.exists(nanum_path):
    font_prop = font_manager.FontProperties(fname=nanum_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    print(f"Font set to: {font_prop.get_name()}")
else:
    print("NanumGothicCoding font not found. Using default font.")
    plt.rcParams['font.family'] = 'sans-serif'

mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.default'] = 'regular'

def load_restaurant_data(food):
    """
    Load restaurant data from JSON files for the specified food type.
    Each JSON file contains summaries, positives, negatives, and keywords.
    """
    base_path = f"/mnt/nas4/sms/review_summarization_project/outputs/organized/{food}"
    data_list = []
    for filename in os.listdir(base_path):
        if filename.endswith('.json'):
            with open(os.path.join(base_path, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                restaurant = filename.replace('.json', '')
                summaries = data.get('summaries', [])
                keywords = data.get('keywords', [])
                data_list.append({
                    'restaurant': restaurant,
                    'summaries': summaries,
                    'keywords': keywords
                })
    return pd.DataFrame(data_list)

def generate_embeddings(df):
    """
    Generate a single embedding per restaurant by averaging embeddings of all texts.
    Uses a Korean-specific sentence transformer model.
    
    Parameters:
    - df (pd.DataFrame): DataFrame with a 'summaries' column containing lists of strings.
    
    Returns:
    - pd.DataFrame: DataFrame with an added 'embedding' column containing embedding lists.
    """
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    
    embeddings_list = []
    for _, row in df.iterrows():
        summaries = row['summaries']
        if isinstance(summaries, list) and all(isinstance(s, str) for s in summaries) and summaries:
            embeddings = model.encode(summaries)
            average_embedding = np.mean(embeddings, axis=0).tolist()
        else:
            embedding_dim = model.get_sentence_embedding_dimension()
            average_embedding = [0.0] * embedding_dim
        embeddings_list.append(average_embedding)
    
    df['embedding'] = embeddings_list
    return df

def find_optimal_k_hierarchical(embeddings, max_k=10):
    """
    Determine the optimal number of clusters for hierarchical clustering
    using the silhouette score. Returns the optimal k and a list of scores.
    """
    if len(embeddings) < 2:
        return 1, [0]  # Not enough points to cluster
    
    max_k = min(max_k, len(embeddings) - 1)
    silhouette_scores = []
    
    for k in range(2, max_k + 1):
        clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = clustering.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        silhouette_scores.append(score)
    
    optimal_k = np.argmax(silhouette_scores) + 2 if silhouette_scores else 1
    return optimal_k, silhouette_scores

def cluster_restaurants_hierarchical(df, n_clusters):
    """
    Apply hierarchical (agglomerative) clustering to the restaurant embeddings.
    """
    embeddings = np.array(df['embedding'].tolist())
    clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df['cluster'] = clustering.fit_predict(embeddings)
    return df

def interpret_clusters(df):
    """
    Print cluster interpretations by listing restaurants and their top 5 keywords.
    """
    for cluster in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster]
        all_keywords = [kw for sublist in cluster_df['keywords'] for kw in sublist]
        keyword_counts = Counter(all_keywords)
        top_keywords = keyword_counts.most_common(5)
        print(f"Cluster {cluster}:")
        print(f"Restaurants: {', '.join(cluster_df['restaurant'])}")
        print(f"Top keywords: {top_keywords}\n")

def visualize_dendrogram(embeddings, labels, output_path, title='Restaurant Dendrogram'):
    """
    Generate and save a dendrogram using Ward linkage,
    with file names (restaurant) as labels.
    
    Parameters:
    - embeddings: np.array of shape (n_samples, embedding_dim)
    - labels: a list of strings (e.g., file/restaurant names) to show as leaf labels
    - output_path: where to save the resulting PNG
    - title: the title to show on the dendrogram
    """
    # Perform hierarchical (Ward) linkage
    linked = shc.linkage(embeddings, method='ward')
    
    plt.figure(figsize=(15, 7))
    plt.title(title)
    # Provide the labels parameter to match the embeddings to each restaurant
    dend = shc.dendrogram(
        linked,
        orientation='right',
        distance_sort='descending',
        show_leaf_counts=True,
        labels=labels  # <- This sets the filename/restaurant name as each leaf label
    )
    plt.tight_layout()
    plt.savefig(output_path)
    # If running in a headless environment, you can comment out plt.show()
    plt.show()

def main(args):
    food = args.food
    # Load data
    df = load_restaurant_data(food)
    if len(df) < 2:
        print("Not enough restaurants to cluster.")
        return
    
    # Generate embeddings
    df = generate_embeddings(df)
    embeddings = np.array(df['embedding'].tolist())
    
    # (Optional) Find optimal number of clusters using silhouette
    optimal_k, silhouette_scores = find_optimal_k_hierarchical(embeddings)
    print(f"Silhouette scores for k=2 to k={len(silhouette_scores)+1}: {silhouette_scores}")
    print(f"Optimal k (hierarchical): {optimal_k}")
    
    # Perform hierarchical clustering with the optimal number of clusters
    df = cluster_restaurants_hierarchical(df, optimal_k)
    
    # Interpret results
    interpret_clusters(df)
    
    # Visualize dendrogram (pass in restaurant filenames as labels)
    output_dir = f"/mnt/nas4/sms/review_summarization_project/outputs/clustered/{food}"
    os.makedirs(output_dir, exist_ok=True)
    
    dendrogram_path = os.path.join(output_dir, "dendrogram.png")
    visualize_dendrogram(
        embeddings=embeddings,
        labels=df['restaurant'].tolist(),  # Use the restaurant names (file names) as labels
        output_path=dendrogram_path,
        title=f"{food.capitalize()} Restaurant Dendrogram"
    )
    
    # Save cluster assignments
    cluster_file = os.path.join(output_dir, "clusters.json")
    with open(cluster_file, 'w', encoding='utf-8') as f:
        cluster_data = df[['restaurant', 'cluster']].to_dict(orient='records')
        json.dump(cluster_data, f, ensure_ascii=False, indent=2)
    print(f"Cluster assignments saved to {cluster_file}")
    print(f"Dendrogram image saved to {dendrogram_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical cluster restaurants based on review summaries and keywords.")
    parser.add_argument("--food", type=str, required=True, help="음식 이름")
    args = parser.parse_args()
    main(args)
