import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def perform_clustering(data_path='data/raw/user_data.csv', output_path='data/processed/clustered_users.csv'):
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}. Please run data_generation.py first.")
        return

    print("Loading data for behavioral clustering...")
    df = pd.read_csv(data_path)
    
    # Select features for clustering
    features = ['age', 'days_since_signup', 'past_purchases', 'time_spent_mins']
    X = df[features].copy()
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("Applying K-Means Clustering (K=3)...")
    # Using 3 clusters for simplicity: e.g., Power Users, Casual Users, Newbies
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['behavioral_cluster'] = kmeans.fit_predict(X_scaled)
    
    # Map clusters to descriptive names based on their centers
    # Let's find the cluster with the highest time_spent and purchases
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=features)
    cluster_centers['cluster'] = range(3)
    
    # Sort by time_spent_mins to naively assign labels
    sorted_clusters = cluster_centers.sort_values(by='time_spent_mins', ascending=False)['cluster'].tolist()
    
    cluster_map = {
        sorted_clusters[0]: 'Power Users',
        sorted_clusters[1]: 'Regular Users',
        sorted_clusters[2]: 'Casual/New Users'
    }
    
    df['cluster_name'] = df['behavioral_cluster'].map(cluster_map)
    
    print("\n--- Cluster Summary ---")
    summary = df.groupby('cluster_name')[features + ['is_retained']].mean().round(2)
    summary['user_count'] = df.groupby('cluster_name').size()
    print(summary)
    
    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nClustered data saved to {output_path}")

if __name__ == "__main__":
    perform_clustering()
