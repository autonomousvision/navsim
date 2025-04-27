import numpy as np
from sklearn.cluster import KMeans
import os
import pickle
from typing import List, Tuple
import matplotlib.pyplot as plt
import gzip
from tqdm import tqdm
import pickle

class TrajectoryClusterer:
    def __init__(self, n_clusters: int = 5):
        """
        Initialize the trajectory clusterer.
        
        Args:
            n_clusters (int): Number of clusters for K-means
        """
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    
    def load_trajectories(self, data_dir: str) -> List[np.ndarray]:
        trajectories = []
        # TODO: Implement trajectory loading based on your dataset format
        # This is a placeholder - you'll need to modify this based on your actual data format
        for file in tqdm(os.listdir(data_dir)):
            for file2 in os.listdir(os.path.join(data_dir,file)):
                with gzip.open(os.path.join(data_dir,file,file2,'trajectory_target.gz'), 'rb') as f:
                    data = pickle.load(f)
                trajectory = data['trajectory']
                trajectories.append(trajectory)
        return trajectories
        
    def preprocess_trajectories(self, trajectories: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess trajectories for clustering.
        
        Args:
            trajectories (List[np.ndarray]): List of trajectory arrays
            
        Returns:
            np.ndarray: Preprocessed trajectory features
        """
        # Flatten trajectories into feature vectors
        # You might want to add more sophisticated preprocessing here
        features = np.array([traj.numpy().flatten() for traj in trajectories])
        return features
    
    def cluster_trajectories(self, trajectories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Cluster trajectories using K-means.
        
        Args:
            trajectories (List[np.ndarray]): List of trajectory arrays
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Cluster labels and cluster centers
        """
        features = self.preprocess_trajectories(trajectories)
        labels = self.kmeans.fit_predict(features)
        centers = self.kmeans.cluster_centers_
        return labels, centers
    
    def save_clusters(self, labels: np.ndarray, centers: np.ndarray, output_dir: str):
        """
        Save clustering results.
        
        Args:
            labels (np.ndarray): Cluster labels
            centers (np.ndarray): Cluster centers
            output_dir (str): Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save cluster labels
        np.save(os.path.join(output_dir, 'cluster_labels.npy'), labels)
        
        # Save cluster centers
        np.save(os.path.join(output_dir, 'cluster_centers.npy'), centers)
        
        # Save the KMeans model
        with open(os.path.join(output_dir, 'kmeans_model.pkl'), 'wb') as f:
            pickle.dump(self.kmeans, f)
    
    def visualize_clusters(self, trajectories: List[np.ndarray], labels: np.ndarray, centers: np.ndarray, output_dir: str):
        """
        Visualize clustered trajectories.
        
        Args:
            trajectories (List[np.ndarray]): List of trajectory arrays
            labels (np.ndarray): Cluster labels
            output_dir (str): Directory to save visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Plot trajectories with different colors for each cluster
        for i, traj in enumerate(trajectories):
            plt.plot(traj[:, 0], traj[:, 1], c=f'C{labels[i]}', alpha=0.3)
        
        plt.title(f'Trajectory Clusters (K={self.n_clusters})')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, 'trajectory_clusters.png'))
        plt.close()

        plt.figure(figsize=(10, 8))
        for i, center in enumerate(centers):
            plt.plot(center[:, 0], center[:, 1], alpha=0.3)
        plt.legend()
        plt.title('Cluster Centers')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)

        plt.savefig(os.path.join(output_dir, 'cluster_centers.png'))
        plt.close()

def main():
    # Configuration
    data_dir = '.exp/training_cache/'  # Updated data directory path
    output_dir = './clustering_results'
    n_clusters = 4096
    
    # Initialize clusterer
    clusterer = TrajectoryClusterer(n_clusters=n_clusters)
    
    # Load trajectories
    trajectories = clusterer.load_trajectories(data_dir)
    
    if not trajectories:
        print("No trajectories found in the dataset directory.")
        return
    
    # Cluster trajectories
    labels, centers = clusterer.cluster_trajectories(trajectories)
    
    # Save results
    clusterer.save_clusters(labels, centers, output_dir)
    
    # Visualize clusters
    clusterer.visualize_clusters(trajectories, labels, centers, output_dir)
    
    print(f"Clustering completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()