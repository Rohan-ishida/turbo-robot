import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters: int, max_iters: int = 100, random_state: int = None):
        """
        Initialize K-Means clustering algorithm
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (K)
        max_iters : int
            Maximum number of iterations for convergence
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        
    def fit(self, X: np.ndarray) -> 'KMeans':
        """
        Fit K-Means clustering to the data
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data of shape (n_samples, n_features)
            
        Returns:
        --------
        self : KMeans
            Fitted estimator
        """
        if self.random_state:
            np.random.seed(self.random_state)
            
        # Randomly initialize centroids
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx]
        
        for _ in range(self.max_iters):
            # Assign points to nearest centroid
            old_centroids = self.centroids.copy()
            distances = self._calculate_distances(X)
            self.labels = np.argmin(distances, axis=1)
            
            # Update centroids
            for k in range(self.n_clusters):
                if len(X[self.labels == k]) > 0:
                    self.centroids[k] = np.mean(X[self.labels == k], axis=0)
                    
            # Check for convergence
            if np.all(old_centroids == self.centroids):
                break
                
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Parameters:
        -----------
        X : numpy.ndarray
            New data of shape (n_samples, n_features)
            
        Returns:
        --------
        labels : numpy.ndarray
            Predicted cluster labels
        """
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """Calculate distances between points and centroids"""
        distances = np.zeros((len(X), self.n_clusters))
        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)
        return distances

def plot_clusters(X: np.ndarray, labels: np.ndarray, centroids: np.ndarray = None):
    """
    Plot the clustered data
    
    Parameters:
    -----------
    X : numpy.ndarray
        Data points
    labels : numpy.ndarray
        Cluster labels
    centroids : numpy.ndarray, optional
        Cluster centroids
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, linewidths=3)
    plt.show()

# Example usage
if __name__ == "__main__":
    # Generate sample data
    np.random.seed(42)
    n_samples = 300
    X = np.concatenate([
        np.random.normal(0, 1, (n_samples, 2)),
        np.random.normal(4, 1, (n_samples, 2)),
        np.random.normal(8, 1, (n_samples, 2))
    ])
    
    # Create and fit KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Plot results
    plot_clusters(X, kmeans.labels, kmeans.centroids)
