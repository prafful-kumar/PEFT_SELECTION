import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph

def compute_similarity_matrix(features, epsilon=1.0, k=10):
    """
    Compute a weighted similarity graph using k-nearest neighbors (k-NN).
    
    Parameters:
        features (np.ndarray): Feature matrix of shape (M, d).
        epsilon (float): Scaling parameter for Gaussian kernel.
        k (int): Number of nearest neighbors to consider.

    Returns:
        np.ndarray: k-NN similarity matrix K of shape (M, M).
    """
    pairwise_dist = pairwise_distances(features, metric='euclidean')
    K = np.exp(-pairwise_dist**2 / (2 * epsilon**2))  # Apply Gaussian kernel
    
    knn_graph = kneighbors_graph(features, k, mode='connectivity', include_self=True)
    K = K * knn_graph.toarray()  # Keep only k-nearest neighbors

    return K

def compute_markov_transition_matrix(K):
    """
    Compute the Markov transition matrix P from the similarity matrix K.

    Parameters:
        K (np.ndarray): Similarity matrix of shape (M, M).

    Returns:
        np.ndarray: Markov transition matrix P of shape (M, M).
    """
    P = K / np.sum(K, axis=1, keepdims=True)

    return P

def compute_spectral_decomposition(P, num_components=50):
    """
    Perform spectral decomposition of the Markov transition matrix.

    Parameters:
        P (np.ndarray): Markov transition matrix.
        num_components (int): Number of dominant eigenvalues/eigenvectors to retain.

    Returns:
        eigenvalues (np.ndarray): Top eigenvalues.
        eigenvectors (np.ndarray): Corresponding eigenvectors.
    """
    eigenvalues, eigenvectors = eigsh(P, k=num_components, which='LM')
    
    # Sorting the eigenvalues in descending order
    idx = np.argsort(eigenvalues)[::-1]  
    eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

    return eigenvalues, eigenvectors

def compute_intra_diffusion_distance(eigenvalues, eigenvectors, features, labels, t=10):
    """
    Compute diffusion distance using spectral decomposition.

    Parameters:
        eigenvalues (np.ndarray): Eigenvalues from spectral decomposition.
        eigenvectors (np.ndarray): Corresponding eigenvectors.
        features (np.ndarray): Feature matrix (M, d).
        labels (np.ndarray): Class labels of shape (M,).
        t (int): Diffusion time step.

    Returns:
        diffusion_scores (dict): Diffusion score per class.
    """
    unique_classes = np.unique(labels)
    class_means = {}

    # Compute class mean embeddings in diffusion space
    for c in unique_classes:
        class_mask = labels == c
        class_means[c] = np.mean(eigenvectors[class_mask], axis=0)

    diffusion_scores = {}

    for c in unique_classes:
        class_mask = labels == c
        distances = np.sum(
            (eigenvectors[class_mask] - class_means[c])**2 * (eigenvalues**(2*t)),
            axis=1
        )
        diffusion_scores[c] = np.mean(distances)  # Average diffusion distance to class mean
    
    return diffusion_scores

def compute_class_means(eigenvectors, labels):
    """ Compute mean embeddings for each class in diffusion space. """
    unique_classes = np.unique(labels)
    class_means = {c: np.mean(eigenvectors[labels == c], axis=0) for c in unique_classes}
    return class_means

def compute_inter_class_diffusion_distance(eigenvalues, class_means, t=10):
    """
    Compute diffusion distance between means of different classes.
    
    Parameters:
        eigenvalues (np.ndarray): Eigenvalues from spectral decomposition.
        class_means (dict): Mean embeddings of each class.
        t (int): Diffusion time step.

    Returns:
        dict: Inter-class diffusion distances.
    """
    unique_classes = list(class_means.keys())
    inter_class_distances = {}

    for i in range(len(unique_classes)):
        for j in range(i + 1, len(unique_classes)):  
            c1, c2 = unique_classes[i], unique_classes[j]
            distance = np.sum((class_means[c1] - class_means[c2])**2 * (eigenvalues**(2*t)))
            inter_class_distances[(c1, c2)] = np.sqrt(distance)  # Symmetric distance

    return inter_class_distances

def compute_diffusion_score(features, labels,eigenvalues, eigenvectors, epsilon=1.0, k=10, t=10, num_components=50,intra=True):
    """
    Compute diffusion score for PEFT selection based on class separability.

    Parameters:
        features (np.ndarray): Feature matrix of shape (M, d).
        labels (np.ndarray): Corresponding class labels of shape (M,).
        epsilon (float): Kernel bandwidth parameter.
        k (int): Number of nearest neighbors.
        t (int): Diffusion time step.
        num_components (int): Number of eigen components to retain.

    Returns:
        dict: Diffusion score per class.
    """

    if intra:

        # Step 1: Compute similarity matrix K
        K = compute_similarity_matrix(features, epsilon, k)

        # Step 2: Compute Markov transition matrix P
        P = compute_markov_transition_matrix(K)

        # Step 3: Perform spectral decomposition
        eigenvalues, eigenvectors = compute_spectral_decomposition(P, num_components=num_components)
        
        # Step 4: Compute diffusion distance and score
        diffusion_scores = compute_intra_diffusion_distance(eigenvalues, eigenvectors, features, labels, t=t)

        # Average intra-class diffusion distance
        avg_intra_class_scores = np.mean(list(diffusion_scores.values()))

        return avg_intra_class_scores,eigenvalues, eigenvectors
    
    else:

        # Step 4: Compute class means in diffusion space
        class_means = compute_class_means(eigenvectors, labels)

        # Step 5: Compute inter-class diffusion distances
        inter_class_distances = compute_inter_class_diffusion_distance(eigenvalues, class_means, t=t)

        # Average inter-class diffusion distance
        avg_inter_class_distance = np.mean(list(inter_class_distances.values()))

        return avg_inter_class_distance
