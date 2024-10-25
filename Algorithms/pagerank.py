import numpy as np

def create_adjacency_matrix(links):
    """
    Create adjacency matrix from a dictionary of links.
    
    Args:
        links (dict): Dictionary where keys are pages and values are lists of pages they link to
        
    Returns:
        tuple: (adjacency matrix, index mapping dictionary)
    """
    # Create index mapping for pages
    pages = sorted(list(set(list(links.keys()) + [item for sublist in links.values() for item in sublist])))
    page_to_index = {page: idx for idx, page in enumerate(pages)}
    
    n = len(pages)
    adjacency_matrix = np.zeros((n, n))
    
    # Fill adjacency matrix
    for page, outlinks in links.items():
        if page in page_to_index:  # Check if page exists in mapping
            for outlink in outlinks:
                if outlink in page_to_index:  # Check if outlink exists in mapping
                    adjacency_matrix[page_to_index[outlink]][page_to_index[page]] = 1
    
    return adjacency_matrix, page_to_index

def normalize_matrix(matrix):
    """
    Normalize the adjacency matrix by column to create transition probability matrix.
    Handles dangling nodes (pages with no outlinks).
    """
    # Get column sums
    col_sums = matrix.sum(axis=0)
    
    # Handle columns with zero sum (dangling nodes)
    col_sums[col_sums == 0] = 1
    
    # Normalize columns
    return matrix / col_sums[np.newaxis, :]

def pagerank(links, damping_factor=0.85, epsilon=1e-8, max_iterations=100):
    """
    Calculate PageRank for a network of pages.
    
    Args:
        links (dict): Dictionary where keys are pages and values are lists of pages they link to
        damping_factor (float): Damping factor, typically 0.85
        epsilon (float): Convergence threshold
        max_iterations (int): Maximum number of iterations
        
    Returns:
        dict: Dictionary of page ranks for each page
    """
    # Create and normalize adjacency matrix
    adjacency_matrix, page_to_index = create_adjacency_matrix(links)
    transition_matrix = normalize_matrix(adjacency_matrix)
    
    n = len(transition_matrix)
    
    # Initialize PageRank values
    pagerank_vector = np.ones(n) / n
    
    # Teleportation matrix
    teleport = np.ones((n, n)) / n
    
    # Calculate final transition matrix with damping factor
    M = damping_factor * transition_matrix + (1 - damping_factor) * teleport
    
    # Power iteration
    for _ in range(max_iterations):
        new_pagerank = M @ pagerank_vector
        
        # Check convergence
        if np.sum(np.abs(new_pagerank - pagerank_vector)) < epsilon:
            break
            
        pagerank_vector = new_pagerank
    
    # Convert results to dictionary
    index_to_page = {idx: page for page, idx in page_to_index.items()}
    return {index_to_page[i]: score for i, score in enumerate(pagerank_vector)}

# Example usage
if __name__ == "__main__":
    # Sample web graph
    web_graph = {
        'A': ['B', 'C'],
        'B': ['C'],
        'C': ['A'],
        'D': ['C']
    }
    
    # Calculate PageRank
    ranks = pagerank(web_graph)
    
    # Print results
    print("\nPageRank values:")
    for page, rank in sorted(ranks.items()):
        print(f"Page {page}: {rank:.4f}")
