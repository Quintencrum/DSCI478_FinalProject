import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")




def plot_network(distance_matrix, title):   #create network graphs
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title(title, fontsize=16)

    dissimilarity_matrix = 1 - distance_matrix / np.max(distance_matrix)
    
    positions = MDS(n_components=2, dissimilarity="precomputed", random_state=42).fit_transform(dissimilarity_matrix) 
    
    # Edges
    for i in range(distance_matrix.shape[0]):
        for j in range(i+1, distance_matrix.shape[0]):
            ax.plot(
                [positions[i,0], positions[j,0]], 
                [positions[i,1], positions[j,1]], 
                linewidth=distance_matrix[i,j]*5, 
                color="gray", 
                alpha=0.5
            )
    
    # Nodes
    for i in range(distance_matrix.shape[0]):
        ax.scatter(
            positions[i,0], 
            positions[i,1], 
            s=np.power(10, 2+np.sqrt(distance_matrix[i,i])), 
            alpha=0.8, 
            color="#DA70D6"
        )
        
    # Remove axis elemtns
    ax.set_xticks([])
    ax.set_yticks([])

    plt.show()



# #sample distance matix to test plot
# distance_matrix = np.array([
#     [0.0, 0.2, 0.4, 0.6],
#     [0.2, 0.0, 0.3, 0.5],
#     [0.4, 0.3, 0.0, 0.8],
#     [0.6, 0.5, 0.8, 0.0]
# ])
# plot_network(distance_matrix, "Sample Network")
