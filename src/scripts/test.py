import torch
from torchvision import transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import ast


# Global variables
IMAGE_PATH = '../../data/'
CHECKPOINT_PATH = '../../src/model/checkpoint.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def find_similar_images(base_image_path, comparison_paths, embeddings_csv='../../data/embeddings.csv', top_k=12):
    """
    Find the top K most similar images to a base image using embeddings stored in a CSV file.
    
    Args:
        base_image_path (str): Path to the base image to compare against.
        embeddings_csv (str, optional): Path to the CSV file containing image embeddings.
        top_k (int, optional): Number of most similar images to return.
    
    Returns:
        List[str]: Paths of the top K most similar images.
    """
    # Read the CSV file containing paths and embeddings
    df = pd.read_csv(embeddings_csv)
    base_image_path = IMAGE_PATH+base_image_path

    # Convert embeddings from strings to numerical lists
    def parse_embedding_string(embedding_str):
        return np.array(ast.literal_eval(embedding_str))

    # Apply the conversion function to each embedding in the DataFrame
    df['Embedding'] = df['Embedding'].apply(parse_embedding_string)

    # Extract the image paths and embeddings
    image_paths = df['Image Path'].values
    embeddings = np.vstack(df['Embedding'].values)

    # Locate the base image's embedding
    try:
        base_idx = list(image_paths).index(base_image_path)
    except ValueError:
        raise ValueError(f"Base image path '{base_image_path}' not found in CSV.")

    # Reshape the base embedding to 2D (1, number of features)
    base_embedding = embeddings[base_idx].reshape(1, -1)

    # Filter the embeddings DataFrame to only include comparison paths
    comparison_df = df[df['Image Path'].isin(IMAGE_PATH+comparison_paths)]
    comparison_embeddings = np.vstack(comparison_df['Embedding'].values)
    comparison_image_paths = comparison_df['Image Path'].values

    # Compute cosine similarities between the base image and the comparison subset
    similarities = cosine_similarity(base_embedding, comparison_embeddings)[0]

    # Sort indices by descending similarity
    top_indices = np.argsort(-similarities)[:top_k]

    # Retrieve the paths of the most similar images
    most_similar_paths = [comparison_image_paths[i] for i in top_indices]

    return most_similar_paths
