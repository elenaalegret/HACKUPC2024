import torch
from torchvision import transforms
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from torch.utils.data import Dataset
from PIL import Image 
from prediction_network import TripletNetwork
import csv

# Global variables
IMAGE_PATH = '../../data/images_dataset'
CHECKPOINT_PATH = '../../src/model/checkpoint.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Custom Dataset Class
class SubsetImageFolder(Dataset):
    """
    Dataset class for loading specific indices from a single folder.
    """
    def __init__(self, img_dir, all_files, indices=None, transform=None):
        self.img_dir = img_dir # P.e.: '../../data/images_dataset'
        all_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))] # image_0_3.jpg
        self.img_files = all_files if indices is None else [all_files[i] for i in indices] # P.e.: ../../data/images_dataset/image_0_3.jpg
        self.transform = transform
        self.img_paths = [os.path.join(img_dir, f) for f in self.img_files]  # Maintain full paths

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_paths[index]  # Full path to the image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, img_path

def load_triplet_model(num_attributes, checkpoint_file=CHECKPOINT_PATH):
    """
    Load and initialize the TripletNetwork model.
    """
    model = TripletNetwork(num_attributes, device, pretrained=False)
    model.load_checkpoint(checkpoint_file)
    model.to(device)
    model.eval()
    return model

# Function to compute and save embeddings to a CSV file
def save_embeddings_to_csv(dataset, model, output_csv='embeddings.csv'):
    """
    Compute embeddings for a dataset and save them to a CSV file.

    Args:
        dataset (Dataset): Dataset containing images for embedding computation.
        model (nn.Module): Trained model to extract features.
        output_csv (str, optional): Output CSV file name, including the path.
    """
    print('Inicialitzating DataLoader...')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)

    # Initialize a list to store results
    results = []
    header = None

    # Iterate through the DataLoader to compute embeddings
    print('Starting embedding.csv ...')
    for images, paths in dataloader:
        images = images.to(device)

        with torch.no_grad():
            embeddings, _ = model.forward_once(images) # Extract embeddings using the model

        # Convert embeddings to NumPy arrays and append paths
        embeddings_np = embeddings.cpu().numpy()
        for path, embedding in zip(paths, embeddings_np):
            results.append([path] + [embedding.flatten().tolist()])
        
        # Initialize the header based on the first batch
        if header is None:
            print('Header Done')
            header = ['Image Path'] + ['Embedding']
    
    # Write the header and all rows to the CSV file
    with open(output_csv, 'w', newline='') as csvfile:
        print('Writting the embeddings ...')
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(results)

    print(f"Embeddings have been saved to {output_csv}")


# Example Usage
if __name__ == "__main__":
    # Initialize the dataset
    all_files = [f for f in os.listdir(IMAGE_PATH) if f.endswith(('.png', '.jpg', '.jpeg'))]
    dataset = SubsetImageFolder(img_dir=IMAGE_PATH, all_files=all_files, transform=transform)

    # Initialize the model
    model = load_triplet_model(num_attributes=4)
    save_embeddings_to_csv(dataset, model, output_csv='../../data/embeddings.csv')
