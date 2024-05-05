import random
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import os

# Import the previously defined components
from prediction_network import TripletNetwork
from loss_functions import AttributeLoss, AttributeGuidedTripletLoss

# Global variables
PATH = '../../data/metadata.csv'
PATH_IMAGES = '../../data/'
RESIZE_SHAPE = (256, 256) # Reduce 87.5%

class TripletDataset(Dataset):
    def __init__(self, data, model, device, num_attributes, threshold=0.7, transform=True):
        """
        Args:
            data: List or structure containing all available data.
            model: The TripletNetwork model for attribute vector extraction and filtering.
            threshold: Cosine similarity threshold for anchor-positive pairs.
        """
        self.data = data
        self.model = model
        self.num_attributes = num_attributes
        self.threshold = threshold
        self.transform = transform
        self.filtered_triplets = []
        self.device = device
        
        self.filter_triplets()

    def filter_triplets(self):
        """
        Generate triplets based on the provided data and the attribute similarity.
        """
        label_groups = self.data.groupby('Product Type')
        all_labels = label_groups.groups.keys()
        # Generate triplets for each category
        for label in all_labels:
            label_data = label_groups.get_group(label).reset_index(drop=True)
            num_images = len(label_data)

            # Skip categories with less than 2 images
            if num_images < 2:
                continue  

            for _ in range(200):  # Generate multiple triplets per category
                # Choose an anchor image at random
                anchor_idx = random.choice(range(num_images))
                anchor = label_data.iloc[anchor_idx]
                anchor_image_path = anchor['Path']
                anchor_prefix = anchor['Path'].split('/')[-1].split('_')[0] + '_' + anchor['Path'].split('/')[-1].split('_')[1]

                # Find a positive example (with the same prefix + one extra character)
                positive_candidates = [
                    img_path for img_path in label_data['Path']
                    if img_path.split('/')[-1].startswith(anchor_prefix) and img_path != anchor_image_path
                ]

                if positive_candidates:
                    positive_image_path = random.choice(positive_candidates)

                # Find a negative example (different label)
                negative_label = random.choice(list(all_labels - {label}))
                negative_data = label_groups.get_group(negative_label)
                negative = negative_data.sample(n=1).iloc[0]
                negative_image_path = negative['Path']

                binary_label = self.extract_binary_vector_label(all_labels)

                # Append the triplet (anchor, positive, negative, attributes)
                self.filtered_triplets.append((anchor_image_path, positive_image_path, negative_image_path, binary_label[label]))

    def extract_binary_vector_label(self, all_labels):
        """
        Create a dictionary mapping each label to a one-hot encoded vector using NumPy.
            :param all_labels (list): List of unique label integers.

        returns:  dict: Mapping of each label to a one-hot encoded vector.
        """
        num_classes = max(all_labels)
        label_to_one_hot = {}
        for i,label in enumerate(all_labels):
            one_hot = np.zeros(int(num_classes))
            one_hot[i] = 1
            label_to_one_hot[label] = one_hot

        return label_to_one_hot
        
    def __len__(self):
        return len(self.filtered_triplets)

    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path, attributes = self.filtered_triplets[idx]

        def load_image(path):
            image = Image.open(PATH_IMAGES+path).convert('RGB')
            if self.transform:
                transform = transforms.Compose([
                    transforms.Resize(RESIZE_SHAPE),
                    transforms.ToTensor()
                ])
                image = transform(image)
            return image

        anchor = load_image(anchor_path)
        positive = load_image(positive_path)
        negative = load_image(negative_path)

        return anchor, positive, negative, torch.tensor(attributes, dtype=torch.float32, device=self.device)

def load_data_from_csv(csv_path):
    # Load the CSV into a pandas DataFrame
    data = pd.read_csv(csv_path)
    filtered_data = preprocessing_dataset(data)

    filtered_data.to_csv(csv_path, index=False) # Store data in metadata.csv
    return filtered_data

def preprocessing_dataset(data):
    """
    Remove rows with specified Product Types from the dataset.
        :param data: Dataset containing the data.
        :return: Dataset with rows removed.
    """
    filtered_data = data[data['Product Type'].isin(['0', '1', '2', '3', '4', 1, 2, 3, 4, 0])]
    return filtered_data

# Main training loop
def train(model, dataloader, criterion_attr, criterion_triplet, optimizer, device, num_epochs=10):
    model.train()
    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        total_loss = 0.0
        num_batches = len(dataloader)
        print(f"Number of batches (iterations): {num_batches}")
        for i, (anchor, positive, negative, attributes) in enumerate(dataloader):
            print(i)
            # Mover los datos al dispositivo
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)
            attributes = attributes.to(device)

            optimizer.zero_grad()

            # Forward pass through the model
            (embedding_a, attr_a), (embedding_p, attr_p), (embedding_n, attr_n) = model(anchor, positive, negative)

            # Mover attr_a al dispositivo
            attr_a = attr_a.to(device)

            # Calcular la pérdida de atributos
            loss_attr = criterion_attr(attr_a, attributes)
            
            # Calcular la pérdida triplet
            loss_triplet = criterion_triplet(embedding_a, embedding_p, embedding_n, attr_a, attr_p, attr_n)

            # Pérdida total
            total_loss_batch = loss_attr + 1 * loss_triplet
            total_loss_batch.backward()
            optimizer.step()

            total_loss += total_loss_batch.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

    checkpoint_path = '../model'
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    
    model.save_checkpoint(checkpoint_path + '/checkpoint.pth')
    

# Main function
if __name__ == '__main__':
    # Establecer parámetros
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Preparar el conjunto de datos y el dataloader con tripletas filtradas
    data = load_data_from_csv(PATH)
    num_attributes = len(data['Product Type'].unique())
    
    # Inicializar la red y moverla al dispositivo (GPU si está disponible)
    model = TripletNetwork(num_attributes, device).to(device)
    dataset = TripletDataset(data, model, device, num_attributes)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Definir funciones de pérdida
    criterion_attr = AttributeLoss()
    criterion_triplet = AttributeGuidedTripletLoss()

    # Definir optimizador
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Entrenar la red
    train(model, dataloader, criterion_attr, criterion_triplet, optimizer, device, num_epochs=num_epochs)
