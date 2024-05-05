import torch.nn as nn
from torchvision import models
import torch
import os


class TripletNetwork(nn.Module):
    def __init__(self, num_attributes, device, checkpoint_file='checkpoint.pth', pretrained=True):
        super(TripletNetwork, self).__init__()
        checkpoint_file = 'checkpoint.pth'
        self.device = device 
        # Attempt to load from checkpoint
        print(os.path.exists(checkpoint_file))
        if pretrained and os.path.exists(checkpoint_file):
            self.vgg = models.vgg16(pretrained=False)
            state_dict = torch.load(checkpoint_file)
            self.vgg.load_state_dict(state_dict)
        else:
            self.vgg = models.vgg16(pretrained=pretrained)

        num_features = self.vgg.classifier[-1].in_features

        # Remove the original last fully connected layer
        self.vgg.classifier = nn.Sequential(
            *list(self.vgg.classifier.children())[:-1]
        )

        # Create the embedding and attribute branches
        self.embedding_layer = nn.Linear(num_features, 128)  # Example: 128-D embedding
        self.attribute_layer = nn.Linear(num_features, num_attributes)
        self.sigmoid = nn.Sigmoid()

    def forward_once(self, x):
        features = self.vgg(x)  # Extract features from the modified VGG16
        embedding = self.embedding_layer(features)  # Compute embedding
        attributes = self.sigmoid(self.attribute_layer(features))   # Compute attributes

        return embedding, attributes

    def forward(self, anchor, positive, negative):
        # Forward pass for each input type
        embedding_a, attr_a = self.forward_once(anchor)
        embedding_p, attr_p = self.forward_once(positive)
        embedding_n, attr_n = self.forward_once(negative)

        return (embedding_a, attr_a), (embedding_p, attr_p), (embedding_n, attr_n)
    def load_checkpoint(self, checkpoint_file):
        """
        Load model weights from a checkpoint file.
        Args:
            checkpoint_file (str): Path to the checkpoint file.
        """
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.vgg.load_state_dict(checkpoint['vgg_state_dict'])
        self.attribute_layer.load_state_dict(checkpoint['attribute_layer_state_dict'])
        self.embedding_layer.load_state_dict(checkpoint['embedding_layer_state_dict'])
    def save_checkpoint(self, checkpoint_file):
        """
        Save current model weights to a checkpoint file.
        Args:
            checkpoint_file (str): Path to the checkpoint file.
        """
        torch.save({
            'vgg_state_dict': self.vgg.state_dict(),
            'attribute_layer_state_dict': self.attribute_layer.state_dict(),
            'embedding_layer_state_dict': self.embedding_layer.state_dict()
        }, checkpoint_file)
      
