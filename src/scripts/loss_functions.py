import torch
import torch.nn as nn
import torch.nn.functional as F

class AttributeLoss(nn.Module):
    """
    Computes binary cross-entropy loss for multi-label attribute prediction.
    """
    def __init__(self):
        super(AttributeLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, predicted, target):
        """
        Args:
            predicted: Predicted attribute probabilities (N, K)
            target: Ground-truth binary attribute labels (N, K)
        Returns:
            Loss value
        """
        return self.bce_loss(predicted, target)

class AttributeGuidedTripletLoss(nn.Module):
    """
    Computes an attribute-guided triplet loss using soft-weighted and soft-margin methods.
    """
    def __init__(self, margin=0.8):
        super(AttributeGuidedTripletLoss, self).__init__()
        self.margin = margin

    def cosine_similarity(self, x, y):
        return F.cosine_similarity(x, y)

    def soft_weighted(self, anchor_attr, positive_attr, negative_attr):
        sim_ap = self.cosine_similarity(anchor_attr, positive_attr)
        sim_an = self.cosine_similarity(anchor_attr, negative_attr)
        return sim_ap * sim_an

    def soft_margin(self, anchor_attr, positive_attr, negative_attr):
        sim_ap = self.cosine_similarity(anchor_attr, positive_attr)
        sim_an = self.cosine_similarity(anchor_attr, negative_attr)
        return self.margin * torch.log(1 + sim_ap * sim_an)

    def forward(self, anchor_emb, positive_emb, negative_emb, anchor_attr, positive_attr, negative_attr):
        """
        Args:
            anchor_emb: Embedding of the anchor image (N, D)
            positive_emb: Embedding of the positive image (N, D)
            negative_emb: Embedding of the negative image (N, D)
            anchor_attr: Attribute vector of the anchor image (N, K)
            positive_attr: Attribute vector of the positive image (N, K)
            negative_attr: Attribute vector of the negative image (N, K)
        Returns:
            Loss value
        """
        # Calculate the Euclidean distance between pairs
        dist_ap = F.pairwise_distance(anchor_emb, positive_emb, p=2)
        dist_an = F.pairwise_distance(anchor_emb, negative_emb, p=2)

        # Attribute-guided weighting and margin
        weight_factor = self.soft_weighted(anchor_attr, positive_attr, negative_attr)
        margin_factor = self.soft_margin(anchor_attr, positive_attr, negative_attr)

        # Triplet loss with weighting and margin adjustments
        loss = weight_factor * (dist_ap - dist_an + margin_factor).clamp(min=0.0)

        return loss.mean()
