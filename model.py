import torch
import torch.nn as nn
import timm


class AttentionPooling(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):

        # x: (B, T, C)
        weights = self.attention(x)
        weights = torch.softmax(weights, dim=1)
        pooled = (x * weights).sum(dim=1)

        return pooled


class BirdModel(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            in_chans=1,
            num_classes=0
        )

        feature_dim = self.encoder.num_features
        self.pool = AttentionPooling(feature_dim)
        self.head = nn.Linear(
            feature_dim,
            num_classes
        )

    def forward(self, x):

        # x: (B,1,H,W)
        features = self.encoder.forward_features(x)
        # (B,C,H,W)
        features = features.mean(dim=2)
        # (B,C,T)
        features = features.permute(0, 2, 1)
        # (B,T,C)
        pooled = self.pool(features)
        logits = self.head(pooled)

        return logits
