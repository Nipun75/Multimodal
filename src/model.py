import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


class MultimodalCardiomegalyModel(nn.Module):
    """
    X-ray -> DenseNet-121 -> 256-D image embedding
    Age + Sex -> MLP -> 64-D demographic embedding
    Concatenate -> 320-D -> 128 -> 1 logit
    """

    def __init__(self, pretrained=True, dropout=0.3):
        super().__init__()

        weights = DenseNet121_Weights.DEFAULT if pretrained else None
        backbone = densenet121(weights=weights)

        # Keep the convolutional feature extractor for Grad-CAM.
        self.image_encoder = backbone.features
        image_feature_dim = backbone.classifier.in_features

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Input: normalized age + binary sex encoding.
        self.demographic_encoder = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward_image_features(self, images):
        x = self.image_encoder(images)
        x = torch.relu(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.image_projection(x)

    def forward_demographic_features(self, age, sex):
        demographic = torch.cat([age, sex], dim=1)
        return self.demographic_encoder(demographic)

    def forward(self, images, age, sex):
        image_embedding = self.forward_image_features(images)
        demographic_embedding = self.forward_demographic_features(age, sex)

        fused = torch.cat([image_embedding, demographic_embedding], dim=1)
        return self.fusion(fused).squeeze(1)
