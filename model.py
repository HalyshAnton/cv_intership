import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models, transforms

class TextClassifier(nn.Module):
    """
    Model with pretrained vgg19_bn feature extractor.
    """
    def __init__(self, num_classes=1):
        super().__init__()

        # pretrained model
        vgg = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

        for param in vgg.parameters():
            param.requires_grad = False

        in_features = vgg.classifier[0].in_features

        vgg.classifier = nn.Identity()

        self.feature_extractor = vgg

        # trainable part
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(in_features, num_classes)


    def forward(self, x):
        out = self.feature_extractor(x)

        out = self.dropout(out)
        out = self.linear(out)
        out = F.sigmoid(out)

        return out

    def predict(self, X):
        """
        Predict the class of the input image.

        Args:
            X (numpy.ndarray): The input image with shape (H, W, C).

        Returns:
            bool: True if the image is predicted to be handwritten,
            False if it is printed
        """

        crop_shape = (max(X.shape[0]*0.5, 20),
                      max(X.shape[1]*0.5, 20))

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.CenterCrop(crop_shape),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        X = transform(X)
        X = X.reshape(1, *X.shape)

        with torch.no_grad():
            y_pred = self.forward(X)

        return y_pred.item() < 0.5
