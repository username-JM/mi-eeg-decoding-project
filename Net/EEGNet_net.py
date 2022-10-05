
import torch.nn as nn


class EEGNet(nn.Module):
    def __init__(self):
        activation = nn.ReLU
        super(EEGNet, self).__init__()

        self.spectral_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 125), stride=(1, 1), padding=(0, 62), bias=False),
            nn.BatchNorm2d(16)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(22, 1), stride=(1, 1), groups=16, bias=False),
            nn.BatchNorm2d(32),
            activation(),
            nn.MaxPool2d((1, 4)),
            nn.Dropout(0.25)
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), stride=(1, 1), groups=32, bias=False),
            nn.Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(32),
            activation(),
            nn.MaxPool2d((1, 8)),
            nn.Dropout(0.25)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1056, out_features=4, bias=True)
        )

    def forward(self, x):
        out = self.spectral_conv(x)
        out = self.spatial_conv(out)
        out = self.separable_conv(out)
        out = out.view(-1, self.classifier[0].in_features)
        out = self.classifier(out)
        return out