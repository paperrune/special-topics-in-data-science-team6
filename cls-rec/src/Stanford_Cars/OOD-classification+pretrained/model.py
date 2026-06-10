import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights

class Encoder(nn.Module):
    def __init__(self, out_dim_1=512, out_dim_2=196, hidden_dim=512, pretrained=True):
        super().__init__()

        weights  = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        self.fc1 = nn.Linear(hidden_dim, 2 * out_dim_1)
        self.fc2 = nn.Linear(hidden_dim, out_dim_2)

    def forward(self, x, mean=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        z_mean, z_var = self.fc1(x).chunk(2, dim=1)
        return (z_mean + F.softplus(z_var) * torch.randn_like(z_mean) if mean == False else z_mean).tanh(), self.fc2(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.forward1 = nn.Sequential(
                nn.LeakyReLU(0.2),
                nn.Linear(1024, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.1),
                nn.Linear(1024, 1024))

        self.fc1 = nn.Linear( 512, 1024)
        self.fc2 = nn.Linear(1024,    1)

    def forward(self, x, feature=False):
        x = self.fc1(x)
        x = self.fc2(F.leaky_relu(self.forward1(x) + x, 0.2))

        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.forward1 = nn.Sequential(
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear(1024, 1024),
                nn.BatchNorm1d(1024),
                nn.LeakyReLU(inplace=True),
                nn.Linear(1024, 1024))

        self.bn  = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear( 512, 1024)
        self.fc2 = nn.Linear(1024,  196)

    def forward(self, z):
        z = self.fc1(z)
        z = self.fc2(F.leaky_relu(self.bn(self.forward1(z) + z)))

        return z
