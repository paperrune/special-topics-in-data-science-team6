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

class DecBlock(nn.Module):
    def __init__(self, channels):
        super(DecBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels[0], channels[2], kernel_size=1)
        self.bn1   = nn.BatchNorm2d(channels[0])
        self.bn2   = nn.BatchNorm2d(channels[1])
        
    def forward(self, x):
        y = self.conv3(F.interpolate(x, scale_factor=2, mode='nearest'))
        
        x = F.relu(self.bn1(x))
        x = self.conv1(F.interpolate(x, scale_factor=2, mode='nearest'))
        x = self.conv2(F.relu(self.bn2(x)))
        
        return x + y

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.fc = nn.Linear(512, 512 * 7 * 7)
        self.forward1 = nn.ModuleList()
        self.forward1.append(DecBlock([512, 512, 512]))
        self.forward1.append(DecBlock([512, 512, 512]))
        self.forward1.append(DecBlock([512, 256, 256]))
        self.forward1.append(DecBlock([256, 128, 128]))
        self.forward1.append(DecBlock([128,  64,  64]))
        self.forward1.append(nn.Sequential(
                nn.BatchNorm2d(64),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, padding=1),
                nn.Tanh()))
        
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, 7, 7)
        
        for i in range(len(self.forward1)):
            x = self.forward1[i](x)
            
        return x
