import torch
import torch.nn as nn
import math

## The model definition, updated for compatibility with PreResNet164.
# The core changes involve adopting the layer naming and module construction
# strategy from the PreResNet implementation, particularly for the downsampling
# (shortcut) connections.

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

class Bottleneck(nn.Module):
    """
    Bottleneck block updated to match the PreResNet implementation.
    It now accepts a 'downsample' module directly, rather than creating a
    'shortcut' connection internally. This is the key change for state_dict
    compatibility.
    """
    expansion = 4  # # output channels / # input channels

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        # --- Layer Definition ---
        # conv 1x1
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        # conv 3x3
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        # conv 1x1
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)

        self.relu = nn.ReLU(inplace=True)

        # The downsample module (for shortcut connections) is now passed in.
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        # --- Forward Pass with Pre-activation ---
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        return out


class ResNet(nn.Module):
    """
    ResNet class updated to build layers in a way that is compatible with PreResNet.
    This involves managing `self.inplanes` and creating the `downsample` module
    within the `_make_layer` method.
    """
    def __init__(self, block, depth, seed: int, output_classes=1000):
        super(ResNet, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2 (e.g., 164 or 1001)'

        # Number of blocks per stage. For depth=164, n=18.
        n = (depth - 2) // 9

        self.inplanes = 16

        # Initial convolution
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1,
                               padding=1, bias=False)

        # --- ResNet Stages ---
        # Stage 1 (spatial size: 32x32) -> output planes: 16 * 4 = 64
        self.layer1 = self._make_layer(block, 16, n)
        # Stage 2 (spatial size: 16x16) -> output planes: 32 * 4 = 128
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        # Stage 3 (spatial size: 8x8) -> output planes: 64 * 4 = 256
        self.layer3 = self._make_layer(block, 64, n, stride=2)

        # Final layers
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, output_classes)

        # Weight initialization
        self._init_weights(seed)

    def _init_weights(self, seed):
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        This method is now identical to the one in PreResNetBase to ensure
        structural and naming compatibility.
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# --- The following code remains unchanged as it is not part of the ---
# --- resnet_164 model architecture that needed to be updated.    ---

class ConvFCBase(nn.Module):
    def __init__(self, num_classes: int, seed: int):
        super(ConvFCBase, self).__init__()
        self.conv_part = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
        )
        self.fc_part = nn.Sequential(
            nn.Linear(1152, 1000),
            nn.ReLU(True),
            nn.Linear(1000, 1000),
            nn.ReLU(True),
            nn.Linear(1000, num_classes)
        )
        self._init_weights(seed)

    def _init_weights(self, seed: int):
        """Initializes the weights of the network."""
        torch.manual_seed(seed)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.conv_part(x)
        x = x.view(x.size(0), -1)
        x = self.fc_part(x)
        return x


def resnet_164(output_classes, seed: int):
    """Constructs a ResNet-164 model."""
    model = ResNet(Bottleneck, 164, seed, output_classes)
    return model

def conv33fc(output_classes, seed: int):
    """Constructs a ConvFCBase model."""
    model = ConvFCBase(output_classes, seed)
    return model
