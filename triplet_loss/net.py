from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_NET(nn.Module):
    def __init__(self, classes=10):
        super(MNIST_NET, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.classifier = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(True),
            nn.Linear(120, 84),
            nn.ReLU(True),
            nn.Linear(84, 10)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        # out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


if __name__ == '__main__':
    x = torch.randn(32, 1, 28, 28)
    print(x.size())
    net = MNIST_NET()
    out = net(x)
    print(out.size())
