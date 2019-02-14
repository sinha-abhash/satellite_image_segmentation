import torch.nn as nn


class FCN(nn.Module):
    def __init__(self, nclasses):
        super(FCN, self).__init__()
        self.nclasses = nclasses

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(16, 32, (3, 3), 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 16, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )

        self.upsample_block4 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, 2, 1)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(8, nclasses, 5, 1, 2)
        )

    def forward(self, x):
        score = self.conv_block1(x)
        score = self.conv_block2(score)
        score = self.conv_block3(score)
        score = self.upsample_block4(score)
        out = self.conv_block5(score)

        return out