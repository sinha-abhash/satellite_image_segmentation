import torch.nn as nn
import torch.nn.functional as F


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
            nn.ReLU(inplace=True)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(32, 16, (3, 3), 1, 1),
            nn.ReLU(inplace=True)
        )

        self.upsample_block4 = nn.Sequential(
            nn.ConvTranspose2d(16, 32, 4, 2, 1)
        )

        self.conv_block5 = nn.Sequential(
            nn.Conv2d(32, nclasses, 5, 1, 2)
        )

    def forward(self, x):
        score1 = self.conv_block1(x)
        score2 = self.conv_block2(score1)
        score_maxpool = F.max_pool2d(score2, 2)
        score3 = self.conv_block3(score_maxpool)
        score4 = self.upsample_block4(score3)
        score4 += score2
        out = self.conv_block5(score4)

        return out