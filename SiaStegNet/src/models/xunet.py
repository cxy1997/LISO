import torch
import torch.nn as nn
import torch.nn.functional as F


class XuNet(nn.Module):
    def __init__(self):
        super(XuNet, self).__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.KV = torch.tensor([[-1, 2, -2, 2, -1],
                           [2, -6, 8, -6, 2],
                           [-2, 8, -12, 8, -2],
                           [2, -6, 8, -6, 2],
                           [-1, 2, -2, 2, -1]]) / 12.
        self.KV = self.KV.view(1, 1, 5, 5).to(device=device, dtype=torch.float)
        self.KV = torch.autograd.Variable(self.KV, requires_grad=False)

        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 64, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(128)

        self.fc = nn.Linear(128 * 1 * 1, 2)

    def forward(self, x):
        batch_size, _, h, w = x.shape
        x = x.view(batch_size * 3, 1, h, w)
        prep = F.conv2d(x, self.KV.to(x.device), padding=2)
        prep = prep.view(batch_size, 3, h, w)

        out = torch.tanh(self.bn1(torch.abs(self.conv1(prep))))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = torch.tanh(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = torch.relu(self.bn3(self.conv3(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = torch.relu(self.bn4(self.conv4(out)))
        out = F.avg_pool2d(out, kernel_size=5, stride=2, padding=2)

        out = torch.relu(self.bn5(self.conv5(out)))
        out = F.adaptive_avg_pool2d(out, (1, 1))

        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out