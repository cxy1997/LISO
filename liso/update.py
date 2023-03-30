import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """
    Input: (N, input_dim, H, W)
    Output: (N, hidden_dim, H, W)
    """
    def __init__(self, input_dim=64, hidden_dim=256):
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class SepConvGRU(nn.Module):
    """
    Input:
        h: (N, hidden_dim, H, W)
        x: (N, input_dim, H, W)
    Output: (N, hidden_dim, H, W)
    """
    def __init__(self, hidden_dim=64, input_dim=192+64):
        super(SepConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)


    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)  # 256
        z = torch.sigmoid(self.convz(hx))  # 64
        r = torch.sigmoid(self.convr(hx))  # 64
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))
        h = (1-z) * h + z * q

        return h


class BasicMotionEncoder(nn.Module):
    """
    Input:
        noise: (N, channels, H, W)
        corr: (N, channels, H, W)
    Output: (N, output_dim, H, W)
    """
    def __init__(self, channels=3, hidden_dim=32, output_dim=64):
        super(BasicMotionEncoder, self).__init__()
        self.convc1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.convc2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.convf1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.convf2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv = nn.Conv2d(output_dim, output_dim-channels, 3, padding=1)

    def forward(self, noise, corr):
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))
        flo = F.relu(self.convf1(noise))
        flo = F.relu(self.convf2(flo))

        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, noise], dim=1)


class BasicUpdateBlock(nn.Module):
    """
    Input:
        net: (N, hidden_dim, H, W)
        inp: (N, hidden_dim, H, W)
        corr: (N, 3, H, W)
        noise: (N, 3, H, W)
    Output:
        net: (N, hidden_dim, H, W)
        delta_flow: (N, 3, H, W)
    """
    def __init__(self, hidden_dim=64):
        super(BasicUpdateBlock, self).__init__()
        self.encoder = BasicMotionEncoder()
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=64+hidden_dim)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=64)

    def forward(self, net, inp, corr, noise):
        motion_features = self.encoder(noise, corr)  # 64
        inp = torch.cat([inp, motion_features], dim=1)  # 64+64

        net = self.gru(net, inp)  # 64, 192 -> 64
        delta_flow = self.flow_head(net)

        return net, delta_flow