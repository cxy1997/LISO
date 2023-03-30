import numpy as np
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torchvision.transforms import RandomCrop, CenterCrop

from .update import BasicUpdateBlock

import sys
sys.path.append("SiaStegNet/src")
from models import KeNet, XuNet


class ContextEncoder(nn.Module):
    """
    The ContextEncoder module takes a cover image + embedded data,
        and produces a corresponding context feature.

    Input:
        image: (N, 3, H, W)
        data: (N, data_depth, H, W)
    Output: (N, hidden_size, H, W)
    """

    def __init__(self, data_depth, hidden_size):
        super(ContextEncoder, self).__init__()
        self.data_depth = data_depth
        self.hidden_size = hidden_size

        self.features = nn.Sequential(
            self._conv2d(3, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
        )
        self.layers = nn.Sequential(
            self._conv2d(self.hidden_size + self.data_depth, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(self.hidden_size),
            self._conv2d(self.hidden_size, self.hidden_size),
            nn.Tanh(),
        )

    def _conv2d(self, in_channels, out_channels):
        return nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1
        )

    def forward(self, image, data):
        x = self.features(image)
        x = torch.cat([x] + [data], dim=1)
        x = self.layers(x)

        return x


class BasicEncoder(nn.Module):
    """
    The BasicEncoder module takes a cover image + embedded data,
        and produces a list of steganographic images.

    Input:
        image: (N, 3, H, W)
        data: (N, data_depth, H, W)
    Output: [(N, 3, H, W)] * iters
    """
    def __init__(self, data_depth, hidden_size, iters=15, kenet_weight=0, xunet_weight=0):
        super(BasicEncoder, self).__init__()
        self.criterion = BCEWithLogitsLoss(reduction="sum")
        self.iters = iters

        assert hidden_size % 2 == 0
        self.hdim = self.cdim = hidden_size // 2

        self.cnet = ContextEncoder(data_depth, hidden_size)
        self.update_block = BasicUpdateBlock(self.hdim)
        self.set_kenet(kenet_weight)
        self.set_xunet(xunet_weight)

    def set_kenet(self, kenet_weight):
        self.kenet_weight = kenet_weight
        if kenet_weight > 0:
            print("Loading KeNet from checkpoints/kenet.pth.tar")
            self.kenet = KeNet()
            self.kenet.load_state_dict(torch.load("checkpoints/kenet.pth.tar", map_location="cuda")["state_dict"])
            self.kenet.cuda()
            self.kenet.eval()

    def kenet_loss(self, x):
        if self.kenet_weight <= 0:
            return 0
        x = RandomCrop(224)((x+1) * 127.5)
        h, w = x.shape[-2:]
        ch, cw, h0, w0 = h, w, 0, 0
        cw = cw & ~1
        x = [
            x[..., h0:h0 + ch, w0:w0 + cw // 2],
            x[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
        outputs, _, _ = self.kenet(*x)
        batch_size = outputs.shape[0]
        loss = 0
        for i in range(batch_size):
            if outputs[i, 1] > outputs[i, 0]:
                loss += outputs[i, 1]
        return (loss * self.kenet_weight) / batch_size

    def _kenet_loss(self, image):
        if self.kenet_weight <= 0:
            return 0
        if isinstance(image, list):
            gamma = 0.8
            weights = [gamma ** x for x in range(len(image)-1, -1, -1)]
            loss = 0
            for w, x in zip(weights, image):
                loss += self.kenet_loss(x) * w
            return loss
        else:
            return self.kenet_loss(image)

    def _kenet_detect(self, x):
        """Returns 1 if the image is detected as steganographic, 0 otherwise."""
        x = RandomCrop(224)((x+1) * 127.5)
        h, w = x.shape[-2:]
        ch, cw, h0, w0 = h, w, 0, 0
        cw = cw & ~1
        x = [
            x[..., h0:h0 + ch, w0:w0 + cw // 2],
            x[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
        outputs, _, _ = self.kenet(*x)
        batch_size = outputs.shape[0]
        assert batch_size == 1
        return float(outputs[0, 1] > outputs[0, 0])

    def kenet_detect(self, image):
        if isinstance(image, list):
            return [self._kenet_detect(x) for x in image]
        else:
            return self._kenet_detect(image)

    def set_xunet(self, xunet_weight):
        self.xunet_weight = xunet_weight
        if xunet_weight > 0:
            print("Loading XuNet from checkpoints/xunet.pth.tar")
            self.xunet = XuNet()
            self.xunet.load_state_dict(torch.load("checkpoints/xunet.pth.tar", map_location="cuda")["state_dict"])
            self.xunet.cuda()
            self.xunet.eval()

    def xunet_loss(self, x):
        if self.xunet_weight <= 0:
            return 0
        x = ((x+1) * 127.5)
        outputs = self.xunet(x)
        batch_size = outputs.shape[0]
        loss = 0
        for i in range(batch_size):
            if outputs[i, 1] > outputs[i, 0]:
                loss += outputs[i, 1]
        return (loss * self.xunet_weight) / batch_size

    def _xunet_detect(self, x):
        """Returns 1 if the image is detected as steganographic, 0 otherwise."""
        x = ((x+1) * 127.5)
        outputs = self.xunet(x)
        batch_size = outputs.shape[0]
        assert batch_size == 1
        return float(outputs[0, 1] > outputs[0, 0])

    def xunet_detect(self, image):
        if isinstance(image, list):
            return [self._xunet_detect(x) for x in image]
        else:
            return self._xunet_detect(image)

    def corr_fn(self, x, data):
        with torch.enable_grad():
            x.requires_grad = True
            loss = self.criterion(self.decoder(x), data)
            if self.kenet_weight > 0:
                # KeNet has a random crop step, sample multiple times help avoid detection
                for _ in range(6):
                    loss += self.kenet_loss(x)
            if self.xunet_weight > 0:
                loss += self.xunet_loss(x)
            loss.backward()
            grad = x.grad.clone().detach()
            x.requires_grad = False
        return grad

    def forward(self, image, data, init_noise=False, verbose=False):
        cnet = self.cnet(image, data)
        net, inp = torch.split(cnet, [self.hdim, self.cdim], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        perturb = image.clone()
        if init_noise:
            perturb = perturb + (torch.randn(image.size()) * 0.05).to(perturb)
       
        predictions = []
        corrs = []
        noises = []
        step_size = 1.0
        for itr in range(self.iters):
            perturb = perturb.detach()
            corr = self.corr_fn(perturb, data)  # index correlation volume
            noise = perturb - image
            net, delta_noise = self.update_block(net, inp, corr, noise)

            perturb = perturb + delta_noise * step_size
            perturb = torch.clamp(perturb, -1, 1)
            if self.constraint is not None:
                perturb = torch.clamp(perturb, image - self.constraint, image + self.constraint)
            predictions.append(perturb)
            if verbose:
                corrs.append(corr.detach().cpu())
                noises.append(noise.detach().cpu())

        return predictions, corrs, noises
