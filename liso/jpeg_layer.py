import torch
from io import BytesIO
from PIL import Image
import numpy as np


class JPEG_Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        batch_size = input.shape[0]
        res = []
        for i in range(batch_size):
            pil_image = Image.fromarray(((input[i].clamp(-1.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8))
            f = BytesIO()
            pil_image.save(f, format='jpeg', quality=80)  # quality level specified in paper
            jpeg_image = (np.asarray(Image.open(f)).astype(np.float32) / 127.5) - 1.0
            res.append(torch.tensor(jpeg_image).permute(2, 0, 1).unsqueeze(0).to(input.device))
        return torch.cat(res, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone()


class JPEG_Layer(torch.nn.Module):
    def __init__(self):
        super(JPEG_Layer, self).__init__()
        self.func = JPEG_Function()

    def forward(self, x):
        return self.func.apply(x)
