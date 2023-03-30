import numpy as np
import torch
from torch.optim import LBFGS

from .utils import to_np_img


def solve_lbfgs(
    model,
    image,
    payload,
    eps=0.3,
    steps=2000,
    max_iter=10,
    alpha=1,
    quantize=True
):
    image = torch.FloatTensor(image.astype(np.float32) / 127.5 - 1).permute(2, 0, 1).unsqueeze(0).to("cuda")
    criterion = torch.nn.BCEWithLogitsLoss(reduction="sum")
    adv_image = image.clone().detach()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for i in range(steps // max_iter):
        adv_image.requires_grad = True
        optimizer = LBFGS([adv_image], lr=alpha, max_iter=max_iter)

        def closure():
            outputs = model(adv_image)
            loss = criterion(outputs, payload)

            optimizer.zero_grad()
            loss.backward()
            return loss

        optimizer.step(closure)
        delta = torch.clamp(adv_image - image, min=-eps, max=eps)
        adv_image = torch.clamp(image + delta, min=-1, max=1).detach().clone()

        adv_image_q = adv_image.clone()
        if quantize:
            adv_image_q = ((adv_image_q + 1.0) * 127.5).long()
            adv_image_q = adv_image_q.float() / 127.5 - 1.0

        err = float(1 - (model(adv_image_q) >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel())
        if err < 0.0005:
            eps = 0.7
        if err == 0:
            break

    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end)), to_np_img(adv_image_q[0])