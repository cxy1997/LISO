import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def to_np_img(x, dtype=np.uint8):
    return ((x.clamp(-1.0, 1.0).permute(1, 2, 0).detach().cpu().numpy() + 1.0) * 127.5).astype(np.uint8).astype(dtype)


def calc_psnr(img1, img2):
    return peak_signal_noise_ratio(
        img1.astype(np.float32),
        img2.astype(np.float32),
        data_range=255)


def calc_ssim(img1, img2):
    return structural_similarity(
        img1.astype(np.float32),
        img2.astype(np.float32),
        data_range=255,
        multichannel=True)