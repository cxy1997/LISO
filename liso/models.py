import gc
import inspect
import os

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.optim import Adam, SGD
from tqdm import tqdm

from .jpeg_layer import JPEG_Layer
from .utils import calc_psnr, calc_ssim, to_np_img


METRIC_FIELDS = [
    'val.encoder_mse',
    'val.decoder_loss',
    'val.decoder_acc',
    'val.cover_score',
    'val.generated_score',
    'val.ssim',
    'val.psnr',
    'val.bpp',
    'train.encoder_mse',
    'train.decoder_loss',
    'train.decoder_acc',
    'train.cover_score',
    'train.generated_score',
]


def seq_loss(loss_func, generated, target, gamma=0.8, normalize=False):
    weights = [gamma ** x for x in range(len(generated)-1, -1, -1)]
    loss = 0
    for w, x in zip(weights, generated):
        loss += loss_func(x, target) * w
    if normalize:
        loss /= sum(weights)
    return loss


class LISO(object):
    def _get_instance(self, class_or_instance, kwargs):
        """Returns an instance of the class"""

        if not inspect.isclass(class_or_instance):
            return class_or_instance

        argspec = inspect.getfullargspec(class_or_instance.__init__).args
        argspec.remove('self')
        init_args = {arg: kwargs[arg] for arg in argspec}

        return class_or_instance(**init_args)

    def set_device(self, cuda=True):
        """Sets the torch device depending on whether cuda is available or not."""
        if cuda and torch.cuda.is_available():
            self.cuda = True
            self.device = torch.device('cuda')
        else:
            self.cuda = False
            self.device = torch.device('cpu')

        if not cuda:
            print('Using CPU device')
        elif not self.cuda:
            print('CUDA is not available. Defaulting to CPU device')
        else:
            print('Using CUDA device')

        self.encoder.to(self.device)
        self.decoder.to(self.device)
        if not self.no_critic:
            self.critic.to(self.device)

    def __init__(self, data_depth, encoder, decoder, critic, lr=1e-4, opt="adam", jpeg=False, no_critic=False,
                 cuda=True, verbose=True, extra_verbose=True, **kwargs):

        self.verbose = verbose
        self.extra_verbose = extra_verbose
        self.lr = lr
        self.opt = opt

        self.data_depth = data_depth
        self.jpeg = jpeg
        self.no_critic = no_critic

        print("data_depth", self.data_depth)
        kwargs['data_depth'] = data_depth
        self.decoder = self._get_instance(decoder, kwargs)
        if not self.no_critic:
            self.critic = self._get_instance(critic, kwargs)
        self.encoder = self._get_instance(encoder, kwargs)
        self.set_device(cuda)

        self.encoder.decoder = self.decoder

        self.critic_optimizer = None
        self.decoder_optimizer = None

        # Misc
        self.fit_metrics = None
        self.history = list()

    def _decoder(self, x):
        if self.jpeg:
            x = JPEG_Layer()(x)
        return self.decoder(x)

    def _random_data(self, cover):
        """Generate random data ready to be hidden inside the cover image.

        Args:
            cover (N, 3, H, W): Images to use as cover.

        Returns:
            payload (N, bits, H, W): Secret message to be concealed in cover images.
        """
        N, _, H, W = cover.size()
        return torch.zeros((N, self.data_depth, H, W), device=self.device).random_(0, 2)

    def _encode_decode(self, cover, quantize=False, payload=None, init_noise=False, verbose=False):
        if payload is None:
            payload = self._random_data(cover)
        generated, grads, ptbs = self.encoder(cover, payload, init_noise=init_noise, verbose=verbose)
        if quantize and not self.jpeg:
            for i in range(len(generated)):
                generated[i] = (255.0 * (generated[i] + 1.0) / 2.0).long()
                generated[i] = torch.clamp(generated[i], 0, 255)
                generated[i] = 2.0 * generated[i].float() / 255.0 - 1.0

        decoded = [self._decoder(x) for x in generated]

        return generated, payload, decoded, grads, ptbs

    def _critic(self, image):
        if isinstance(image, list):
            gamma = 0.8
            weights = [gamma ** x for x in range(len(image)-1, -1, -1)]
            score = 0
            for w, x in zip(weights, image):
                score += torch.mean(self.critic(x)) * w
            return score / sum(weights)
        else:
            return torch.mean(self.critic(image))

    def _get_optimizers(self):
        _dec_list = list(self.decoder.parameters()) + list(self.encoder.parameters())
        opt_cls = Adam if self.opt == "adam" else SGD
        if not self.no_critic:
            critic_optimizer = opt_cls(self.critic.parameters(), lr=self.lr)
        else:
            critic_optimizer = None
        decoder_optimizer = opt_cls(_dec_list, lr=self.lr)

        return critic_optimizer, decoder_optimizer

    def _fit_critic(self, train, metrics):
        print("Training critic.")
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            payload = self._random_data(cover)
            generated, _, _ = self.encoder(cover, payload)
            cover_score = self._critic(cover)
            generated_score = self._critic(generated)

            self.critic_optimizer.zero_grad()
            (cover_score - generated_score).backward(retain_graph=False)
            self.critic_optimizer.step()

            for p in self.critic.parameters():
                p.data.clamp_(-0.1, 0.1)

            metrics['train.cover_score'].append(cover_score.item())
            metrics['train.generated_score'].append(generated_score.item())

    def _fit_coders(self, train, metrics, finetune=False):
        print("Training encoder & decoder.")
        for cover, _ in tqdm(train, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            generated, payload, decoded, _, _ = self._encode_decode(cover)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)
            if not self.no_critic:
                generated_score = self._critic(generated)
                detection_score = self.encoder._kenet_loss(generated)
            else:
                generated_score = detection_score = 0

            self.decoder_optimizer.zero_grad()
            if finetune:
                decoder_loss.backward()
            else:
                (self.mse_weight * encoder_mse + decoder_loss + generated_score + detection_score).backward()
            self.decoder_optimizer.step()

            metrics['train.encoder_mse'].append(encoder_mse.item())
            metrics['train.decoder_loss'].append(decoder_loss.item())
            metrics['train.decoder_acc'].append(decoder_acc.item())

    def _coding_scores(self, cover, generated, payload, decoded):
        encoder_mse = seq_loss(mse_loss, generated, cover, gamma=0.8)
        decoder_loss = seq_loss(binary_cross_entropy_with_logits, decoded, payload, gamma=0.8)
        decoder_acc = [(x >= 0.0).eq(payload >= 0.5).sum().float() / payload.numel() for x in decoded]
        
        if self.extra_verbose:
            print(f"  encoder_mse {encoder_mse:0.4f}, decoder_loss {decoder_loss:0.4f}")
            print("   decoder_acc " + ", ".join([f"{x*100:0.2f}%" for x in decoder_acc]))

        return encoder_mse, decoder_loss, max(decoder_acc)

    def _validate(self, validate, metrics):
        print("Validating.")
        for cover, _ in tqdm(validate, disable=not self.verbose):
            gc.collect()
            cover = cover.to(self.device)
            with torch.no_grad():
                generated, payload, decoded, _, _ = self._encode_decode(cover, quantize=True)
            encoder_mse, decoder_loss, decoder_acc = self._coding_scores(cover, generated, payload, decoded)

            if not self.no_critic:
                generated_score = self._critic(generated)
                cover_score = self._critic(cover)
            else:
                generated_score = torch.tensor(0)
                cover_score = torch.tensor(0)

            metrics['val.encoder_mse'].append(encoder_mse.item())
            metrics['val.decoder_loss'].append(decoder_loss.item())
            metrics['val.decoder_acc'].append(decoder_acc.item())
            metrics['val.cover_score'].append(cover_score.item())
            metrics['val.generated_score'].append(generated_score.item())
            metrics['val.ssim'].append(
                np.mean([
                    calc_ssim(
                        to_np_img(cover[i]),
                        to_np_img(generated[-1][i])) for i in range(cover.shape[0])
                ])
            )
            metrics['val.psnr'].append(
                np.mean([
                    calc_psnr(
                        to_np_img(cover[i]),
                        to_np_img(generated[-1][i])) for i in range(cover.shape[0])
                ])
            )
            metrics['val.bpp'].append(self.data_depth * (2 * decoder_acc.item() - 1))

    def fit(self, train, validate, save_path, epochs=5):
        """Train a new model with the given ImageLoader class."""
        print("Start training.")
        best_acc = 0
        os.makedirs(save_path, exist_ok=True)

        if self.critic_optimizer is None:
            self.critic_optimizer, self.decoder_optimizer = self._get_optimizers()
            self.epochs = 0

        # Start training
        total = self.epochs + epochs
        for epoch in range(self.epochs + 1, epochs + 1):
            # Count how many epochs we have trained for this liso
            self.epochs += 1

            metrics = {field: list() for field in METRIC_FIELDS}

            if self.verbose:
                print('Epoch {}/{}'.format(self.epochs, total))

            if not self.no_critic:
                self._fit_critic(train, metrics)
            self._fit_coders(train, metrics)
            # self.save(os.path.join(save_path, f"{epoch}.steg"))
            self.save(os.path.join(save_path, f"latest.steg"))

            self._validate(validate, metrics)

            self.fit_metrics = {k: sum(v) / max(len(v), 1) for k, v in metrics.items()}
            self.fit_metrics['epoch'] = epoch
            if self.fit_metrics["val.decoder_acc"] > best_acc:
                best_acc = self.fit_metrics["val.decoder_acc"]
                self.save(os.path.join(save_path, f"best.steg"))
            print(self.fit_metrics)
            with open(os.path.join(os.path.dirname(save_path), "log.txt"), "a") as f:
                if epoch == 1:
                    f.write(", ".join(["epoch"] + list(self.fit_metrics.keys())) + "\n")
                f.write(", ".join(map(str, [epoch] + list(self.fit_metrics.values()))) + "\n")

            # Empty cuda cache (this may help for memory leaks)
            if self.cuda:
                torch.cuda.empty_cache()
            gc.collect()

    def save(self, path):
        """Save the fitted model in the given path. Raises an exception if there is no model."""
        torch.save(self, path)

    @classmethod
    def load(cls, path=None, cuda=True, verbose=False):
        """Loads an instance of SteganoGAN for the given architecture (default pretrained models)
        or loads a pretrained model from a given path.

        Args:
            architecture(str): Name of a pretrained model to be loaded from the default models.
            path(str): Path to custom pretrained model. *Architecture must be None.
            cuda(bool): Force loaded model to use cuda (if available).
            verbose(bool): Force loaded model to use or not verbose.
        """
        model = torch.load(path, map_location='cpu')
        model.verbose = verbose
        if not hasattr(model, 'jpeg'):
            model.jpeg = False
        if not hasattr(model.encoder, 'kenet_weight'):
            model.encoder.kenet_weight = 0
        if not hasattr(model.encoder, 'xunet_weight'):
            model.encoder.xunet_weight = 0

        model.set_device(cuda)
        return model
