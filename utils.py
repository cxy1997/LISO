import os
import numpy as np
from liso.loader import DataLoader


def get_path(args):
    path = f"logs/{args.dataset}/{args.bits}_bits"
    if args.epochs != 100:
        path = f"{path}_{args.epochs}_epochs"
    if args.iters != 15:
        path = f"{path}_{args.iters}_iters"
    if args.hidden_size != 32:
        path = f"{path}_{args.hidden_size}_hs"
    if args.random_crop != 360:
        path = f"{path}_{args.random_crop}_crop"
    if args.mse_weight != 1.0:
        path = f"{path}_{args.mse_weight}_mse"
    if args.jpeg:
        path = f"{path}_jpeg80"
    if args.kenet_weight > 0:
        path = f"{path}_{args.kenet_weight}_kenet"
    if args.xunet_weight > 0:
        path = f"{path}_{args.xunet_weight}_xunet"
    if args.dense_decoder:
        path = f"{path}_dense"
    if args.step_size != 1.0:
        path = f"{path}_{args.step_size}x_step"
    if args.seed is not None:
        path = f"{path}_{args.seed}_seed"
    if args.lr != 1e-4:
        path = f"{path}_{args.lr}_lr"
    if args.opt != "adam":
        path = f"{path}_{args.opt}"
    if args.no_critic:
        path = f"{path}_nc"

    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def get_loader(args):
    if args.eval:
        train = None
    else:
        train = DataLoader(
            f"datasets/{args.dataset}/train/",
            limit=args.limit,
            shuffle=True,
            batch_size=args.batch_size,
            train=True,
            crop_size=args.random_crop)
    validation = DataLoader(
        f"datasets/{args.dataset}/val/",
        limit=np.inf,
        shuffle=False,
        batch_size=1,
        train=False,
        crop_size=args.random_crop)
    return train, validation