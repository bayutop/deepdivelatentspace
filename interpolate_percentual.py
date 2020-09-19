import argparse
import math
import torch
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm
import numpy as np


def slerp(val, low, high):
    """
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    """
    omega = np.arccos(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


def interpolate(args, generator, l1, l2, n1, n2):
    with torch.no_grad():
        generator.eval()
        slice_l1 = l1[0, :]
        slice_l2 = l2[0, :]

        percent = args.percent
        # interp_vals = np.linspace(1./args.n_frames, 1, num=args.n_frames)
        latent_interp = slerp(percent, slice_l1.cpu().numpy(), slice_l2.cpu().numpy())

        interpolated = torch.tensor(latent_interp)

        input = interpolated.view(1, 512)
        input = input.to("cuda")
        image, _ = generator([input], input_is_latent=True, noise=n1)


        if not os.path.exists("interpolation_percentual"):
            os.makedirs("interpolation_percentual")
        torch.save({
            "latent": input,
            "noise": n1,
            "noises": n1,
        }, "interpolation_percentual/result.pt")



        utils.save_image(
            image,
            "interpolation_percentual/result.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--n_frames", type=int, default=100)
    parser.add_argument("--percent", type=float, default=.5)
    parser.add_argument("--ckpt", type=str, default="ckpt/stylegan2-ffhq.pt")
    parser.add_argument("--latent1", type=str, default="")
    parser.add_argument("--latent2", type=str, default="")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    generator = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    checkpoint = torch.load(args.ckpt)
    generator.load_state_dict(checkpoint["g_ema"])

    l1 = torch.load(args.latent1)["latent"]
    l2 = torch.load(args.latent2)["latent"]

    n1 = torch.load(args.latent1)["noises"]
    n2 = torch.load(args.latent2)["noises"]

    interpolate(args, generator, l1, l2, n1, n2)
