import numpy as np
import os
from datetime import datetime
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils

from pdb import set_trace as st


CKPT_PATHS = {
    'BEGAN': '/deep/u/fredlu/coord_trans/celeba_gan/BEGAN/gen_430000.pth',
    'WGAN': '/deep/u/fredlu/coord_trans/celeba_gan/WGAN/wgan_celeba_64.pth',
    'StyleGAN_1024': '/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/karras2019stylegan-ffhq-1024x1024.for_g_all.pt',
    'StyleGAN_64': '/deep/u/fredlu/coord_trans/celeba_gan/StyleGAN/stylegan_celeba_64_g_all.pt'
}


class ModelOpts(object):
    """ Mock argparser for initializing GAN models that require an opt.

    Parameter help:
        BEGAN
            nc: no. channels for conv layers
            b_size: batch size
            h: latent input dimension
            tanh: not actually used
            scale_size: not actually used
    """
    def __init__(self, nc=64, b_size=16, h=64, tanh=1, scale_size=64, **kwargs):
        self.nc = nc
        self.b_size = b_size
        self.h = h
        self.tanh = tanh
        self.scale_size = scale_size
        self.__dict__.update(kwargs)


def load_began_decoder():
    from .BEGAN.models import Decoder

    ckpt_path = CKPT_PATHS['BEGAN']
    decoder_opts = ModelOpts()

    decoder = Decoder(decoder_opts)
    decoder.load_state_dict(torch.load(ckpt_path))

    return decoder


def load_wgan_decoder():
    from .WGAN.wgan import GoodGenerator

    ckpt_path = CKPT_PATHS['WGAN']

    decoder = GoodGenerator()
    decoder.load_state_dict(torch.load(ckpt_path))

    return decoder


def load_stylegan_decoder(px=1024, dataset='ffhq'):
    from .StyleGAN.model import G_mapping, G_synthesis

    ckpt_path = CKPT_PATHS[f'StyleGAN_{px}']

    if px == 64:
        decoder = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            #('truncation', Truncation(avg_latent)),
            ('g_synthesis', G_synthesis(dlatent_size=512,
                                        resolution=64,
                                        blur_filter=None,
                                        fmap_base=8192,
                                        fmap_decay=1.0,
                                        use_styles=True,
                                        const_input_layer=True,
                                        use_noise=True,
                                        randomize_noise=True,
                                        nonlinearity='lrelu',
                                        use_wscale=True))
        ]))
    elif px == 1024 and dataset == 'ffhq':
        decoder = nn.Sequential(OrderedDict([
            ('g_mapping', G_mapping()),
            #('truncation', Truncation(avg_latent)),
            ('g_synthesis', G_synthesis())
        ]))
    else:
        raise Exception('No SyleGAN for specified resolution and dataset')

    decoder.load_state_dict(torch.load(ckpt_path))
    return decoder


def try_decoder(decoder, model='BEGAN', num_ims=10, h=64, output_dir='./'):
    """ Generate some images from the decoders with some random normal input
    """
    name = datetime.now().strftime('%b%d_%H%M%S')

    z = torch.FloatTensor(num_ims, h)
    z.data.normal_(mean=0, std=1)

    if model == 'BEGAN':
        gen_z = decoder(z)
    if model == 'WGAN':
        gen_z = decoder(z)
        # gen_z = gen_z.view(num_ims, 3, 64, 64)
    elif model == 'StyleGAN':
        gen_z = decoder(z)
        gen_z = (gen_z.clamp(-1, 1) + 1) / 2.0

    vutils.save_image(
        gen_z.data,
        f'{output_dir}/{model}_{name}.png',
        nrow=(num_ims // 5) + 1,
        normalize=True)


if __name__ == '__main__':
    output_dir = './'

    decoder = load_began_decoder()
    decoder.eval()
    try_decoder(decoder, 'BEGAN', 20, 64, output_dir)

    decoder = load_wgan_decoder()
    decoder.eval()
    try_decoder(decoder, 'WGAN', 20, 128, output_dir)

    decoder = load_stylegan_decoder(px=1024)
    decoder.eval()
    try_decoder(decoder, 'StyleGAN', 4, 512, output_dir)

    decoder = load_stylegan_decoder(px=64)
    decoder.eval()
    try_decoder(decoder, 'StyleGAN', 6, 512, output_dir)
