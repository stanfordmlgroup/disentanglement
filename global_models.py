import numpy as np
import torchlayers as tl
import json
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.distributions.multivariate_normal import MultivariateNormal

import FrEIA.framework as Ff
import FrEIA.modules as Fm


################################################
##  Encoder
################################################

class Encoder(nn.Module):
    def __init__(self, nc, nz, width=2, use_spectral_norm=True, use_nn=True):
        super().__init__()

        self.nc = nc
        self.nz = nz
        self.width = width
        self.use_spectral_norm = use_spectral_norm
        self.use_nn = use_nn

        def _spectral_norm(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                m = spectral_norm(m)

        if use_nn:
            self.model = nn.Sequential(
                nn.Conv2d(nc, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(32 * width, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(32 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Conv2d(64 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                nn.Flatten(),
                nn.Linear(1024 * width, 128 * width), nn.LeakyReLU(),
                nn.Linear(128 * width, 2 * nz),
            )

        else:
            # Use tl - pytorch doesn't support spectral_norm with tl
            use_spectral_norm = False
            self.model = tl.Sequential(
                tl.Conv2d(32 * width, 4, 2), nn.LeakyReLU(),
                tl.Conv2d(32 * width, 4, 2), nn.LeakyReLU(),
                tl.Conv2d(64 * width, 4, 2), nn.LeakyReLU(),
                tl.Conv2d(64 * width, 4, 2), nn.LeakyReLU(),
                nn.Flatten(),
                tl.Linear(128 * width), nn.LeakyReLU(),
                tl.Linear(2 * nz)
            )

        if use_spectral_norm:
            self.model.apply(_spectral_norm)

        print("Building encoder...")

    def forward(self, x):
        h = self.model(x)

        _, size_both_params = h.shape
        size_one_param = size_both_params // 2

        mu, diag = torch.split(h, size_one_param, dim=-1)
        cov = torch.diag_embed(diag)

        return mu, cov

    def args_dict(self):
        model_args = {
                        'nc': self.nc,
                        'nz': self.nz,
                        'width': self.width,
                        'use_spectral_norm': self.use_spectral_norm,
                        'use_nn': self.use_nn,
                     }

        return model_args


################################################
##  Discriminator
################################################

class Discriminator(nn.Module):
    def __init__(self, nc, ns, width=1, use_spectral_norm=True, use_nn=True):
        super().__init__()

        self.nc = nc
        self.ns = ns

        def _spectral_norm(m):
            if type(m) == nn.Conv2d or type(m) == nn.Linear:
                m = spectral_norm(m)

        if use_nn:
            # Use nn
            self.body = nn.Sequential(
                    nn.Conv2d(nc, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                    nn.Conv2d(32 * width, 32 * width, 4, 2, 1), nn.LeakyReLU(),
                    nn.Conv2d(32 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                    nn.Conv2d(64 * width, 64 * width, 4, 2, 1), nn.LeakyReLU(),
                    nn.Flatten(),
            )

            self.aux = nn.Sequential(
                nn.Linear(ns, 128 * width), nn.LeakyReLU(),
            )

            self.head = nn.Sequential(
                nn.Linear(1152 * width, 128 * width), nn.LeakyReLU(),
                nn.Linear(128 * width, 128 * width), nn.LeakyReLU(),
                nn.Linear(128 * width, 1, bias=False)
            )

        else:
            # Use tl - pytorch doesn't support spectral_norm with tl
            use_spectral_norm = False

            self.body = tl.Sequential(
                    tl.Conv2d(32 * width, 4, 2), nn.LeakyReLU(),
                    tl.Conv2d(32 * width, 4, 2), nn.LeakyReLU(),
                    tl.Conv2d(64 * width, 4, 2), nn.LeakyReLU(),
                    tl.Conv2d(64 * width, 4, 2), nn.LeakyReLU(),
                    nn.Flatten(),
            )

            self.aux = tl.Sequential(
                tl.Linear(128 * width), nn.LeakyReLU(),
            )

            self.head = tl.Sequential(
                tl.Linear(128 * width), nn.LeakyReLU(),
                tl.Linear(128 * width), nn.LeakyReLU(),
                tl.Linear(1, bias=False)
            )

        if use_spectral_norm:
            self.body.apply(_spectral_norm)
            self.aux.apply(_spectral_norm)
            self.head.apply(_spectral_norm)


        print("Building discriminator...")

    def forward(self, x, y):
        hx = self.body(x)
        hy = self.aux(y)
        o = self.head(torch.cat((hx, hy), dim=-1))
        return o


################################################
##  Decoder /  Generator
################################################

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)


class Generator(nn.Module):
    def __init__(self, nz, nc, bn=True, use_nn=True):
        super().__init__()
        self.nz = nz
        self.nc = nc
        self.bn = bn
        self.use_nn = use_nn

        def linearblock(num_feat, in_feat=None):
            if self.use_nn:
                layers = [nn.Linear(in_feat, num_feat)]
                if self.bn:
                    layers.append(nn.BatchNorm1d(num_feat))
            else:
                layers = [tl.Linear(num_feat)]
                if self.bn:
                    layers.append(tl.BatchNorm())
            layers.append(nn.ReLU(inplace=True))
            return layers

        def deconvblock(num_feat, in_feat=None, kernel=4, stride=2, padding=1):
            if self.use_nn:
                layers = [nn.ConvTranspose2d(in_feat, num_feat, kernel_size=kernel, stride=stride, padding=padding)]
                if self.bn:
                    layers.append(nn.BatchNorm2d(num_feat))
            else:
                layers = [tl.ConvTranspose2d(num_feat, kernel_size=kernel, stride=stride, padding=padding)]
                if self.bn:
                    layers.append(tl.BatchNorm())
            layers.append(nn.LeakyReLU(inplace=True))
            return layers

        if self.use_nn:
            self.model = nn.Sequential(
                *linearblock(128, in_feat=nz),
                *linearblock(4 * 4 * 64, in_feat=128),
                View(-1, 64, 4, 4),
                *deconvblock(64, in_feat=64),
                *deconvblock(32, in_feat=64),
                *deconvblock(32, in_feat=32),
                nn.ConvTranspose2d(32, nc, 4, 2, 1),
                nn.Sigmoid(),
            )
        else:
            self.model = tl.Sequential(
                *linearblock(128, in_feat=nz),
                *linearblock(4 * 4 * 64),
                View(-1, 64, 4, 4),
                *deconvblock(64),
                *deconvblock(32),
                *deconvblock(32),
                tl.ConvTranspose2d(nc, 4, 2, 1),
                nn.Sigmoid(),
            )

        print("Building generator...")


    def forward(self, z):
        return self.model(z)


    def args_dict(self):
        model_args = {
                        'nz': self.nz,
                        'nc': self.nc,
                        'bn': self.bn,
                        'use_nn': self.use_nn,
                     }

        return model_args


class ALAEStyleGAN(nn.Module):
    def __init__(self, dataset_name, is_encoder=False):
        super().__init__()

        self.dataset_name = dataset_name
        self.is_encoder = is_encoder

        from alae.defaults import get_cfg_defaults as alae_get_cfg_defaults
        from alae.checkpointer_nologger import Checkpointer as ALAECheckpointer
        from alae.model import Model as ALAEModel
        
        cfg = alae_get_cfg_defaults()
        alae_dir = '/deep/group/sharonz/ALAE'
        if dataset_name == 'celebahq':
            config_file = f'{alae_dir}/configs/celeba-hq256.yaml'
        elif dataset_name == 'celeba':
            config_file = f'{alae_dir}/configs/celeba.yaml'
        elif dataset_name == 'ffhq':
            config_file = f'{alae_dir}/configs/ffhq.yaml'
        
        cfg.merge_from_file(config_file)
        
        # Change output dir to work with dis
        #cfg.OUTPUT_DIR = f'{alae_dir}/{cfg.OUTPUT_DIR}'
        cfg.freeze()
        self.cfg = cfg

        self.model = ALAEModel(
            startf=cfg.MODEL.START_CHANNEL_COUNT,
            layer_count=cfg.MODEL.LAYER_COUNT,
            maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
            latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
            truncation_psi=None,
            truncation_cutoff=None,
            style_mixing_prob=cfg.MODEL.STYLE_MIXING_PROB,
            mapping_layers=cfg.MODEL.MAPPING_LAYERS,
            channels=cfg.MODEL.CHANNELS,
            generator=cfg.MODEL.GENERATOR,
            encoder=cfg.MODEL.ENCODER)

        self.model.cuda(0)
        self.model.eval()
        self.model.requires_grad_(False)

        self.decoder = self.model.decoder
        self.encoder = self.model.encoder
        
        self.mapping_fl = self.model.mapping_fl
        self.dlatent_avg = self.model.dlatent_avg
        
        model_dict = {
            'discriminator_s': self.encoder,
            'generator_s': self.decoder,
            'mapping_fl_s': self.mapping_fl,
            'dlatent_avg_s': self.dlatent_avg
        }

        checkpointer = ALAECheckpointer(self.cfg,
                                        model_dict,
                                        {},
                                        save=False)
        checkpointer.load()
        self.model.eval()
        
        self.lod = self.cfg.DATASET.MAX_RESOLUTION_LEVEL - 2


    def forward(self, x):
        if self.is_encoder:
            z = self.encoder(x, self.lod, 1)
            z = z.squeeze(1)
            return z
        else:
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            image = self.model.generate(self.lod, 1, x, 1, mixing=False)
            return image


class TorchGenerator(nn.Module):
    def __init__(self, architecture, dataset_name, pretrained):
        super().__init__()

        self.architecture = architecture
        self.dataset_name = dataset_name
        self.pretrained = pretrained

        if self.dataset_name == 'celebahq':
            dataset_name = 'celebAHQ-256'

        assert dataset_name in ['celebAHQ-256', 'celebAHQ-512', 'DTD', 'celeba'], \
                f'{dataset_name} not valid for PGAN'

        self.model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                                    architecture,
                                    model_name=dataset_name,
                                    pretrained=pretrained,
                                    useGPU=True)

        self.generator = self.model.avgG
        if isinstance(self.generator, nn.DataParallel):
            # Remove DataParallel
            self.generator = self.generator.module
        self.nz = self.model.config.latentVectorDim

        print(f"Building torch generator {self.architecture}...")


    def forward(self, x):
        out = self.generator(x)
        #out_viz = out.mul(0.5).add_(0.5).clamp_(0, 1)
        #out_viz = out.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to(torch.uint8)
        return out #out_viz


    def args_dict(self):
        model_args = {
                        'architecture': self.architecture,
                        'dataset_name': self.dataset_name,
                        'pretrained': self.pretrained,
                     }

        return model_args

class CustomTorchGenerator(nn.Module):
    def __init__(self, architecture, ckpt_path):
        super().__init__()

        assert ckpt_path is not None

        self.architecture = architecture
        self.ckpt_path = ckpt_path

        self.set_config_from_ckpt_path()
        self.set_model()

        self.nz = self.model.config.latentVectorDim

        print(f"Building torch generator {self.architecture}...")


    def set_model(self):
        if self.architecture == 'PGAN':
            from models.progressive_gan import ProgressiveGAN
            self.model = ProgressiveGAN(**self.config)
        elif self.architecture == 'StyleGAN':
            from models.styleGAN import StyleGAN
            self.model = StyleGAN(**self.config)
        self.model.load(self.ckpt_path)


    def set_config_from_ckpt_path(self):
        self.config_path = self.ckpt_path.split('_s')[0] + '_train_config.json'

        with open(self.config_path, 'rb') as file:
            self.config = json.load(file)


    def forward(self, x):
        out = self.model.test(x, getAvG=True)
        return out


    def args_dict(self):
        model_args = {
                        'architecture': self.architecture,
                        'ckpt_path': self.ckpt_path,
                        'config_path': self.config_path,
                     }

        return model_args


class CustomPretrainedGenerator(nn.Module):
    def __init__(self, architecture, dataset):
        super().__init__()

        self.architecture = architecture
        self.dataset = dataset

        self.set_model()

    def set_model(self):
        if self.architecture == 'BEGAN':
            assert self.dataset == 'celeba'
            from celeba_gan.loaders import load_began_decoder
            self.model = load_began_decoder()

        elif self.architecture == 'WGAN':
            assert self.dataset == 'celeba'
            from celeba_gan.loaders import load_wgan_decoder
            self.model = load_wgan_decoder()

        elif self.architecture == 'StyleGAN':
            assert self.dataset == 'ffhq'
            from celeba_gan.loaders import load_stylegan_decoder
            self.model = load_stylegan_decoder()

    def forward(self, x):
        return self.model(x)


################################################
##  Transcoder
################################################

# https://github.com/ANLGBOY/MADE-with-PyTorch/blob/master/made.py

class MADE(nn.Module):
    def __init__(self, in_channels, out_channels):
        print(in_channels, out_channels)
        super().__init__()
        # companion layers are for Connectivity-agnostic training
        # direct layer is for direct connection between input and output
        self.z1_dim = 1024
        self.in_channels = in_channels
        self.out_channels = out_channels # Gets overwritten, but needs to be passed
        self.z2_dim = out_channels
        out_channels = in_channels # Out channels here means of the autoencoded representation
        self.fc1 = nn.Linear(in_channels, self.z1_dim)
        self.fc1_companion = nn.Linear(in_channels, self.z1_dim)
        self.fc2 = nn.Linear(self.z1_dim, self.z2_dim)
        self.fc2_companion = nn.Linear(self.z1_dim, self.z2_dim)
        self.fc3 = nn.Linear(self.z2_dim, self.z1_dim)
        self.fc3_conpanion = nn.Linear(self.z2_dim, self.z1_dim)
        self.fc4 = nn.Linear(self.z1_dim, out_channels)
        self.fc4_companion = nn.Linear(self.z1_dim, out_channels)
        self.fc_direct = nn.Linear(in_channels, out_channels)

        self.MW0 = np.zeros((self.z1_dim, in_channels))
        self.MW1 = np.zeros((self.z2_dim, self.z1_dim))
        self.MW2 = np.zeros((self.z1_dim, self.z2_dim))
        self.MV = np.zeros((out_channels, self.z1_dim))
        self.MA = np.zeros((out_channels, in_channels))

        m0 = list(np.ones(10, dtype=int)) + list(range(1, 785))
        m1 = random.choices(range(1, in_channels), k=self.z1_dim)
        m2 = random.choices(range(min(m1), in_channels), k=self.z2_dim)
        m3 = random.choices(range(min(m2), in_channels), k=self.z1_dim)

        for i in range(self.z1_dim):
            for j in range(in_channels):
                self.MW0[i][j] = 1 if m1[i] >= m0[j] else 0

        for i in range(self.z2_dim):
            for j in range(self.z1_dim):
                self.MW1[i][j] = 1 if m2[i] >= m1[j] else 0

        for i in range(self.z1_dim):
            for j in range(self.z2_dim):
                self.MW2[i][j] = 1 if m3[i] >= m2[j] else 0

        for i in range(out_channels):
            for j in range(self.z2_dim):
                self.MV[i][j] = 1 if m0[i] > m3[j] else 0

        for i in range(out_channels):
            for j in range(in_channels):
                self.MA[i][j] = 1 if m0[i] > m0[j] else 0

        self.MW0 = torch.from_numpy(self.MW0).float()
        self.MW1 = torch.from_numpy(self.MW1).float()
        self.MW2 = torch.from_numpy(self.MW2).float()
        self.MV = torch.from_numpy(self.MV).float()
        self.MA = torch.from_numpy(self.MA).float()


    def forward(self, x):
        masked_fc1 = self.fc1.weight * self.MW0
        masked_fc2 = self.fc2.weight * self.MW1
        masked_fc1_companion = self.fc1_companion.weight * self.MW0
        masked_fc2_companion = self.fc2_companion.weight * self.MW1
        h1 = F.relu(F.linear(x, masked_fc1, self.fc1.bias) +
                    F.linear(torch.ones_like(x), masked_fc1_companion, self.fc1_companion.bias))
        h2 = F.relu(F.linear(h1, masked_fc2, self.fc2.bias) +
                    F.linear(torch.ones_like(h1), masked_fc2_companion, self.fc2_companion.bias))
        return h2

    def reverse(self, h2, x=None):
        masked_fc3 = self.fc3.weight * self.MW2
        masked_fc4 = self.fc4.weight * self.MV
        masked_fc3_companion = self.fc3_conpanion.weight * self.MW2
        masked_fc4_companion = self.fc4_companion.weight * self.MV
        masked_fc_direct = self.fc_direct.weight * self.MA
        h3 = F.relu(F.linear(h2, masked_fc3, self.fc3.bias) +
                    F.linear(torch.ones_like(h2), masked_fc3_companion, self.fc3_conpanion.bias))
        recon_x = F.linear(h3, masked_fc4, self.fc4.bias) +\
                    F.linear(torch.ones_like(h3), masked_fc4_companion, self.fc4_companion.bias)
        if x is not None:
            recon_x = recon_x + F.linear(x, masked_fc_direct, self.fc_direct.bias)
        return recon_x

    def args_dict(self):
        model_args = {
                        'in_channels': self.in_channels,
                        'out_channels': self.out_channels,
                     }

        return model_args


class MADE_Rev(nn.Module):
    def __init__(self, made):
        super().__init__()
        self.made = made

    def forward(self, h2, x=None):
        return self.made.reverse(h2, x)

    def args_dict(self):
        model_args = self.made.args_dict()

        return model_args


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.model = nn.Sequential(
            nn.Linear(in_channels, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, out_channels)
        )

        print(f"Building basic MLP with {in_channels} in and {out_channels} out...")


    def forward(self, x):
        return self.model(x)


    def args_dict(self):
        model_args = {
                        'in_channels': self.in_channels,
                        'out_channels': self.out_channels,
                     }

        return model_args


class MLPCycleTranscoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.fwd_model = nn.Sequential(
            nn.Linear(in_channels, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, out_channels)
        )

        self.rev_model = nn.Sequential(
            nn.Linear(in_channels, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, 100), nn.LeakyReLU(),
            nn.Linear(100, out_channels)
        )

    def forward(self, x):
        self.trans_s = self.fwd_model(x)
        trans_s_soft = softmax_classes(self.trans_s, ncls)
        trans_z_cycle = self.rev_model(trans_s_soft)

        s_fake = self.trans_s.detach()
        s_fake_soft = softmax_classes(s_fake, ncls)
        trans_z = transcoder_rev(s_fake_soft)
        trans_s_cycle = transcoder(trans_z)

    def compute_loss(self):
         # forward mapping loss
            forward_error = torch.zeros(1).to(args.device)
            prev = 0
            for s in range(ns):
                num_classes = ncls[s]
                y_hat = trans_s[:, prev : prev + num_classes]
                if args.use_both_datasets:
                    y = label.long()[:, s]
                else:
                    y = label[f'factor_{s}']
                    y = y.argmax(1)
                forward_error += ce(y_hat, y) / np.log(num_classes)
                prev += num_classes

            # forward cycle loss
            forward_cycle_error = torch.zeros(1).to(args.device)
            forward_cycle_error += mse(trans_z_cycle, z)

            # reverse mapping loss
            rev_error = torch.zeros(1).to(args.device)
            rev_error += mse(trans_z, z)

            # reverse cycle loss
            rev_cycle_error = torch.zeros(1).to(args.device)
            prev = 0
            for s in range(ns):
                num_classes = ncls[s]
                y_hat, y = \
                    trans_s_cycle[:, prev : prev + num_classes], s_fake[:, prev : prev + num_classes]
                rev_cycle_error += mse(y_hat, y) / num_classes
                prev += num_classes

            lambda_forward = 0.1
            lambda_rev = 0.1
            total_error = forward_error + rev_error + lambda_forward * forward_cycle_error + lambda_rev * rev_cycle_error
            total_error.backward()

    def optimizer_step(self):
        optimizer_transcoder.step()
        optimizer_transcoder_rev.step()



class MLPTranscoder(nn.Module):
    def __init__(self, nz, ns, num_classes_per_head):
        super().__init__()

        self.nz = nz
        self.ns = ns
        self.num_heads = len(num_classes_per_head)
        self.num_classes_per_head = num_classes_per_head

        self.model = nn.Sequential(
            nn.Linear(nz, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
        )

        self.heads = []
        for i in range(self.num_heads):
            self.__setattr__(f'factor_{i}', nn.Linear(1024, self.num_classes_per_head[i]))

        self.reverse_model = nn.Sequential(
            nn.Linear(1024 // self.num_heads * self.num_heads, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, 1024), nn.LeakyReLU(),
            nn.Linear(1024, nz), nn.LeakyReLU(),
        )

        for i in range(self.num_heads):
            self.__setattr__(f'reverse_factor_{i}', nn.Linear(self.num_classes_per_head[i], 1024 // self.num_heads))

        print("Building transcoder...")


    def forward(self, x, reverse=False):
        if not reverse:
            out = self.model(x)
            return {f'factor_{i}': self.__getattr__(f'factor_{i}')(out) for i in range(self.num_heads)}
        else:
            # Input is list of lists
            reverse_factors = []
            prev_idx = 0
            for i in range(self.num_heads):
                num_classes = self.num_classes_per_head[i]
                reverse_head = x[:, prev_idx : prev_idx + num_classes]
                reverse_factor = self.__getattr__(f'reverse_factor_{i}')(reverse_head)
                reverse_factors.append(reverse_factor)
                prev_idx += num_classes

            # Stack them to combine in body of model
            stacked_reverse_factors = torch.cat(reverse_factors, dim=1)

            out = self.reverse_model(stacked_reverse_factors)
            return out


    def args_dict(self):
        model_args = {
                        'nz': self.nz,
                        'ns': self.ns,
                        'num_classes_per_head': self.num_classes_per_head,
                     }

        return model_args


def subnet_lin(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 128), nn.ReLU(),
                        nn.Linear(128,  c_out))


def subnet_fc(c_in, c_out):
    return nn.Sequential(nn.Linear(c_in, 512), nn.ReLU(),
                         nn.Linear(512,  c_out))


class Transcoder(nn.Module):
    def __init__(self, nz, ns, num_classes_per_head):
        # num_classes_per_head is a list, where each elt is the num classes for the head at that index
        super().__init__()

        self.nz = nz
        self.ns = ns
        self.num_classes_per_head = num_classes_per_head

        nodes = [Ff.InputNode(max(self.num_classes_per_head), name='input')]

        # https://github.com/VLL-HD/FrEIA
        # Use a loop to produce a chain of coupling blocks
        for k in range(1):
            nodes.append(Ff.Node(nodes[-1],
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_fc, 'clamp':2.0},
                                name=F'coupling_one_{k}'))
            nodes.append(Ff.Node(nodes[-1],
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_one_{k}'))

        final_node = nodes[-1].out0
        for k in range(self.ns):
            nodes.append(Ff.Node(final_node,
                                Fm.GLOWCouplingBlock,
                                {'subnet_constructor':subnet_lin, 'clamp':2.0},
                                name=F'coupling_two_{k}'))
            nodes.append(Ff.Node(nodes[-1], #
                                Fm.PermuteRandom,
                                {'seed':k},
                                name=F'permute_two_{k}'))
            cur_classes = self.num_classes_per_head[k]
            nodes.append(Ff.Node(nodes[-1], #
                                Fm.Split1D,
                                {'split_size_or_sections': [
                                    cur_classes,
                                    max(self.num_classes_per_head) - cur_classes
                                ], 'dim':0},
                                name=F'split_{k}'))
            nodes.append(Ff.OutputNode(nodes[-1].out0, name=f'output_{k}'))


        self.inn = Ff.ReversibleGraphNet(nodes)
        self.trainable_parameters = [p for p in self.inn.parameters() if p.requires_grad]

        print("Building transcoder...")


    def forward(self, x, detach=False, to_cpu=False, ret_jac=False):
        zero = torch.zeros(len(x), max(self.num_classes_per_head) - self.nz, device='cuda')
        x = torch.cat([x, zero], dim=1)
        out = self.inn(x)
        out_list = []
        for i, num_classes in enumerate(self.num_classes_per_head):
            to_add = out[i]
            if detach:
                to_add = to_add.detach()
            if to_cpu:
                to_add = to_add.cpu()
            out_list.append(to_add)
            # print(out[0].shape)

        if ret_jac:
            jac = self.inn.log_jacobian(run_forward=False)
            return {f'factor_{i}': out_list[i] for i in range(self.ns)}, jac
        return {f'factor_{i}': out_list[i] for i in range(self.ns)}

    def reverse_sample(self, z):
        inputs = []
        for i, num_classes in enumerate(self.num_classes_per_head):
            one_hot = F.one_hot(z[:, i], num_classes=num_classes).float()
            inputs.append(one_hot)
        x = self.inn(inputs, rev=True)
        return x[:, :self.nz]

    def reverse_sample_logits(self, z):
        inputs = []
        for i in range(self.ns):
            inputs.append(z[f"factor_{i}"])
        x = self.inn(inputs, rev=True)
        return x[:, :self.nz]

    def args_dict(self):
        model_args = {
                        'nz': self.nz,
                        'ns': self.ns,
                        'num_classes_per_head': self.num_classes_per_head,
                     }

        return model_args
