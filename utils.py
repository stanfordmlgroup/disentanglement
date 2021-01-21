import os
import sys
import time
import numpy as np
import torchlayers as tl
from tqdm import tqdm
from torchsummary import summary
from scipy.stats import truncnorm
from prdc import compute_prdc

from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils
from torch.utils.data import SubsetRandomSampler
import torchvision.models as tvmodels

from datasets import ClassificationDataset

# Get noise vecs ready
def sample_noise(batch_size, nz, device, truncation=None):
    if truncation:
        z = truncation * truncnorm.rvs(-2, 2, size=(batch_size, nz))
        z = torch.from_numpy(z).to(device)
    else:
        z = torch.randn(batch_size, nz, device=device)
    return z

def visualize(images, save_path):
    gridded_images = vutils.make_grid(images, padding=2, pad_value=255, normalize=True)
    vutils.save_image(gridded_images, save_path)
    print(f'Saved images to {save_path}')

def load_models(args, ns, np, nc, ncls=None, model_types=['decoder', 'discriminator', 'encoder'], model_params=[{}, {}, {}], use_nn=True, model_ckpts=None):
    models = []
    for idx, model_type in enumerate(model_types):
        if model_ckpts is not None:
            assert len(model_ckpts) == len(model_types), \
                'Wrong number of model checkpoints. Use None if no checkpoint'
            model_ckpt = model_ckpts[idx]
        else:
            model_ckpt = None
        model_param = model_params[idx]
        model = load_model(args, ns, np, nc, ncls, model_type, model_param, model_ckpt, use_nn)
        models.append(model)
    if len(models) == 1:
        models = models[0]
    return models

def load_model(args, ns, np, nc, ncls, model_type, model_param, model_ckpt, use_nn=True):
    print(f"Loading {model_type}")
    total_classes = sum(ncls)
    if model_type == 'decoder':
        model = args.decoder_model
        if model == 'dcgan':
            from global_models import Generator
            decoder = Generator(args.nz, nc, np).to(args.device)

        elif 'vae' in model.lower():
            from disentangling_vae.disvae.models.utils.modelIO import load_model as load_vae
            assert args.dataset_name in ['celeba', 'dsprites', 'mnist', 'chairs']
            model_name = model.split('_')[0] if '_' else model
            vae_name = f'{model_name}_{args.dataset_name}'

            vae_ckpt_dir = "/deep/group/sharonz/dis/disentangling_vae/results"
            vae_dir = os.path.join(vae_ckpt_dir, vae_name)
            model = load_vae(vae_dir)
            decoder = model.decoder
            
        elif model == 'PGAN' and args.dataset_name == 'celebahq':
            pretrained = True # default pretrained True
            if 'pretrained' in model_param:
                pretrained = model_param['pretrained']
            from global_models import TorchGenerator
            decoder = TorchGenerator(architecture=model,
                                     dataset_name=args.dataset_name,
                                     pretrained=pretrained)

        elif model == 'StyleGAN' and args.dataset_name == 'celebahq':
            from global_models import ALAEStyleGAN
            decoder = ALAEStyleGAN(args.dataset_name)

        elif model in ['WGAN', 'BEGAN'] \
            or (model == 'StyleGAN' and args.dataset_name == 'ffhq'):
            from global_models import CustomPretrainedGenerator
            decoder = CustomPretrainedGenerator(architecture=model,
                                                dataset=args.dataset_name)

        else:
            raise Exception(f'{model_type} with name {model} does not exist.')

        if model_ckpt is not None and model in ['dcgan']:
            decoder.load_state_dict(torch.load(model_ckpt)["model_state"])
        decoder = nn.DataParallel(decoder, args.gpu_ids)
        return decoder.to(args.device)


    elif model_type == 'discriminator':
        model = args.discriminator_model
        if model == 'dcgan':
            from global_models import Discriminator
            discriminator = Discriminator(nc, np).to(args.device)
        else:
            raise Exception(f'{model_type} with name {model} does not exist.')

        if model_ckpt is not None:
            discriminator.load_state_dict(torch.load(model_ckpt)["model_state"])
        discriminator = nn.DataParallel(discriminator, args.gpu_ids)
        return discriminator.to(args.device)

    elif model_type == 'encoder':
        model = args.encoder_model
        if model == 'dcgan':
            raise Exception(f'No encoder in dcgan')

        elif 'vae' in model.lower():
            from disentangling_vae.disvae.models.utils.modelIO import load_model as load_vae
            model_name = model.split('_')[0] if '_' else model
            vae_name = f'{model_name}_{args.dataset_name}'

            vae_ckpt_dir = "/deep/group/sharonz/dis/disentangling_vae/results"
            vae_dir = os.path.join(vae_ckpt_dir, vae_name)
            model = load_vae(vae_dir)
            encoder = model.encoder
        
        elif model == 'StyleGAN' and args.dataset_name == 'celebahq':
            from global_models import ALAEStyleGAN
            encoder = ALAEStyleGAN(args.dataset_name, is_encoder=True)

        else:
            # NB: PGAN has no corresponding encoder
            raise Exception(f'{model_type} with name {model} does not exist.')

        if model_ckpt is not None and model != 'StyleGAN':
            encoder.load_state_dict(torch.load(model_ckpt)["model_state"])
        encoder = nn.DataParallel(encoder, args.gpu_ids)
        return encoder.to(args.device)

    elif model_type == 'transcoder':
        model = args.transcoder_model
        transcoder_rev = None
        if model == 'mlptranscoder':
            from global_models import MLPTranscoder
            transcoder = MLPTranscoder(args.nz, ns, num_classes_per_head=ncls)
        elif model == 'mlp':
            from global_models import MLP
            transcoder = MLP(args.nz, total_classes)
            transcoder_rev = MLP(total_classes, args.nz)
        elif model == 'freia':
            from global_models import Transcoder
            transcoder = Transcoder(args.nz, ns, num_classes_per_head=ncls)
        elif model == 'made':
            from global_models import MADE, MADE_Rev
            transcoder = MADE(args.nz, total_classes)
            transcoder.MW0 = transcoder.MW0.to(args.device)
            transcoder.MW1 = transcoder.MW1.to(args.device)
            transcoder.MW2 = transcoder.MW2.to(args.device)
            transcoder.MV = transcoder.MV.to(args.device)
            transcoder.MA = transcoder.MA.to(args.device)
            transcoder_rev = MADE_Rev(transcoder)
        else:
            raise Exception(f'{model_type} with name {model} does not exist.')

        transcoder = nn.DataParallel(transcoder, args.gpu_ids)
        if transcoder_rev:
            transcoder_rev = nn.DataParallel(transcoder_rev, args.gpu_ids)

            if model_ckpt is not None:
                print('Loading transcoder checkpoint')
                transcoder.load_state_dict(torch.load(model_ckpt)["model_state"], strict=False)
                if model == 'mlp':
                    print('Loading reverse transcoder checkpoint')
                    ckpt_path = Path(model_ckpt)
                    rev_ckpt = ckpt_path.parent / ckpt_path.name.replace('t_', 't_r_')
                    transcoder_rev.load_state_dict(torch.load(rev_ckpt)["model_state"], strict=False)
            return [transcoder.to(args.device), transcoder_rev.to(args.device)]
        else:
            if model_ckpt is not None:
                transcoder.load_state_dict(torch.load(model_ckpt)["model_state"], strict=False)
            return transcoder.to(args.device)

    return models

def load_optimizer(args, models):
    if not isinstance(models, list):
        models = [models]
    optimizers = []
    for model in models:
        if model is None:
            opt = None
        else:
            opt = optim.Adam(model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
        optimizers.append(opt)
    return optimizers if len(optimizers) > 1 else optimizers[0]


def save_model(args, model, optimizer, step, ckpt_paths, ckpt_name=None, save_step=True):
    ckpt_dict = {
        'ckpt_info': {'step': step},
        'model_name': model.module.__class__.__name__,
        'model_args': model.module.args_dict(),
        'model_state': model.module.to('cpu').state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    model.to(args.device)

    if not save_step:
        ckpt_filename = f'{ckpt_name}.pth.tar' if ckpt_name else 'best.pth.tar'
    elif ckpt_name:
        ckpt_filename = f'{ckpt_name}_step_{step}.pth.tar'
    else:
        ckpt_filename = f'step_{step}.pth.tar'
    ckpt_path = Path(args.ckpt_dir) / f'{ckpt_filename}'
    torch.save(ckpt_dict, ckpt_path)

    # Only keep track of ckpts if not default model - default model overrides filename
    if save_step:
        ckpt_paths.append(ckpt_path)
        if len(ckpt_paths) > args.max_ckpts:
            oldest_ckpt = ckpt_paths.pop(0)
            if os.path.exists(oldest_ckpt):
                os.remove(oldest_ckpt)

    return ckpt_paths


def softmax_classes(trans_s, ncls):
    new_trans_s = trans_s.clone()
    prev = 0
    for num_classes in ncls:
        new_trans_s[:, prev : prev + num_classes] = F.softmax(trans_s[:, prev : prev + num_classes])
        prev += num_classes
    return new_trans_s

def ce_with_probs(input_logits, target_logits):
    return -(F.softmax(target_logits) * F.log_softmax(input_logits)).sum(1).mean()

def argmax_classes(trans_s, ncls):
    new_trans_s = []
    prev = 0
    for num_classes in ncls:
        new_trans_s.append(torch.argmax(trans_s[:, prev : prev + num_classes], dim=1))
        prev += num_classes
    return torch.stack(new_trans_s, dim=1)


def get_dataset_args(args, return_factor_name_map=False):

    # Loaded just to get dataset information
    dataset = ClassificationDataset(args.dataset_name, args.batch_size)

    # Get parameters from dataset:
    #   ns: num s factors (s dim)
    #   npix: num pixels (img height)
    #   nc: num channels
    ns = dataset.ns
    if args.nz is None:
        args.nz = ns

    n_extra_z = args.nz - ns # Extra dimensions in z, beyond s_dim (ns)
    image_shape = dataset.dataset.observation_shape
    npix, nc = image_shape[0], image_shape[-1]
    ncls = dataset.dataset.factors_num_values

    if return_factor_name_map:
        factor_id2name = dataset.dataset.factor_id2name
        return dataset, ns, image_shape, npix, nc, ncls, factor_id2name
    return dataset, ns, image_shape, npix, nc, ncls, None


def labels_to_onehot(label, ns):
    onehot_s = torch.zeros(len(label), sum(ns)).cuda()
    for ii, single_label in enumerate(label):
        prev = 0
        for new_val, num_classes in zip(single_label, ns):
            onehot_s[ii, new_val + prev] = 1
            prev += num_classes
    return onehot_s


def dict_to_labels(label):
    return torch.stack([v.argmax(1) for k,v in label.items()], dim=1)


def dict_to_onehot(label, ns):
    label_stack = dict_to_labels(label)
    return labels_to_onehot(label_stack, ns)


def create_dataset_splits(args, dataset, train_split=.98, valid_split=.01, include_test_set=False):

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    dataset.random_state.shuffle(indices)

    train_idx = int(np.floor(train_split * dataset_size))
    train_indices = indices[:train_idx]

    if include_test_set:
        valid_idx = int(np.floor((train_split + valid_split) * dataset_size))
        valid_indices = indices[train_idx:valid_idx]

        test_split = 1 - train_split - valid_split
        test_indices = indices[valid_idx:]
        test_sampler = SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=args.batch_size,
                                                  sampler=test_sampler,
                                                  num_workers=args.num_workers,
                                                  )

    else:
        valid_indices = indices[train_idx:]

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=args.num_workers,
                                               )

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=args.batch_size,
                                               sampler=valid_sampler,
                                               num_workers=args.num_workers,
                                               )

    return (train_loader, valid_loader, test_loader) if include_test_set else (train_loader, valid_loader)


def accumulate_cycle_error(args, preds, truth, ncls, ns, criterion):
    """ Accumulates reconstruction error (generally MSE weighted by factor)
    This is very similar to above function so can be merged in future, but for
    clarity is kept separate for now.
    """
    error = torch.zeros(1).to(args.device)
    prev = 0

    # Loop through all heads and accumulate losses
    for s in range(ns):
        num_classes = ncls[s]

        y_hat = preds[:, prev : prev + num_classes]
        y_true = truth[:, prev : prev + num_classes]

        # Accumulate error
        error += criterion(y_hat, y_true) / ns
        prev += num_classes

    return error


def compute_factor_accuracy(preds, label, ncls, ns):
    """ Debug classifier accuracy by computing the accuracy over each
    factor and return as a list """
    accs = []
    prev = 0

    for s in range(ns):
        num_classes = ncls[s]

        if isinstance(preds, dict):
            guess = preds[f'factor_{s}'].argmax(1)
        else:
            guess = preds[:, prev : prev + num_classes].argmax(1)

        acc = torch.sum(guess == label[:, s]) / float(len(guess))
        accs.append(acc.item())

        prev += num_classes

    return accs


def get_vgg(nc=1, pretrained=False, out_features_smaller=True):
    vgg = tvmodels.vgg16(pretrained=pretrained)
    if out_features_smaller:
        vgg.classifier._modules['3'] = nn.Linear(in_features=4096, out_features=64)

    if nc != 3:
        vgg.features[0] = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)

    remove_layers = ['4', '5', '6']
    for l in remove_layers:
        vgg.classifier._modules[l] = nn.Identity()

    vgg.eval()
    return vgg


def prdc(reals, fakes, k=5, embed=False, nc=1, pretrained=False, out_features_smaller=True):
    if embed:
        vgg = get_vgg(nc, pretrained, out_features_smaller)
        
        # Check if numpy 
        if type(reals) is np.ndarray:
            reals = torch.from_numpy(reals)
        if type(fakes) is np.ndarray:
            fakes = torch.from_numpy(fakes)

        # Check if need to move channel dim
        if reals.shape[-1] in [1,3]:
            reals = reals.permute(0, 3, 1, 2)
        if fakes.shape[-1] in [1,3]:
            fakes = fakes.permute(0, 3, 1, 2)

        with torch.no_grad():
            reals = vgg(reals)
            fakes = vgg(fakes)

    reals = reals.reshape(reals.shape[0], -1).cpu().numpy()
    fakes = fakes.reshape(fakes.shape[0], -1).cpu().numpy()
    metrics = compute_prdc(reals, fakes, k)
    print(metrics)
    return metrics


