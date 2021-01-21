import argparse
import json
import numpy as np
import os
import random
from datetime import datetime

import torch
import torch.backends.cudnn as cudnn

class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='disentanglement')

        self.parser.add_argument('--name', type=str, default='debug', help='Experiment name prefix.')

        # Architecture options
        self.parser.add_argument('--decoder_model', type=str, default='VAE', choices=('dcgan', 'btcvae', 'factor_VAE', 'VAE', 'betaH_VAE', 'betaB_VAE', 'PGAN', 'StyleGAN', 'BEGAN', 'WGAN', 'InfoGAN-CR', 'InfoGAN'), help='Decoder model to use (and corresponding encoder)')


        self.parser.add_argument('--encoder_model', type=str, default='VAE', choices=('dcgan', 'btcvae', 'factor_VAE', 'VAE', 'betaH_VAE', 'betaB_VAE'), help='Decoder model to use (and corresponding encoder)')

        self.parser.add_argument('--transcoder_model', type=str, default='mlp', choices=('mlp', 'mlptranscoder',  'freia', 'made'), help='Transcoder architecture')

        # Architecture ckpts
        self.parser.add_argument('--decoder_ckpt', type=str, default='/deep/group/sharonz/dis/results/latent_viz_dsprites_conditional_Apr17_150905/ckpts/step_14008320.pth.tar', help='Decoder checkpoint file path.')
        self.parser.add_argument('--transcoder_ckpt', type=str, default='/sailhome/ezelikma/coord_trans/results/debug_Apr26_175927/ckpts/step_400000.pth.tar', help='Transcoder checkpoint file path.')
        self.parser.add_argument('--encoder_ckpt', type=str, default=None, help='Encoder checkpoint file path.')
        self.parser.add_argument('--discriminator_ckpt', type=str, default=None, help='Discriminator checkpoint file path.')

        self.parser.add_argument('--dataset_name', type=str, default='dsprites', choices=('dsprites', 'shapes3d', 'norb', 'cars3d', 'mpi3d', 'scream', 'celeba', 'celebahq', 'ffhq'), help='Dataset to use.')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size.') # 2048 fits
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated list of GPU IDs. Use -1 for CPU.')

        self.parser.add_argument('--nz', type=int, default=None, help='When training: Latent z dimension. If None, default to size of s. Note that enforced dimensions are below for certain decoder models.')

        self.parser.add_argument('--save_dir', type=str, default='./results', help='Directory for results, prefix.')
        self.parser.add_argument('--viz_batch_size', type=int, default=1, help='Visualization image batch size.')
        self.parser.add_argument('--num_workers', default=0, type=int, help='Number of threads for the DataLoader.')
        self.parser.add_argument('--init_method', type=str, default='kaiming', choices=('kaiming', 'normal', 'xavier'), help='Initialization method to use for conv kernels and linear weights.')


    def parse_args(self):
        args = self.parser.parse_args()

        # Enforce nz for certain models
        if args.decoder_model in ['PGAN', 'StyleGAN']:
            args.nz = 512
        elif 'vae' in args.decoder_model.lower():
            args.nz = 10
        elif args.decoder_model == 'BEGAN':
            args.nz = 64
        elif args.decoder_model == 'WGAN':
            args.nz = 128
        print(f'Enforcing nz to be {args.nz} for decoder model {args.decoder_model}')

        if args.decoder_model != args.encoder_model:
            print(f'Note that decoder model {args.decoder_model} and encoder model {args.encoder_model} are different')

        # Create save dir for run
        args.name = args.name + '_' + datetime.now().strftime('%b%d_%H%M%S')
        save_dir = os.path.join(args.save_dir, f'{args.name}')
        os.makedirs(save_dir, exist_ok=False)
        args.save_dir = save_dir

        # Save args to a JSON file
        with open(os.path.join(save_dir, 'args.json'), 'w') as fh:
            json.dump(vars(args), fh, indent=4, sort_keys=True)
            fh.write('\n')

        # Create ckpt dir and viz dir
        args.ckpt_dir = os.path.join(args.save_dir, 'ckpts')
        os.makedirs(args.ckpt_dir, exist_ok=False)

        args.viz_dir = os.path.join(args.save_dir, 'viz')
        os.makedirs(args.viz_dir, exist_ok=False)

        # Set up available GPUs
        def args_to_list(csv, arg_type=int):
            """Convert comma-separated arguments to a list."""
            arg_vals = [arg_type(d) for d in str(csv).split(',')]
            return arg_vals

        args.gpu_ids = args_to_list(args.gpu_ids)

        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            args.device = 'cuda'
        else:
            args.device = 'cpu'

        if hasattr(args, 'supervised_factors'):
            args.supervised_factors = args_to_list(args.supervised_factors)

        # Checkpoints to None if passed as None
        if args.decoder_ckpt == 'None':
            args.decoder_ckpt = None

        if args.gs_results_dir:
            os.makedirs(args.gs_results_dir, exist_ok=True)

        print(args)
        return args
