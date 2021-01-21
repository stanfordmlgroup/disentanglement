from InfoGAN_CR_CelebA.tf_load_decoder import load_infogan_celeba_decoder
from InfoGAN_CR_dSprites.tf_load_decoder import load_infogan_dsprites_decoder

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent))

# original imports from barcode_gen.py
import os
import time
import gs
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from args import BarcodeArgParser
from utils import sample_noise, load_models, load_optimizer, visualize, save_model, softmax_classes, argmax_classes, labels_to_onehot, get_dataset_args, dict_to_labels, ce_with_probs, create_dataset_splits

from pdb import set_trace as st

rs = np.random.RandomState(1)


def compare_embedding_spaces_fake(args, plot=False):
    # Doesn't work (just output same diff 0) for vgg embeddings that are 4096 and for 64, pretrained=False or pretrained=True
    nc = 1 if args.dataset_name == "dsprites" else 3
    # dataset, ns, image_shape, npix, nc, ncls, factor_id2name = get_dataset_args(args, return_factor_name_map=args.dataset_name == "celeba")
    print('nc: ', nc)

    # --- load TF custom decoder here --- #
    if args.dataset_name == 'celeba':
        sess, decoder = load_infogan_celeba_decoder()
    else:
        sess, decoder = load_infogan_dsprites_decoder()

    num_samples = 100 if "hq" in args.dataset_name else 1000
    num_batches = 4 if args.decoder_model == 'WGAN' else 1
    batch_size = num_samples // num_batches
    assert num_samples / num_batches == num_samples // num_batches, f'num samples needs to be divisible by num batches'

    vgg = models.vgg16(pretrained=True)
    vgg.classifier._modules['3'] = nn.Linear(in_features=4096, out_features=64)
    vgg.features[0] = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)

    remove_layers = ['4', '5', '6']
    for l in remove_layers:
        vgg.classifier._modules[l] = nn.Identity()

    vgg.eval()
    vgg.to(args.device)

    # infogan-cr takes uniform samples (looks really bad with normal)
    samples = np.random.uniform(size=(num_samples, args.nz))

    correctness = defaultdict(list)
    factors_num_values = args.nz
    results_dict = dict([(i, {}) for i in range(args.nz)])
    num_value = 10
    for cur_factor in range(args.nz):
        for _ in range(num_value):
            # take uniform sample again
            cur_value = np.random.uniform()
            embeds = []
            for b in range(num_batches):
                obslist = []
                for z in tqdm(samples[batch_size * b:batch_size * (b+1)]):
                    zz = z.copy()   # copy instead of clone
                    zz[cur_factor] = cur_value

                    # feed into InfoGAN decoder
                    obs = decoder.sample_from(zz.reshape(1, -1))
                    obslist.append(obs)

                # swap axes from TF to PyTorch convension
                obslist = torch.from_numpy(np.vstack(obslist)).permute(0, 3, 1, 2)
                obslist = obslist.to(args.device)

                # # upsample to 64x64
                # obslist = F.interpolate(obslist, size=(64, 64), mode='bilinear', align_corners=False)

                print(f'embedding...')
                with torch.no_grad():
                    embed = vgg(obslist).cpu().detach().numpy()
                embeds.append(embed)
            embeds = np.concatenate(embeds)
            print('starting rlts...')
            rlts = gs.rlts(embeds, L_0=args.L_0, gamma=args.gamma, n=100)
            if plot:
                import matplotlib.pyplot as plt
                gs.fancy_plot(mrlt, label=f'MRLT of {cur_factor}_{cur_value}')
                # plt.xlim([0, 30])
                plt.legend()
                plt.savefig(f"{args.gs_results_dir}/plots/embedding_space_fake_{args.decoder_model}_{args.dataset_name}_{cur_factor}_{cur_value}.png")
                plt.close()
            results_dict[cur_factor][cur_value] = rlts.tolist()

            # save turned off by default now since barcodes are directly made
            if args.save_mode == 'single':
                for im in range(obslist.shape[0]):
                    vutils.save_image(
                        obslist[im:im + 1, :],
                        Path(args.infogan_im_path) / f'fixed_{cur_factor}_at_{np.round(cur_value, 3)}_sample_{im}.png',
                        normalize=True
                    )
            elif args.save_mode == 'batch':
                torch.save(
                    obslist,
                    Path(args.infogan_im_path) / f'batch_fixed_{cur_factor}_at_{np.round(cur_value, 3)}.pt'
                )
            else:
                pass

        # re-write to file regularly to save progress (instead of at very end)
        with open(args.results_file, "w") as f:
            json.dump(results_dict, f)
        print(f'Done')

    sess.close()


if __name__ == '__main__':
    parser = BarcodeArgParser()
    args_ = parser.parse_args()

    args_.decoder_model = 'InfoGAN-CR'

    assert args_.dataset_name in ['dsprites', 'celeba']
    if args_.dataset_name == 'celeba':
        args_.nz = 105
        args_.infogan_im_path = f'/deep/group/disentangle/infogan_samples/celeba'
    else:
        args_.nz = 10
        args_.infogan_im_path = f'/deep/group/disentangle/infogan_samples/dsprites'

    # change settings as needed. 'batch' saves samples as .pt,
    # cleaner than 'single' which saves separate images, 'skip' ignores it
    args_.save_mode = 'skip'
    assert args_.save_mode in ['batch', 'single', 'skip']
    args_.override = True

    # parent code
    if args_.gamma is None:
        args_.gamma = 1/128

    if args_.L_0 is None:
        args_.L_0 = 100

    # only need to run comparison on fakes
    results_file = f"{args_.gs_results_dir}/barcodes/fake_{args_.decoder_model}_{args_.dataset_name}"
    if args_.suffix is not None:
        results_file += f"_{args_.suffix}"
    if os.path.exists(f'{results_file}.json') and not args_.override:
        # Do not override
        timestamp = str(time.time()).replace('.','')
        args_.results_file = f"{results_file}_{timestamp}.json"
    else:
        args_.results_file = f"{results_file}.json"
    print(f'(Over)writing to barcodes file {args_.results_file}')
    compare_embedding_spaces_fake(args_)
