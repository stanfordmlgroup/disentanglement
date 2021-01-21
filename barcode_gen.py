import os
import time
import gs
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from args import BarcodeArgParser
from utils import sample_noise, load_models, load_optimizer, visualize, save_model, softmax_classes, argmax_classes, labels_to_onehot, get_dataset_args, dict_to_labels, ce_with_probs, create_dataset_splits


rs = np.random.RandomState(1)

def compare_embedding_spaces_real(args, plot=False):
    # Doesn't work (just output same diff 0) for vgg embeddings that are 4096 and for 64, pretrained=False or pretrained=True
    nc = 1 if args.dataset_name == "dsprites" else 3

    if args.dataset_name == "dsprites":
        from disentanglement_lib.data.ground_truth.dsprites import DSprites
        dataset = DSprites(list(range(1,6)))
    elif "celeba" in args.dataset_name:
        from datasets.classification_dataset import ClassificationDataset
        dataset = ClassificationDataset(args.dataset_name, 128)
    
    num_samples = 100 if "hq" in args.dataset_name else 1000

    vgg = models.vgg16(pretrained='celeba' in args.dataset_name)
    vgg.classifier._modules['3'] = nn.Linear(in_features=4096, out_features=64)
    vgg.features[0] = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    remove_layers = ['4', '5', '6']
    for l in remove_layers:
        vgg.classifier._modules[l] = nn.Identity()
    
    vgg.eval()

    correctness = defaultdict(list)
    if args.dataset_name == "dsprites":
        samples = dataset.sample_factors(num_samples, rs)
        factors_num_values = dataset.factors_num_values
    elif "celeba" in args.dataset_name:
        factors_num_values = [2 for _ in range(40)]
    results_dict = dict([(i, {}) for i, _ in enumerate(factors_num_values)])
    for cur_factor, num_value in enumerate(factors_num_values):
        for cur_value in range(num_value):          
            obslist = []
            if args.dataset_name == "dsprites":  
                for s in tqdm(samples):
                    ss = s.copy()
                    ss[cur_factor] = cur_value
                    obs = dataset.sample_observations_from_factors(ss, rs)[0]
                    obslist.append(obs)
                obslist = torch.from_numpy(np.array(obslist)).permute(0,3,1,2)
            elif "celeba" in args.dataset_name:
                dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
                for image, label in dataloader:
                    if label[0][cur_factor] == cur_value:
                        print(len(obslist), end=",")
                        obslist.append(image[0])
                    if len(obslist) >= num_samples:
                        break
                obslist = torch.stack(obslist)
            print(f'embedding...')
            with torch.no_grad():
                embed = vgg(obslist).detach().numpy()
            print('starting rlts...')
            rlts = gs.rlts(embed, L_0=args.L_0, gamma=args.gamma, n=100)
            if plot:
                import matplotlib.pyplot as plt
                gs.fancy_plot(mrlt, label=f'MRLT of {cur_factor}_{cur_value}')
                # plt.xlim([0, 30])
                plt.legend()
                plt.savefig(f"{args.gs_results_dir}/plots/embedding_space_real_{args.dataset_name}_{cur_factor}_{cur_value}.png")
                plt.close()
            results_dict[cur_factor][cur_value] = rlts.tolist()

            # Write to file
            with open(args.results_file, "w+") as f:
                json.dump(results_dict, f)
    print(f'Done')



def compare_embedding_spaces_fake(args, plot=False):
    # Doesn't work (just output same diff 0) for vgg embeddings that are 4096 and for 64, pretrained=False or pretrained=True
    nc = 1 if args.dataset_name == "dsprites" else 3

    dataset, ns, image_shape, npix, nc, ncls, factor_id2name = get_dataset_args(args, return_factor_name_map=args.dataset_name == "celeba")

    decoder_params = {'dataset_name': args.dataset_name}
    decoder = load_models(
        args, ns, npix, nc, ncls,
        model_types=["decoder"],
        model_params=[decoder_params],
        model_ckpts=[args.decoder_ckpt]
    )
    decoder.eval()

    num_samples = 100 if "hq" in args.dataset_name else 1000
    num_batches = 4 if args.decoder_model == 'WGAN' else 1
    batch_size = num_samples // num_batches
    assert num_samples / num_batches == num_samples // num_batches, f'num samples needs to be divisible by num batches'

    vgg = models.vgg16(pretrained='celeba' in args.dataset_name)
    vgg.classifier._modules['3'] = nn.Linear(in_features=4096, out_features=64)
    vgg.features[0] = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    remove_layers = ['4', '5', '6']
    for l in remove_layers:
        vgg.classifier._modules[l] = nn.Identity()
    
    vgg.eval()
    vgg.to(args.device)

    correctness = defaultdict(list)
    samples = torch.Tensor(np.random.randn(num_samples, args.nz)).to(args.device)
    factors_num_values = args.nz
    results_dict = dict([(i, {}) for i in range(args.nz)])
    num_value = 10
    for cur_factor in range(args.nz):
        for _ in range(num_value):
            cur_value = np.asscalar(np.random.randn(1))
            embeds = []
            for b in range(num_batches):
                obslist = []
                for z in tqdm(samples[batch_size * b:batch_size * (b+1)]):
                    zz = z.clone()
                    zz[cur_factor] = cur_value
                    obs = decoder(zz.view(1, -1))
                    obslist.append(obs)
                obslist = torch.cat(obslist)
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

    # Write to file
    with open(args.results_file, "w") as f:
        json.dump(results_dict, f)
    print(f'Done')


if __name__ == "__main__":
    parser = BarcodeArgParser()
    args_ = parser.parse_args()

    if args_.gamma is None:
        args_.gamma = 1/128

    if args_.L_0 is None:
        args_.L_0 = 100

    if args_.real:
        results_file = f"{args_.gs_results_dir}/barcodes/real_{args_.dataset_name}"
        if args_.suffix is not None:
            results_file += f"_{args_.suffix}"
        if os.path.exists(f'{results_file}.json') and not args_.override:
            # Do not override
            timestamp = str(time.time()).replace('.','')
            args_.results_file = f"{results_file}_{timestamp}.json"
        else:
            args_.results_file = f"{results_file}.json"
        print(f'(Over)writing to barcodes file {args_.results_file}')
        compare_embedding_spaces_real(args_)
    else:
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
