from infogan_barcode import load_infogan_decoder

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parent.parent.parent))

import os
import gs
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import time
import pandas as pd
import operator

import matplotlib.pyplot as plt
from pylab import text
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from args import BarcodeArgParser


def get_real():
    real_name = f'real_{args_.dataset_name}'
    real_filename = f"{args_.gs_results_dir}/mean_barcodes/{real_name}.json"
    with open(real_filename, "r") as f:
        real_results_dict = json.load(f)
    return real_results_dict

def covar(results_dict, name, gs_results_dir, verbose, real=False):
    diffs = []
    cur_results_dict = results_dict
    if real:
        target_results_dict = get_real()
    else:
        target_results_dict = results_dict
    for cur_factor, cur_factor_dict in cur_results_dict.items():
        for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
            cur_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                for target_value, target_barcode in sorted(target_factor_dict.items()):
                    score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                    cur_diffs.append(score)
                    if verbose:
                        print(score, end=",")
            if verbose:
                print()
            diffs.append(cur_diffs)
    data = np.asarray(diffs)
    plt.matshow(data, cmap=plt.cm.Blues)
    plt.savefig(f"{gs_results_dir}/covar/{name}.png")
    plt.close()
    return data

def agg_covar(results_dict, ones_only, name, gs_results_dir, real=False):
    agg_diffs = []
    cur_results_dict = results_dict
    if real:
        target_results_dict = get_real()
    else:
        target_results_dict = results_dict
    if ones_only:
        for cur_factor, cur_factor_dict in cur_results_dict.items():
            cur_agg_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
                    for target_value, target_barcode in sorted(target_factor_dict.items()):
                        if int(cur_value) == 1 and int(target_value) == 1:
                            score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                            cur_agg_diffs.append(score)
            agg_diffs.append(cur_agg_diffs)
    else:
        for cur_factor, cur_factor_dict in cur_results_dict.items():
            cur_agg_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                factor_avg = 0
                for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
                    for target_value, target_barcode in sorted(target_factor_dict.items()):
                        score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode))
                        factor_avg += score
                factor_avg /= len(list(target_factor_dict.items())) * len(list(cur_factor_dict.items()))
                cur_agg_diffs.append(factor_avg)
            agg_diffs.append(cur_agg_diffs)
    agg_diffs = np.asarray(agg_diffs)
    rev_diffs = 1 - agg_diffs
    plt.matshow(rev_diffs, cmap=plt.cm.Blues)
    plt.savefig(f"{gs_results_dir}/covar/agg_{name}.png")
    plt.close()
    return agg_diffs, rev_diffs


def bicluster_mean(cocluster, data, n_clust, real=False):
    if real:
        # Assymmetric: don't meanify columns (reals)
        sorted_idx_row = np.argsort(cocluster.row_labels_)
        sorted_data = data[sorted_idx_row]

        prev = None
        bounds = []
        sorted_labels = cocluster.row_labels_[sorted_idx_row]
        for i, c in enumerate(sorted_labels):
            if prev != c:
                bounds.append(i)
            prev = c
        bounds.append(len(sorted_labels))

        # Collapse rows
        avg_row_means = []
        for i in range(len(bounds) - 1):
            try:
                points = data[bounds[i]:bounds[i+1]]
            except:
                import pdb;pdb.set_trace()
            mean = points.mean(axis=0)
            avg_row_means.append(mean)
        avg_row_means = np.array(avg_row_means)
        return avg_row_means

    else:
        sorted_idx_row = np.argsort(cocluster.row_labels_)
        sorted_idx_col = np.argsort(cocluster.column_labels_)
        sorted_data = data[sorted_idx_row]
        sorted_data = sorted_data[:, sorted_idx_col]

        prev = None
        bounds = []
        sorted_labels = cocluster.row_labels_[sorted_idx_row]
        for i, c in enumerate(sorted_labels):
            if prev != c:
                bounds.append(i)
            prev = c
        bounds.append(len(sorted_labels))

        # Collapse rows
        avg_row_means = []
        for i in range(n_clust):
            points = data[bounds[i]:bounds[i+1]]
            mean = points.mean(axis=0)
            avg_row_means.append(mean)

        prev = None
        bounds = []
        sorted_labels = cocluster.column_labels_[sorted_idx_col]
        for i, c in enumerate(sorted_labels):
            if prev != c:
                bounds.append(i)
            prev = c
        bounds.append(len(sorted_labels))

        # Collapse cols
        avg_data = []
        avg_row_means = np.array(avg_row_means)
        for i in range(n_clust):
            points = avg_row_means[:, bounds[i]:bounds[i+1]]
            mean = points.mean(axis=1)
            avg_data.append(mean)

        avg_data = np.array(avg_data)
        return avg_data


def bicluster_score(i, cocluster, data, real=False):
    rows, cols = cocluster.get_indices(i)
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]

    denom_in = len(rows) * len(cols)
    denom_out = len(row_complement) * len(cols) + len(col_complement) * len(cols)
    if denom_in == 0 or denom_out == 0:
        # Skip because no correspondence
        if real:
            return 0
        else:
            print('Denom should not be 0 in unsupervised case')
            import pdb;pdb.set_trace()

    # Get sum of values inside of cluster
    in_sum = data[rows][:, cols].sum()

    # Get sum of values outside of cluster
    out_sum = (data[row_complement][:, cols].sum() + data[rows][:, col_complement].sum())

    in_norm = in_sum / denom_in
    out_norm = out_sum / denom_out

    score = out_norm - in_norm
    return score


def bicluster(data, n_clust, name, gs_results_dir):
    model = SpectralCoclustering(n_clusters=n_clust, random_state=0)
    cluster = model.fit(data)

    fit_data = data[np.argsort(model.row_labels_)]
    fit_data = fit_data[:, np.argsort(model.column_labels_)]

    plt.matshow(fit_data, cmap=plt.cm.Blues)
    plt.title(f"{name} with {n_clust} coclusters")
    plt.savefig(f"{gs_results_dir}/cocluster/agg_{n_clust}_{name}.png")
    plt.close()

    print(f"Saved cocluster png to {gs_results_dir}/cocluster/agg_{n_clust}_{name}.png")
    return cluster


def interpolate(args, z2s):
    # Given z dim num, name for the corresponding factor (for filename), generate latent interpolations using decoder
    from utils import sample_noise, load_models, visualize, get_dataset_args

    decoder_params = {'dataset_name': args.dataset_name}
    real_dataset, ns, image_shape, npix, nc, ncls, factor_id2name = get_dataset_args(args, return_factor_name_map='celeba' in args.dataset_name)
    decoder = load_models(
        args, ns, npix, nc, ncls,
        model_types=["decoder"],
        model_params=[decoder_params],
        model_ckpts=[args.decoder_ckpt]
    )
    decoder.eval()

    args.viz_batch_size = 8
    fixed_zz = sample_noise(args.viz_batch_size, args.nz, args.device)
    fixed_zs = []
    n_view = 1
    for dim in range(args.nz):
        fixed_z = np.tile(np.random.randn(n_view, args.nz), (args.viz_batch_size, n_view)).astype(np.float32)
        fixed_z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, args.viz_batch_size))
        fixed_zs.append(torch.from_numpy(fixed_z))

    fakes = []
    for iz, fixed_z in enumerate(fixed_zs):
        with torch.no_grad():
            fake = decoder(fixed_z).detach().cpu()

            # Save individual ones with factor names - easier to inspect b/c can't write text and don't know groupings
            s_num = z2s[iz]
            if 'celeba' in args.dataset_name:
                factor_name = factor_id2name[s_num]
            else:
                dsprites_factor_map = ['shape', 'scale', 'orient', 'xpos', 'ypos']
                factor_name = dsprites_factor_map[s_num]
            i_save_path = Path(args.gs_results_dir) / 'interpolations' / f'{args.name}_{factor_name}_z{iz}_s{s_num}.png'
            visualize(fake, i_save_path)

            fakes.append(fake)

    for iviz in range(10):
        fixed_zz = sample_noise(args.viz_batch_size, args.nz, args.device)
        fixed_zs = []
        for dim in range(args.nz):
            fixed_z = np.tile(np.random.randn(1, args.nz), (args.viz_batch_size, 1)).astype(np.float32)
            fixed_z[:, dim] = norm.ppf(np.linspace(0.01, 0.99, args.viz_batch_size))
            fixed_zs.append(torch.from_numpy(fixed_z))

        fakes = []
        for iz, fixed_z in enumerate(fixed_zs):
            with torch.no_grad():
                fake = decoder(fixed_z).detach().cpu()

                # Save individual ones with factor names - easier to inspect b/c can't write text and don't know groupings
                s_num = z2s[iz]
                if 'celeba' in args.dataset_name:
                    factor_name = factor_id2name[s_num]
                else:
                    dsprites_factor_map = ['shape', 'scale', 'orient', 'xpos', 'ypos']
                    factor_name = dsprites_factor_map[s_num]
                i_save_path = Path(args.gs_results_dir) / 'interpolations' / f'{args.name}_{factor_name}_z{iz}_s{s_num}_{iviz}.png'
                i_save_path = f'{args.name}_{factor_name}_z{iz}_s{s_num}_{iviz}.png'
                visualize(fake, i_save_path)

                fakes.append(fake)

        # Save concatenated full one
        fakes_concat = torch.cat(fakes, 0)
        save_path = Path(args.gs_results_dir) / 'interpolations' / f'{args.name}_match2s_{iviz}.png'
        visualize(fakes_concat, save_path)

selected_celeba = [20, 4, 31, 15, 22, 5, 8, 9, 11, 18]
celeba_factor_map = ['male', 'bald', 'smiling', 'eyeglasses', 'mustache', 'bangs', 'black_hair', 'blond_hair', 'brown_hair', 'heavy_makeup']

def preprocess_wbary(args, filename):
    original_filename = filename.replace('mean_', '')
    print(f'Reading preprocessed RLTs at {original_filename}')

    with open(original_filename, "r") as f:
        results_dict = json.load(f)

    if args.show_vis:
        vis_folder = f"{args.gs_results_dir}/wbary_vis/{args.name}"
        os.makedirs(vis_folder, exist_ok=True)
        print(f'Saving vis to {vis_folder}')

    for i, (key, relevant) in tqdm(enumerate(results_dict.items())):
        for val, barcode in relevant.items():
            cur_code = gs.barymean(np.asarray(barcode))
            results_dict[key][val] = cur_code.tolist()
            if args.show_vis:
                plt.bar(np.arange(len(cur_code)), cur_code, alpha=0.2, color=f'C{i}')
                plt.savefig(f"{vis_folder}/{i}_barcode")
        if args.show_vis:
            plt.close()

    if os.path.exists(filename) and not args.override:
        # Do not override
        timestamp = str(time.time()).replace('.','')
        filename = f"{filename}_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(results_dict, f)
    print(f'Saved Wbary barcodes to {filename}')
    return results_dict


def compare_barcodes(args, filename):
    assert 'mean_barcodes' in filename, 'Need to run on wbary barcodes'
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results_dict = json.load(f)
    else:
        # Preprocess W-barycenter RLTs
        results_dict = preprocess_wbary(args, filename)

    diffs = covar(results_dict, args.name, args.gs_results_dir, args.verbose, real=args.sup)
    agg_diffs, rev_diffs = agg_covar(results_dict, args.ones_only, args.name, args.gs_results_dir, real=args.sup)

    if args.bicluster:
        num_latents = len(results_dict)
        unsup_scores = {}
        if args.search_n_clusters:
            n_clusters = list(range(2, num_latents + 1))
        else:
            # Don't do +1 b/c not in range(), just use that value
            if 'celeba' in args.dataset_name:
                n_clusters = [min(40, num_latents)]
            else:
                n_clusters = [min(5, num_latents)]

        for n_clust in n_clusters:
            # Use reversed diffs: Higher the value the better for inside a factor, lower better for outside a factor
            cluster = bicluster(rev_diffs, n_clust, args.name, args.gs_results_dir)
            avg_cluster = bicluster_mean(cluster, rev_diffs, n_clust, real=args.sup)
            if args.plot:
                np.save(f'{args.gs_results_dir}/avg_cluster_np/{args.name}.npy', avg_cluster)

            # Supervised
            row_matches, col_matches = linear_sum_assignment(avg_cluster)
            match_dists = np.array(avg_cluster)[row_matches, col_matches]
            score = match_dists.sum() / avg_cluster.shape[1] # This is num real factors
            #sorted_match_dists = match_dists[:, col_matches]
            if args.plot:
                plt.matshow(avg_cluster[:, col_matches], cmap=plt.cm.Blues)
                plt.title(f"Averaged {args.name} {n_clust} coclusters")
                if args.dataset_name == 'dsprites' and (args.sup or args.real):
                    #plt.gcf().text(.02, .05, str(sorted_match_dists), fontsize=8)
                    dsprites_factor_map = ['shape', 'scale', 'orient', 'xpos', 'ypos']
                    factors_found = []
                    for ff in sorted(col_matches):
                        factor_name = dsprites_factor_map[ff]
                        factors_found.append(factor_name)
                    plt.gcf().text(.02, .01, f'{factors_found}', fontsize=8)
                    plt.gcf().text(.7, .01, f'{score}', fontsize=8)
                else:
                    plt.gcf().text(.02, .01, f'{score}', fontsize=8)
                plt.savefig(f"{args.gs_results_dir}/cocluster/average_{n_clust}_{args.name}.png")
                plt.close()

                print(f"Saved averaged fake-real cocluster png to {args.gs_results_dir}/cocluster/average_{n_clust}_{args.name}.png")

                # Save full thing
                plt.matshow(avg_cluster, cmap=plt.cm.Blues)
                if args.dataset_name == 'dsprites':
                    real_n_clust = 5
                else:
                    real_n_cluster = 40
                plt.title(f"Averaged {args.name} {n_clust}x{real_n_clust} coclusters")
                if args.dataset_name == 'dsprites':
                    plt.gcf().text(.02, .05, str(col_matches), fontsize=8)
                    plt.gcf().text(.7, .01, f'{score}', fontsize=8)
                else:
                    plt.gcf().text(.02, .01, f'{score}', fontsize=8)
                plt.savefig(f"{args.gs_results_dir}/cocluster/full_average_{n_clust}_{args.name}.png")
                plt.close()

                print(f"Saved averaged real cocluster png to {args.gs_results_dir}/cocluster/full_average_{n_clust}_{args.name}.png")

            bicluster_scores = [bicluster_score(i, cluster, agg_diffs, real=args.sup) for i in range(n_clust)] # Get scores, lower the better
            biclusters_sorted = np.argsort(bicluster_scores)  # Order groups/cluster/factors by scores
            overall_score = np.asarray(bicluster_scores).mean() * 10000
            print(f"{n_clust} Bicluster sum: {overall_score}")
            unsup_scores[n_clust] = overall_score

            if args.sup and args.save_interpolations:
                # Correspondences between averaged and non-averaged factors/dimensions
                # Save interpolations based on the matched values
                z2s = {}
                for unordered, ordered in enumerate(col_matches):
                    # Get dim from unordered value
                    clusts, _  = cluster.get_indices(unordered)
                    for cl in clusts:
                        # Provide corresponding s value from ordered matches
                        z2s[cl] = ordered
                interpolate(args, z2s)
                print(f'z to s mapping is {z2s}')

        if args.save_scores:
            # Write scores to file
            timestamp = str(time.time()).replace('.','')
            sorted_unsup_scores = sorted(unsup_scores.items(), key=operator.itemgetter(1), reverse=True)
            supervision = 'supervised' if args.sup else 'unsupervised'

            df = pd.DataFrame(sorted_unsup_scores, columns=['n_clust', 'score'])
            df['timestamp'] = timestamp
            df['dataset'] = args.dataset_name
            df['decoder'] = args.decoder_model
            df['type'] = supervision
            df['rank'] = df.index
            df_col_order = ['rank', 'score', 'n_clust', 'decoder', 'dataset', 'type', 'timestamp']
            df = df[df_col_order]

            scores_file = f'{args.gs_results_dir}/scores/all.csv'
            if os.path.exists(scores_file):
                df.to_csv(scores_file, mode='a', header=False, index=False)
            else:
                print(f'Creating scores file {scores_file}...')
                df.to_csv(scores_file, header=True, index=False)

            best_scores_file = f'{args.gs_results_dir}/scores/best.csv'
            best = df.iloc[:1]
            if os.path.exists(best_scores_file):
                best.to_csv(best_scores_file, mode='a', header=False, index=False)
            else:
                print(f'Creating scores file {best_scores_file}...')
                best.to_csv(best_scores_file, header=True, index=False)

            scores_file = f'{args.gs_results_dir}/scores/{args.name}_{supervision}.json'
            with open(scores_file, 'w') as f:
                json.dump(sorted_unsup_scores, f)
            print(f'Saved scores to {scores_file}')

if __name__ == "__main__":
    parser = BarcodeArgParser()
    args_ = parser.parse_args()

    ''' python infogan_gen_cov.py --save_interpolations'''
    # INFOGAN SETTINGS
    args_.decoder_model = 'InfoGAN-CR'
    args_.nz = 105
    args_.dataset_name = 'celeba'
    args_.real = False

    if args_.search_n_clusters:
        args_.bicluster = True

    if args_.save_interpolations:
        args_.bicluster = True

    if args_.real:
        if 'celeba' in args_.dataset_name:
            args_.ones_only = True
            print(f'Using {args_.dataset_name} with reals, so setting aggregation to ones only')
        args_.search_n_clusters = True

        args_.name = f'real_{args_.dataset_name}'
        if args_.suffix is not None:
            args_.name += f'_{args_.suffix}'
        compare_barcodes(args_, f"{args_.gs_results_dir}/mean_barcodes/{args_.name}.json")
    else:
        # this block is being called
        args_.name = f'fake_{args_.decoder_model}_{args_.dataset_name}'
        if args_.suffix is not None:
            args_.name += f'_{args_.suffix}'
        compare_barcodes(args_, f"{args_.gs_results_dir}/mean_barcodes/{args_.name}.json")
