import gs
import os
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import json
from tqdm import tqdm
import pandas as pd

import matplotlib.pyplot as plt
from pylab import text
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from pathlib import Path

from args import BarcodeArgParser
from gen_cov import get_real, bicluster_mean, bicluster, interpolate


def bicluster_score(i, cocluster, data, real=False):
    rows, cols = cocluster.get_indices(i)
    row_complement = np.nonzero(np.logical_not(cocluster.rows_[i]))[0]
    col_complement = np.nonzero(np.logical_not(cocluster.columns_[i]))[0]
    
    denom_in = len(rows) * len(cols)
    denom_out = len(col_complement) * len(rows)
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
    out_sum = data[rows][:, col_complement].sum()
   
    in_norm = in_sum / denom_in
    out_norm = out_sum / denom_out
    
    score = out_norm - in_norm
    return score, out_norm, in_norm


def covar(results_dict, name, gs_results_dir, dist_type, verbose, real=False):
    ''' Following functions are mostly the same as in gen_cov.py but
    with extra functions removed and modified to support both
    wasserstein and euclidean distances '''
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
                    score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode),
                                          dist_type=dist_type)
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


def agg_covar(results_dict, ones_only, name, gs_results_dir, dist_type, real=False):
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
                            score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode),
                                                  dist_type=dist_type)
                            cur_agg_diffs.append(score)
            agg_diffs.append(cur_agg_diffs)
    else:
        for cur_factor, cur_factor_dict in cur_results_dict.items():
            cur_agg_diffs = []
            for target_factor, target_factor_dict in target_results_dict.items():
                factor_avg = 0
                for cur_value, cur_barcode in sorted(cur_factor_dict.items()):
                    for target_value, target_barcode in sorted(target_factor_dict.items()):
                        score = gs.geom_score(np.asarray(cur_barcode), np.asarray(target_barcode),
                                              dist_type=dist_type)
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


def compare_barcodes(args, filename):
    ''' mostly the same as in gen_cov.py but with extra functions removed and
    modified to permit both wasserstein and euclidean distances '''
 
    assert 'mean_barcodes' in filename, 'Need to run on wbary barcodes'
    if os.path.exists(filename):
        with open(filename, "r") as f:
            results_dict = json.load(f)
    else: 
        raise FileNotFoundError

    diffs = covar(results_dict, args.name, args.gs_results_dir,
        args.dist, args.verbose, real=args.sup)

    agg_diffs, rev_diffs = agg_covar(results_dict, args.ones_only,
        args.name, args.gs_results_dir, args.dist, real=args.sup)

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

        clust_score_info = []
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
           
            bicluster_vals = [bicluster_score(i, cluster, agg_diffs, real=args.sup) for i in range(n_clust)] # Get scores, lower the better

            bicluster_scores = [x[0] for x in bicluster_vals]
            out_norms = [x[1] for x in bicluster_vals]
            in_norms = [x[2] for x in bicluster_vals]

            biclusters_sorted = np.argsort(bicluster_scores)  # Order groups/cluster/factors by scores 
            overall_score = np.asarray(bicluster_scores).mean() * 10000
            print(f"{n_clust} Bicluster sum: {overall_score}")
            unsup_scores[n_clust] = overall_score
           
            infos = {'n_clust': n_clust,
                     'score': overall_score,
                     'out_norm': np.mean(out_norms) * 10000,
                     'in_norm': np.mean(in_norms) * 10000}
            clust_score_info.append(infos)

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

        clust_score_info = pd.DataFrame(clust_score_info)
        clust_score_info['mean'] = args.mean
        clust_score_info['dist'] = args.dist
        print(clust_score_info)
        clust_score_info.to_csv(f'./scores_{args.mean}_{args.dist}.csv', index=False)

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
            df['run_name'] = args.name
            df_col_order = ['rank', 'score', 'n_clust', 'decoder', 'dataset', 'type', 'timestamp', 'run_name']
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
    parser.parser.add_argument('--mean', choices=['euclidean', 'wasserstein'])
    parser.parser.add_argument('--dist', choices=['euclidean', 'wasserstein'])
    args_ = parser.parse_args()

    mean_type = args_.mean
    dist_type = args_.dist
    assert mean_type in ['euclidean', 'wasserstein']
    assert dist_type in ['euclidean', 'wasserstein']

    filename = f"{args_.gs_results_dir}/barcodes/{args_.extra}.json"

    with open(filename, "r") as f:
        results_dict = json.load(f)
    for i, (key, relevant) in tqdm(enumerate(results_dict.items())):
        for val, barcode in relevant.items():

            if mean_type == 'euclidean':
                cur_code = np.mean(barcode, axis=0)
            elif mean_type == 'wasserstein':
                cur_code = gs.barymean(np.asarray(barcode))

            results_dict[key][val] = cur_code.tolist()

    filename = f"{args_.gs_results_dir}/mean_barcodes/{args_.extra}_{mean_type}.json"
    if not os.path.exists(f"{args_.gs_results_dir}/mean_barcodes"):
        os.mkdir(f"{args_.gs_results_dir}/mean_barcodes")
    with open(filename, "w+") as f:
        json.dump(results_dict, f)


    args_.search_n_clusters = True
    args_.bicluster = True
    args_.name = f'real_{args_.dataset_name}_{mean_type}_mean_{dist_type}_dist'
    # args_.save_interpolations = True
    compare_barcodes(args_, filename)

    def concat_all():
        a = pd.read_csv('./scores_euclidean_euclidean.csv')
        b = pd.read_csv('./scores_euclidean_wasserstein.csv')
        c = pd.read_csv('./scores_wasserstein_euclidean.csv')
        d = pd.read_csv('./scores_wasserstein_wasserstein.csv')
        pd.concat([a,b,c,d]).to_csv('./scores_concat.csv', index=False)

    concat_all()