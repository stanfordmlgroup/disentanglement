import gs
import os
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import json
import random
from tqdm import tqdm

import matplotlib.pyplot as plt
from pylab import text
from sklearn.cluster import SpectralCoclustering
from sklearn.metrics import consensus_score
from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import seaborn as sns

from args import BarcodeArgParser

if __name__ == "__main__":
    parser = BarcodeArgParser()
    args = parser.parse_args()

    filename = f"{args.gs_results_dir}/mean_barcodes/{args.extra}.json"
    if not os.path.exists(filename):
        print(f'mean  barcodes dont exist')
        filename = f"{args.gs_results_dir}/barcodes/{args.extra}.json"
        mode = "bary"
    else:
        mode = "baryed"
    if args.plot:
        if not os.path.exists(f"figure_interps/{args.extra}"):
            os.makedirs(f"figure_interps/{args.extra}", exist_ok=True)
    with open(filename, "r") as f:
        results_dict = json.load(f)
    import matplotlib.ticker as ticker

    factor_names = ["Shape", "Scale", "Orientation", "X Position", "Y Position"]
    vis_type = "step"
    filetype = "png"
    for i, (key, relevant) in tqdm(enumerate(results_dict.items())):
        num_barcodes = len(relevant.items())
        cur_items = list(relevant.items())
        random.shuffle(cur_items)
        plt.style.use('seaborn')
        num_landmarks = 100 #64
        fig, ax = plt.subplots()
        for val, barcode in cur_items:
            if mode == "bary":
                cur_code = gs.barymean(np.asarray(barcode))
            elif mode == "baryed":
                cur_code = barcode
            else:
                cur_code = np.mean(np.asarray(barcode), 0)
            if args.preprocess:
                if isinstance(cur_code, list):
                    results_dict[key][val] = cur_code
                else:
                    results_dict[key][val] = cur_code.tolist()
            if args.plot:
                if vis_type == "smooth":
                    sns.lineplot(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.2), color=f'C{i}')
                    plt.fill_between(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.1), color=f'C{i}')
                else:
                    import pdb;pdb.set_trace()
                    sns.lineplot(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.2), color=f'C{i}', drawstyle='steps-pre')
                    plt.fill_between(np.arange(num_landmarks), cur_code[:num_landmarks], alpha=max(1 / num_barcodes, 0.1), color=f'C{i}', step="pre")
                plt.xlabel("Holes", fontsize=14)
                plt.ylabel("Density", fontsize=14)
                # plt.title(factor_names[i])
        if args.plot:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
            plt.tight_layout()
            plt.xlim(0,num_landmarks)
            plt.savefig(f"figure_interps/{args.extra}/{i}_{mode}_{vis_type}_barcode.{filetype}")
            plt.close()
        
    if args.preprocess:
        filename = f"{args.gs_results_dir}/mean_barcodes/{args.extra}.json"
        if not os.path.exists(f"{args.gs_results_dir}/mean_barcodes"):
            os.mkdir(f"{args.gs_results_dir}/mean_barcodes")
        with open(filename, "w+") as f:
            json.dump(results_dict, f)
