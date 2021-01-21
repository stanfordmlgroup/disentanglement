import os
import shutil
import pandas as pd
import numpy as np

suffix= 'var2'
out_dir = f'grid_figure_{suffix}'
os.makedirs(out_dir, exist_ok=True)

datasets =  ['celeba']
sups = ['unsup', 'sup']
dfs, fps = [], []
for dataset in datasets:
    for sup in sups:
        fp = f'/deep/group/disentangle/gs_results/scores/all_{dataset}_{sup}_{suffix}.csv'
        df = pd.read_csv(fp)
        fps.append(fp)
        dfs.append(df)

# Get best score per model x dataset
# rank,score,n_clust,decoder,dataset,type,timestamp,run_name
for fp, df in zip(fps, dfs):
    df = df.drop_duplicates(['run_name'], keep='last')
    df = df.sort_values(by=['dataset', 'score'], ascending=False).reset_index(drop=True)
    df['rank'] = df.index
    if 'unsup' in fp:
        sup = 'unsup'
    else:
        sup = 'sup'
    if 'dsprites' in fp:
        dataset = 'dsprites'
    elif 'celebahq' in fp:
        dataset = 'celebahq'
    else:
        dataset = 'celeba'

    # Get n_clust for that
    for decoder in df['decoder'].unique():
        n_clusts = df[df['decoder'] == decoder]['n_clust']
        names = df[df['decoder'] == decoder]['run_name']
        for n_clust,  name in zip(n_clusts, names):
            grid_fn = f'agg_{n_clust}_{name}.png'
            grid_dir = f'/deep/group/disentangle/gs_results/cocluster'
            grid_fp = f'{grid_dir}/{grid_fn}'
         
            # Copy that grid file to one folder
            dest_fp = f'{out_dir}/{sup}_{grid_fn}'
            shutil.copyfile(grid_fp, dest_fp)
            print(f'copied to {dest_fp}')

        # Save max/best with its own name
        max_row = df[df['decoder'] == decoder]['score'].idxmax()

        # fp for grid
        n_clust_max = df.iloc[max_row]['n_clust']
        dataset_max = df.iloc[max_row]['dataset']
        name_max = df.iloc[max_row]['run_name']
        grid_fn = f'agg_{n_clust_max}_{name_max}.png'
        grid_dir = f'/deep/group/disentangle/gs_results/cocluster'
        grid_fp = f'{grid_dir}/{grid_fn}'
         
        # Copy that grid file to one folder
        if 'unsup' in fp:
            sup = 'unsup'
        else:
            sup = 'sup'
        dest_fp = f'{out_dir}/max_{sup}_{grid_fn}'
        shutil.copyfile(grid_fp, dest_fp)
        print(f'copied to {dest_fp}')


