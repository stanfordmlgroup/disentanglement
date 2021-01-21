import pandas as pd
import numpy as np

suffix= 'var2'
datasets =  ['celeba']
sups = ['unsup', 'sup']
dfs, fps = [], []
for dataset in datasets:
    for sup in sups:
        fp = f'/deep/group/disentangle/gs_results/scores/all_{dataset}_{sup}_{suffix}.csv'
        df = pd.read_csv(fp)
        fps.append(fp)
        dfs.append(df)

# rank,score,n_clust,decoder,dataset,type,timestamp,run_name
for fp, df in zip(fps, dfs):
    df = df.drop_duplicates(['run_name'], keep='last')
    df = df.sort_values(by=['dataset', 'score'], ascending=False).reset_index(drop=True)
    df['rank'] = df.index

    means = df.groupby(['decoder'], as_index=False).agg({'score':[np.mean, np.std], 'n_clust': [np.median]}).sort_values(('score', 'mean'), ascending=False).reset_index(drop=True)
    means['rank'] = means.index
    means = means.round(2)
    mfp = fp.replace('all', 'means')
    means.to_csv(mfp, index=False)

"""
csvpathall = f'/deep/group/disentangle/gs_results/scores/all_dsprites_unsup_{suffix}.csv'
df = pd.read_csv(csvpathall)
df = df.drop_duplicates(['dataset', 'decoder'], keep='last')

n_clusts = [5,6]
for n_clust in n_clusts:
    df = df[df['n_clust'] == n_clust].groupby(['decoder'], as_index=False).agg({'score':[np.mean, np.std], 'n_clust': [np.median]}).sort_values(('score', 'mean'), ascending=False).reset_index(drop=True)

    df['rank'] = df.index
    df = df.round(2)
    fp = f'/deep/group/disentangle/gs_results/scores/means_n_clust_{n_clust}_dsprites_unsup_{suffix}.csv'
    df.to_csv(fp, index=False)
print(f'Go to /deep/group/disentangle/gs_results/scores/ with {suffix} on means_ prefixed csvs')
import pdb;pdb.set_trace()
"""
