import os

if __name__ == "__main__":
    datasets = {
        "dsprites": {
            "decoder_models": {
                'btcvae': None, 
                'factor_VAE': None, 
                'VAE': None, 
                'betaH_VAE': None,
                'betaB_VAE': None,
                'InfoGAN-CR': None,
            },
        },
        "celeba": {
            "decoder_models": {
                'btcvae': None, 
                'factor_VAE': None, 
                'VAE': None, 
                'betaH_VAE': None,
                'betaB_VAE': None,
                'InfoGAN-CR': None,
                'BEGAN': None,
                'WGAN': None,
            },
        },
        "celebahq": {
            "decoder_models": {
                'PGAN': None, 
                'StyleGAN': None, 
            },
        },
    }
suffixes = ['l100', '2', '3', '4', '5']
sups = ['sup', 'unsup']
for dataset_name, details in datasets.items():
    print(details)
    for decoder_model, decoder_checkpoint in details["decoder_models"].items():
        for suffix in suffixes:
            print(decoder_model, suffix)
            if decoder_model == 'InfoGAN-CR' and suffix == 'l100':
                suffix = None
            for sup in sups: 
                to_run = f"python gen_cov.py --dataset {dataset_name} --decoder_model {decoder_model} --search_n_clusters --save_scores --plot --scores_file {dataset_name}_{sup}_var2 "
                if suffix is not None:
                    to_run += f" --suffix {suffix} "
                if sup == 'sup':
                    print('supervised')
                    to_run += f' --sup '

                os.system(to_run)
