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
    for dataset_name, details in datasets.items():
        print(details)
        for decoder_model, decoder_checkpoint in details["decoder_models"].items():
            for suffix in suffixes:
                if decoder_model == 'InfoGAN-CR' and suffix == 'l100':
                    suffix = None
                print(decoder_model, suffix)
                extra = f"fake_{decoder_model}_{dataset_name}"
                if suffix is not None:
                    extra += f"_{suffix}"
                to_run = f"python vis_cov.py --extra {extra} --plot "

                os.system(to_run)
