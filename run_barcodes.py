import os

if __name__ == "__main__":
    datasets = {
        "dsprites": {
            "decoder_models": {
                'btcvae': None, 
                'factor_VAE': None, 
                'VAE': None, 
                'betaH_VAE': None,
                'betaB_VAE': None
            },
        },
        "celeba": {
            "decoder_models": {
                'btcvae': None, 
                'factor_VAE': None, 
                'VAE': None, 
                'betaH_VAE': None,
                'betaB_VAE': None,
                'WGAN': None,
                'BEGAN': None,
                #'InfoGAN-CR': None,
            },
        },
        "celebahq": {
            "decoder_models": {
                'PGAN': None, 
                'StyleGAN': None, 
            },
        },
    }
    suffixes = ['2']
    for suffix in suffixes:
        for dataset_name, details in datasets.items():
            print(details)
            for decoder_model, decoder_checkpoint in details["decoder_models"].items():
                print(decoder_model)

                to_run = f"CUDA_VISIBLE_DEVICES=1 python barcode_gen.py --dataset {dataset_name} --decoder_model {decoder_model} --suffix {suffix} "

                os.system(to_run)
