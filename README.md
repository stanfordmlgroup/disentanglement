### Official repository for the ICLR 2021 paper _Evaluating the Disentanglement of Deep Generative Models with Manifold Topology_ 
Sharon Zhou, Eric Zelikman, Fred Lu, Andrew Y. Ng, Gunnar Carlsson, and Stefano Ermon <br>
Computer Science & Math departments, Stanford University

Read the [paper](https://arxiv.org/abs/2006.03680).

### Setup
Requirements and environment files are in `env/`

### Run
`run_barcodes.py`: iterate over various models and datasets to generate their corresponding persistence barcodes <br>
`run_vis_cov.py`: run `vis_cov.py` for all datasets and decoder combinations, where `vis_cov.py` takes saved barcodes and produces W. RLT figures <br>
`run_gen_cov.py`: run `gen_cov.py` for all datasets and decoder combinations, where `gen_cov.py` calculates the similarity matrices between two barcodes, then finds spectrally coclustered biclusters, and finally calculates the scores <br>
`parse_scores.py`: aggregate the results of existing runs <br>

### Other utilities
`global_models.py`: encoders and decoders of various generative models <br>
`utils.py`: collection of utilities for sampling, model-loading, conversions <br>

`args/`: argument parsers/options for different models <br>
`datasets/`: resources for handling our datasets<br>
`env/`: requirements files, various pip/conda details<br>
`supplement/`: scripts for some appendix experiments<br>

### External libraries
[Geometry score](https://github.com/KhrulkovV/geometry-score), modified to include Wasserstein features: `gs/` <br>
[Disentanglement VAEs lib](https://github.com/YannDubs/disentangling-vae): `disentangling_vae/` <br>
[Pytorch GAN zoo](https://github.com/facebookresearch/pytorch_GAN_zoo): `models/` <br>
[StyleGAN resources](https://github.com/podgorskiy/ALAE): `alae/` <br>
Various CelebA GAN implementations: `celeba_gan/` <br>

### Deprecated
Some deprecated code referring to `transcoder` may be present in certain files. This constitutes another set of experiments that we were running in parallel for mapping factors of variation to latents. Thus is research - perhaps future work ;)

### Acknowledgements
We would like to thank Torbjorn Lundh and Samuel Bengmark for their helpful feedback and encouragement in the preparation of our manuscript.


