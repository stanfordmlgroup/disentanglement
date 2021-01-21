''' Custom loader for the TF InfoGAN-CR decoder for generating barcodes.
This is used by tf_barcode_gen.py in the parent directory.
'''

import os
import numpy as np

import tensorflow as tf
from .gan.load_data import load_celeba
from .gan.latent import UniformLatent, JointLatent
from .gan.network import Decoder, InfoGANDiscriminator, \
    CrDiscriminator
from .gan.infogan_cr import INFOGAN_CR
from .config_generate_latent_trans import config


def load_infogan_celeba_decoder():
    # hack configurator
    cfg = config['global_config']
    cfg.update(config['test_config'][0])
    # work_dir = "/deep/group/disentangle/InfoGAN-CR/celeba_results/cr_coe_increase-1.0,cr_coe_increase_batch-80000,cr_coe_increase_times-1,cr_coe_start-0.0,gap_decrease-0.0,gap_decrease_batch-1,gap_decrease_times-0,gap_start-0.0,info_coe_de-2.0,info_coe_infod-2.0,run-0,"
    work_dir = "/deep/group/disentangle/InfoGAN-CR/celeba_results/test_64"

    # data is still needed as placeholder for InfoGAN-CR when sampling
    data = np.random.randn(50000, 32, 32, 3)
    _, height, width, depth = data.shape

    latent_list = []

    for i in range(cfg["uniform_reg_dim"]):
        latent_list.append(UniformLatent(
            in_dim=1, out_dim=1, low=-1.0, high=1.0, q_std=1.0,
            apply_reg=True))
    if cfg["uniform_not_reg_dim"] > 0:
        latent_list.append(UniformLatent(
            in_dim=cfg["uniform_not_reg_dim"],
            out_dim=cfg["uniform_not_reg_dim"],
            low=-1.0, high=1.0, q_std=1.0,
            apply_reg=False))
    latent = JointLatent(latent_list=latent_list)

    decoder = Decoder(
        output_width=width, output_height=height, output_depth=depth)
    infoGANDiscriminator = \
        InfoGANDiscriminator(output_length=latent.reg_out_dim)
    crDiscriminator = \
        CrDiscriminator(output_length=latent.num_reg_latent)

    checkpoint_dir = os.path.join(work_dir, "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    sample_dir = os.path.join(work_dir, "sample")
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
    time_path = os.path.join(work_dir, "time.txt")
    metric_path = os.path.join(work_dir, "metric.csv")

    run_config = tf.ConfigProto()
    sess = tf.Session(config=run_config)
    # with tf.Session(config=run_config) as sess:
    metric_callbacks = []
    gan = INFOGAN_CR(
        sess=sess,
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        time_path=time_path,
        epoch=cfg["epoch"],
        batch_size=cfg["batch_size"],
        data=data,
        vis_freq=cfg["vis_freq"],
        vis_num_sample=cfg["vis_num_sample"],
        vis_num_rep=cfg["vis_num_rep"],
        latent=latent,
        decoder=decoder,
        infoGANDiscriminator=infoGANDiscriminator,
        crDiscriminator=crDiscriminator,
        gap_start=cfg["gap_start"],
        gap_decrease_times=cfg["gap_decrease_times"],
        gap_decrease=cfg["gap_decrease"],
        gap_decrease_batch=cfg["gap_decrease_batch"],
        cr_coe_start=cfg["cr_coe_start"],
        cr_coe_increase_times=cfg["cr_coe_increase_times"],
        cr_coe_increase=cfg["cr_coe_increase"],
        cr_coe_increase_batch=cfg["cr_coe_increase_batch"],
        info_coe_de=cfg["info_coe_de"],
        info_coe_infod=cfg["info_coe_infod"],
        metric_callbacks=metric_callbacks,
        metric_freq=cfg["metric_freq"],
        metric_path=metric_path,
        output_reverse=cfg["output_reverse"],
        de_lr=cfg["de_lr"],
        infod_lr=cfg["infod_lr"],
        crd_lr=cfg["crd_lr"],
        summary_freq=cfg["summary_freq"])

    gan.build()
    gan.load()
    return sess, gan
