import time
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F

from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import shapes3d
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import norb

from .celeba_dataset import CelebA

class Dataset(data.Dataset):
    def __init__(self, dataset_name, batch_size, random_seed=42):
        self.dataset_name = dataset_name
        self.random_state = np.random.RandomState(random_seed)
        
        # Use disentanglement_lib datasets
        self.dataset = self.load_dataset(dataset_name)
        self.images = self.dataset.images
        self.x_shape = self.dataset.observation_shape
        self.ns = self.dataset.num_factors

        self.transform = self._set_transforms()
        self._set_factor_normalization()

    def _set_factor_normalization(self):
        m, s = [], []
        for factor_size in self.dataset.factors_num_values:
            factor_values = list(range(factor_size))
            m.append(np.mean(factor_values))
            s.append(np.std(factor_values))
        self.m, self.s = np.array(m), np.array(s)
       
    def normalize_factors(self, factors):
        return (factors - self.m) / self.s

    def _set_transforms(self, use_normalize=False):

        # Apply transforms
        transforms_list = []

        # Normalize to the mean and standard deviation all pretrained
        # torchvision models expect
        #normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        # Transform: ToTensor()
        # 1) transforms PIL image in range [0,255] to [0,1],
        # 2) tranposes [H, W, C] to [C, H, W]
        if use_normalize:
            pass
            #transforms_list += [transforms.ToTensor(), normalize]
        else:
            transforms_list += [transforms.ToTensor()]

        transform = transforms.Compose([t for t in transforms_list if t])

        return transform

    def load_dataset(self, dataset_name):
      if dataset_name == "dsprites":
        # 5 factors
        return dsprites.DSprites(list(range(1, 6)))
      elif dataset_name == "shapes3d":
        # 6 factors
        return shapes3d.Shapes3D()
      elif dataset_name == "norb":
        # 4 factors + 1 nuisance (which we'll handle via n_dim=2)
        return norb.SmallNORB()
      elif dataset_name == "cars3d":
        # 3 factors
        return cars3d.Cars3D()
      elif dataset_name == "mpi3d":
        # 7 factors
        return mpi3d.MPI3D()
      elif dataset_name == "scream":
        # 5 factors + 2 nuisance (handled as n_dim=2)
        return dsprites.ScreamDSprites(list(range(1, 6)))
      elif dataset_name == "celeba":
        # Dependent factors...
        return CelebA()
      elif dataset_name == "celebahq":
        # Dependent factors...
        return CelebA(isHQ=True)
      else:
          print(f'{dataset_name} does not exist')
    

    def get_image_label_pair(self):
        # HACK: Overriding the index TODO: custom sampler/iterabledatset
        # Image
        sampled_factor_values = self.dataset.sample_factors(1, self.random_state)
        image = self.dataset.sample_observations_from_factors(sampled_factor_values, self.random_state)
        image = self.transform(image[0])

        # Label
        factors = self.normalize_factors(sampled_factor_values)
        label = factors[0]

        return image, label

    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        return self.get_image_label_pair()
        
   
