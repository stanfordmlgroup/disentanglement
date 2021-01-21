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

from .dataset import Dataset
from sklearn.model_selection import train_test_split
import itertools

class ClassificationDataset(Dataset):
    def __init__(self, dataset_name, batch_size):
        super().__init__(dataset_name, batch_size)
        
        self.dataset_name = dataset_name
        self._set_getitem()


    def _set_getitem(self):
        if 'celeba' in self.dataset_name:
            self.getitem = self._getitem_from_names
        else:
            self._set_combs()
            self.getitem = self._getitem_from_combs

    def _getitem_from_names(self, index):
        
        # Load image
        image_name = self.images[index]
        image_path = Path(self.dataset.data_path) / image_name
        image = Image.open(image_path)
        image = self.transform(image) 

        # Find row and load label
        if not self.dataset.isHQ:
            image_name = image_name.replace('_crop.jpg', '.jpg')
        label = self.dataset.factor_imagename2label[image_name]

        label = np.array(label).astype(int)
        label = label * (label > 0) # Turn -1 to 0

        return image, label
        
    def _set_combs(self):
        # Get all combinations of latent factor values
        self.all_factors_values = [list(range(self.dataset.factors_num_values[i])) for i in range(self.dataset.num_factors)]
        self.combs = list(itertools.product(*self.all_factors_values))


    def _getitem_from_combs(self, index):
        # Changes to be deterministic based on comb param
        comb = self.combs[index]

        # Image
        factor_values = np.array(comb)
        image = self.dataset.sample_observations_from_factors(factor_values, self.random_state)
        image = self.transform(image[0])

        # Label - no normalization, keep classes in tact for cls
        label = factor_values
        
        return image, label

    
    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        return self.getitem(index)
   
