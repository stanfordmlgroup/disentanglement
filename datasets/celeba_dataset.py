import os
import pandas as pd

from disentanglement_lib.data.ground_truth import ground_truth_data


class CelebA(ground_truth_data.GroundTruthData):

    """CelebA dataset.

    Dependent factors.
    Following convention from disentanglement_lib
    """
    
    def __init__(self, isHQ=False):
        self.isHQ = isHQ
        self.factors_path = '/deep/group/gen-eval/model-training/data/celeba-dataset/list_attr_celeba.csv'
        if isHQ:
            self.data_path = '/deep/downloads/celebahq'
            self.alias_path = self.data_path + '/image_list.txt'
        else:
            self.data_path = '/deep/group/gen-eval/model-training/data/celeba-dataset/64_crop_splits/train'

        self.factors = pd.read_csv(self.factors_path)
        self.images = [p for p in os.listdir(self.data_path) if p.endswith('.jpg')]
        
        self.factor_names = list(self.factors.columns)[1:] # This removes image id
        self.set_factor_dicts()
        
        # Compute for properties
        self.num_factors_ = len(self.factor_names)
        self.factors_num_values_ = [2 for _ in range(self.num_factors)] # All factors are binary [1, -1]
   

    @property
    def num_factors(self):
        return self.num_factors_

    @property
    def factors_num_values(self):
        return self.factors_num_values_

    @property
    def observation_shape(self):
        return [256, 256, 3] if self.isHQ else [64, 64, 3]
        
    def set_factor_dicts(self):
        """This is for readability when visualizing/saving factor names, as well as faster getitem. If CelebAHQ, use alias."""
        self.factor_name2id, self.factor_id2name = {}, {}
        for i, f in enumerate(self.factor_names):
            self.factor_name2id[f] = i
            self.factor_id2name[i] = f
        if self.isHQ:
            # Trade image_ids
            alias = pd.read_csv(self.alias_path, delimiter='\t')
            alias_cols = list(alias.columns)
            merged = alias.merge(self.factors, how='left', left_on='orig_file', right_on='image_id')
            merged['hq_image_id'] = merged['idx'].apply(lambda x: int(x)+1)
            merged['hq_image_id'] = merged['hq_image_id'].apply(lambda x: '{0:0>5}.jpg'.format(x))
            merged = merged.drop(columns=['image_id'] + alias_cols)
            self.factor_imagename2label = merged.set_index('hq_image_id').T.to_dict('list')
        else:
            self.factor_imagename2label = self.factors.set_index('image_id').T.to_dict('list')

        return

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        # Return a random row in df
        raise Exception('Handle CelebA differently.')

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        # Return random set of rows in df with the given factors
        raise Exception('Handle CelebA differently.')

    def sample(self, num, random_state):
        """Sample a batch of factors Y and observations X."""
        factors = self.sample_factors(num, random_state)
        return factors, self.sample_observations_from_factors(factors, random_state)

    def sample_observations(self, num, random_state):
        """Sample a batch of observations X."""
        return self.sample(num, random_state)[1]
