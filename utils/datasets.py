import random
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Iterable, Dict, Union, List

import torch
from torch.utils.data.dataset import Dataset
from utils.general import load_json


class WSIDataset(Dataset):
    """Basic WSI Dataset, which can obtain the features of each patch of WSIs."""

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: int = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True) -> None:
        """Initialization constructor.

        :param str or Path data_csv: A CSV file's filepath for organization WSI data, as detailed in our README
        :param Iterable[str] indices: A list containing the specified `case_id`, if None, fetching all `case_id` in the `data_csv` file
        :param int num_sample_patches: The number of sampled patches, if None, the value is the number of all patches
        :param bool fixed_size: If True, the size of the number of patches is fixed
        :param bool shuffle: If True, shuffle the order of all WSIs
        :param bool patch_random: If True, shuffle the order of patches within a WSI during reading this dataset
        :param bool preload: If True, all feature files are loaded at initialization
        """
        super(WSIDataset, self).__init__()
        self.data_csv = data_csv
        self.indices = indices
        self.num_sample_patches = num_sample_patches
        self.fixed_size = fixed_size
        self.preload = preload
        self.patch_random = patch_random

        self.samples = self.process_data()
        if self.indices is None:
            self.indices = self.samples.index.values
        if shuffle:
            self.shuffle()

        self.patch_dim = np.load(self.samples.iat[0, 0])['img_features'].shape[-1]

        if self.preload:
            self.patch_features = self.load_patch_features()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.indices[index]

        if self.preload:
            patch_feature = self.patch_features[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']

        patch_feature = self.sample_feat(patch_feature)
        if self.fixed_size:
            patch_feature = self.fix_size(patch_feature)
        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, label, case_id

    def shuffle(self) -> None:
        """Shuffle the order of WSIs. """
        random.shuffle(self.indices)

    def process_data(self):
        """Load the `data_csv` file by `indices`. """
        data_csv = pd.read_csv(self.data_csv)
        data_csv.set_index(keys='case_id', inplace=True)
        if self.indices is not None:
            samples = data_csv.loc[self.indices]
        else:
            samples = data_csv
        return samples

    def load_patch_features(self) -> Dict[str, np.ndarray]:
        """Load the all the patch features of all WSIs. """
        patch_features = {}
        for case_id in self.indices:
            patch_features[case_id] = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
        return patch_features

    def sample_feat(self, patch_feature: np.ndarray) -> np.ndarray:
        """Sample features by `num_sample_patches`. """
        num_patches = patch_feature.shape[0]
        if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
            sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
            sample_indices = sorted(sample_indices)
            patch_feature = patch_feature[sample_indices]
        if self.patch_random:
            np.random.shuffle(patch_feature)
        return patch_feature

    def fix_size(self, patch_feature: np.ndarray) -> np.ndarray:
        """Fixed the shape of each WSI feature. """
        if patch_feature.shape[0] < self.num_sample_patches:
            margin = self.num_sample_patches - patch_feature.shape[0]
            feat_pad = np.zeros(shape=(margin, self.patch_dim))
            feat = np.concatenate((patch_feature, feat_pad))
        else:
            feat = patch_feature[:self.num_sample_patches]
        return feat


class WSIWithCluster(WSIDataset):
    """A WSI Dataset with its cluster result"""

    def __init__(self,
                 data_csv: Union[str, Path],
                 indices: Iterable[str] = None,
                 num_sample_patches: int = None,
                 fixed_size: bool = False,
                 shuffle: bool = False,
                 patch_random: bool = False,
                 preload: bool = True) -> None:
        """Initialization constructor.

        :param str or Path data_csv: A CSV file's filepath for organization WSI data, the end of the filename must provide the number of clusters, as detailed in our README
        :param Iterable[str] indices: A list containing the specified `case_id`, if None, fetching all `case_id` in the `data_csv` file
        :param int num_sample_patches: The number of sampled patches, if None, the value is the number of all patches
        :param bool fixed_size: If True, the size of the number of patches is fixed
        :param bool shuffle: If True, shuffle the order of all WSIs
        :param bool patch_random: If True, shuffle the order of patches within a WSI during reading this dataset
        :param bool preload: If True, all feature files are loaded at initialization
        """
        super(WSIWithCluster, self).__init__(data_csv, indices, num_sample_patches, fixed_size, shuffle, patch_random,
                                             preload)
        # The filename of `data_csv` must provide the number of clusters at the end.
        # eg. camelyon16_10.csv, the 10 indicates the number of clusters.
        self.num_clusters = int(Path(data_csv).stem.split('_')[-1])

        if self.preload:
            self.cluster_indices = self.load_cluster_indices()

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, List[List[int]], torch.Tensor, str]:
        """Return the WSI features, cluster indices, label, and the case_id. """
        case_id = self.indices[index]

        if self.preload:
            patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
            cluster_indices = load_json(self.samples.at[case_id, 'clusters_json_filepath'])

        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, cluster_indices, label, case_id

    def load_cluster_indices(self) -> Dict[str, List[List[int]]]:
        cluster_indices = {}
        for case_id in self.indices:
            cluster_indices[case_id] = load_json(self.samples.at[case_id, 'clusters_json_filepath'])
        return cluster_indices


class ClusterFeatures(WSIWithCluster):
    """A WSI Dataset, which patches sampled by cluster result. """

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        case_id = self.indices[index]

        if self.preload:
            patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
            cluster_indices = load_json(self.samples.at[case_id, 'clusters_json_filepath'])

        patch_feature = self.sample_feat(patch_feature, cluster_indices)
        if self.fixed_size:
            patch_feature = self.fix_size(patch_feature)
        patch_feature = torch.as_tensor(patch_feature, dtype=torch.float32)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return patch_feature, label, case_id

    def sample_feat(self, patch_feature: np.ndarray, cluster_indices: List[List[int]] = None) -> np.ndarray:
        """Sample features by cluster indices. """
        if self.num_sample_patches is None:
            sample_ratio = 1.
        else:
            sample_ratio = self.num_sample_patches / patch_feature.shape[0]
        sample_indices = []
        if sample_ratio < 1:
            for c in range(self.num_clusters):
                num_patch_c = len(cluster_indices[c])
                size = int(np.rint(num_patch_c * sample_ratio))
                sample = np.random.choice(num_patch_c, size=size, replace=False)
                sample_indices.extend([cluster_indices[c][s] for s in sample])
            sample_indices = sorted(sample_indices)
            patch_feature = patch_feature[sample_indices]
        if self.patch_random:
            np.random.shuffle(patch_feature)

        return patch_feature


class WSIPhenotype(ClusterFeatures):
    """A WSI Dataset, which patches sampled by cluster result and reshape it into a tenor list with shape like image. """

    def __getitem__(self, index: int) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, str]:
        case_id = self.indices[index]

        if self.preload:
            patch_feature, cluster_indices = self.patch_features[case_id], self.cluster_indices[case_id]
        else:
            patch_feature = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
            cluster_indices = load_json(self.samples.at[case_id, 'clusters_json_filepath'])

        phenotype, mask = self.create_phenotype(patch_feature, cluster_indices)
        feat = [torch.from_numpy(f) for f in phenotype]
        mask = torch.from_numpy(mask)

        label = self.samples.at[case_id, 'label']
        label = torch.tensor(label, dtype=torch.long)
        return feat, mask, label, case_id

    def create_phenotype(self, patch_feature: np.ndarray, cluster_indices: List[List[int]]) -> \
            Tuple[List[np.ndarray], np.ndarray]:
        """
        Returns a list of phenotypes and a mask, where each phenotype is a tensor composed of features of a cluster,
        mask indicating whether a phenotype is not empty.
        """
        if self.num_sample_patches is None:
            sample_ratio = 1.
        else:
            sample_ratio = self.num_sample_patches / patch_feature.shape[0]
        phenotype = []
        mask = np.ones(shape=self.num_clusters, dtype=np.float32)
        for cluster_idx in range(self.num_clusters):
            if len(cluster_indices[cluster_idx]) == 0:
                cluster_feat = np.zeros(shape=(1, self.patch_dim), dtype=np.float32)
                mask[cluster_idx] = 0
            else:
                if sample_ratio < 1:
                    size = int(np.rint(len(cluster_indices[cluster_idx]) * sample_ratio))
                    sample = np.random.choice(len(cluster_indices[cluster_idx]), size=size, replace=False)
                    sample = sorted(sample)
                    indices = [cluster_indices[cluster_idx][i] for i in sample]
                else:
                    indices = cluster_indices[cluster_idx]
                if self.patch_random:
                    np.random.shuffle(indices)
                cluster_feat = patch_feature[indices]
            cluster_feat = np.swapaxes(cluster_feat, 1, 0)  # dim_features * num_features
            cluster_feat = np.expand_dims(cluster_feat, 1)  # dim_features * 1 * num_features
            phenotype.append(cluster_feat)  # len == num_clusters
        return phenotype, mask


def mixup(inputs: torch.Tensor, alpha: Union[float, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mix-up a batch tentor. """
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx


def get_feats(feat_list: List[torch.Tensor],
              clusters_list: List[List[List[int]]],
              action_sequence: torch.Tensor,
              feat_size: int = 1024) -> torch.Tensor:
    """Construct a WSI-Fset mentioned in our paper. """
    batch_size = len(feat_list)
    device = action_sequence.device
    feats = []
    for i in range(batch_size):
        # compute numbers of each cluster
        num_patch = feat_list[i].shape[-2]
        sample_ratio = feat_size / num_patch
        num_feats_cluster = torch.tensor([len(c) for c in clusters_list[i]], device=device)
        num_feats_cluster_size = torch.round(num_feats_cluster * sample_ratio).int()

        # compute the indices of selected features by action_sequence
        feat_coordinate_l = torch.floor(action_sequence[i] * (num_feats_cluster - num_feats_cluster_size)).int()
        feat_coordinate_r = feat_coordinate_l + num_feats_cluster_size
        indices = []
        for j, c in enumerate(clusters_list[i]):
            index = c[feat_coordinate_l[j].item():feat_coordinate_r[j].item()]
            indices.extend(index)
        indices = sorted(indices)

        # construct WSI-Fset
        per_feat = feat_list[i][:, indices, :]
        if per_feat.shape[-2] < feat_size:
            margin = feat_size - per_feat.shape[-2]
            feat_pad = torch.zeros(size=(1, margin, per_feat.shape[-1]), device=device)
            per_feat = torch.cat((per_feat, feat_pad), dim=1)
        else:
            per_feat = per_feat[:, :feat_size, :]
        feats.append(per_feat)
    feats = torch.cat(feats, 0)
    return feats
