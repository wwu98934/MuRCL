import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data.dataset import Dataset
from utils.general import load_json


class WSIDataset(Dataset):
    def __init__(self, data_csv, indices=None, num_sample_patches=None, fixed_size=False, shuffle=False,
                 patch_random=False, preload=True):
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
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

    def shuffle(self):
        random.shuffle(self.indices)

    def process_data(self):
        data_csv = pd.read_csv(self.data_csv)
        data_csv.set_index(keys='case_id', inplace=True)
        if self.indices is not None:
            samples = data_csv.loc[self.indices]
        else:
            samples = data_csv
        return samples

    def load_patch_features(self):
        patch_features = {}
        for case_id in self.indices:
            patch_features[case_id] = np.load(self.samples.at[case_id, 'features_filepath'])['img_features']
        return patch_features

    def sample_feat(self, patch_feature):
        num_patches = patch_feature.shape[0]
        if self.num_sample_patches is not None and num_patches > self.num_sample_patches:
            sample_indices = np.random.choice(num_patches, size=self.num_sample_patches, replace=False)
            sample_indices = sorted(sample_indices)
            patch_feature = patch_feature[sample_indices]
        if self.patch_random:
            np.random.shuffle(patch_feature)
        return patch_feature

    def fix_size(self, patch_feature):
        if patch_feature.shape[0] < self.num_sample_patches:
            margin = self.num_sample_patches - patch_feature.shape[0]
            feat_pad = np.zeros(shape=(margin, self.patch_dim))
            feat = np.concatenate((patch_feature, feat_pad))
        else:
            feat = patch_feature[:self.num_sample_patches]
        return feat


class WSIWithCluster(WSIDataset):
    def __init__(self, data_csv, indices=None, num_sample_patches=None, fixed_size=False, shuffle=False,
                 patch_random=False, preload=True):
        super(WSIWithCluster, self).__init__(data_csv, indices, num_sample_patches, fixed_size, shuffle, patch_random,
                                             preload)
        self.num_clusters = int(Path(data_csv).stem.split('_')[-1])

        if self.preload:
            self.cluster_indices = self.load_cluster_indices()

    def __getitem__(self, index):
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

    def load_cluster_indices(self):
        cluster_indices = {}
        for case_id in self.indices:
            cluster_indices[case_id] = load_json(self.samples.at[case_id, 'clusters_json_filepath'])
        return cluster_indices


class ClusterFeatures(WSIWithCluster):
    def __getitem__(self, index):
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

    def sample_feat(self, patch_feature, cluster_indices=None):
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
    def __getitem__(self, index):
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

    def create_phenotype(self, patch_feature, cluster_indices):
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
                # print(cluster_features_indices[cluster_idx])
                cluster_feat = patch_feature[indices]
            # print(f"cluster_feat: {cluster_feat.shape}")
            cluster_feat = np.swapaxes(cluster_feat, 1, 0)  # dim_features * num_features
            # print(f"cluster_feat: {cluster_feat.shape}")
            cluster_feat = np.expand_dims(cluster_feat, 1)  # dim_features * 1 * num_features
            phenotype.append(cluster_feat)  # len == num_clusters
        return phenotype, mask


def mixup(inputs, alpha):
    batch_size = inputs.shape[0]
    lambda_ = alpha + torch.rand(size=(batch_size, 1), device=inputs.device) * (1 - alpha)
    rand_idx = torch.randperm(batch_size, device=inputs.device)
    a = torch.stack([lambda_[i] * inputs[i] for i in range(batch_size)])
    b = torch.stack([(1 - lambda_[i]) * inputs[rand_idx[i]] for i in range(batch_size)])
    outputs = a + b
    return outputs, lambda_, rand_idx


def get_feats(feat_list, clusters_list, action_sequence, feat_size=256):
    """Get small patch of the original image"""
    # print('------------------------------------------')
    batch_size = len(feat_list)
    device = action_sequence.device
    # print(f"action_sequence: {action_sequence}")
    feats = []
    for i in range(batch_size):
        # 1 * feature_map * h * w
        num_patch = feat_list[i].shape[-2]
        # print(f"num_patch: {num_patch}")
        sample_ratio = feat_size / num_patch
        # print(f"sample_ratio: {sample_ratio}")
        num_feats_cluster = torch.tensor([len(c) for c in clusters_list[i]], device=device)
        # print(f"num_feats_cluster: {num_feats_cluster}")
        num_feats_cluster_size = torch.round(num_feats_cluster * sample_ratio).int()
        # print(f"num_feats_cluster_size: {num_feats_cluster_size}")
        feat_coordinate_l = torch.floor(action_sequence[i] * (num_feats_cluster - num_feats_cluster_size)).int()
        # print(f"feat_coordinate_l: {feat_coordinate_l}")
        feat_coordinate_r = feat_coordinate_l + num_feats_cluster_size
        # print(f"feat_coordinate_r: {feat_coordinate_r}")
        indices = []
        for j, c in enumerate(clusters_list[i]):
            index = c[feat_coordinate_l[j].item():feat_coordinate_r[j].item()]
            # print(f"index[{j}]: {index}")
            indices.extend(index)
        indices = sorted(indices)
        # print(f"indices: {len(indices)}")

        # print(f"feat: {feat_list[i].shape}")
        per_feat = feat_list[i][:, indices, :]
        # print(f"per_feat: {per_feat.shape}")
        if per_feat.shape[-2] < feat_size:
            margin = feat_size - per_feat.shape[-2]
            feat_pad = torch.zeros(size=(1, margin, per_feat.shape[-1]), device=device)
            per_feat = torch.cat((per_feat, feat_pad), dim=1)
        else:
            per_feat = per_feat[:, :feat_size, :]
        # print(f"per_feat: {per_feat.shape}")
        feats.append(per_feat)
    feats = torch.cat(feats, 0)
    # print(f"feats: {feats.shape}")
    # print('------------------------------------------')
    return feats


# Test -----------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    pass
