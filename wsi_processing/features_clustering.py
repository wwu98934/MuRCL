import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.cluster import KMeans
from .utils import dump_json


def clustering(feats, num_clusters, filepath=None):
    k_means = KMeans(n_clusters=num_clusters, random_state=985).fit(feats)
    features_cluster_indices = np.expand_dims(k_means.labels_, axis=1)

    if filepath is not None:
        np.savez(file=filepath, features_cluster_indices=features_cluster_indices)
    return features_cluster_indices


def save_to_json(features_cluster_indices, num_clusters, filepath=None):
    cluster_features = [[] for _ in range(num_clusters)]
    for patch_idx, cluster_idx in enumerate(features_cluster_indices):
        cluster_features[cluster_idx.item()].append(patch_idx)
    if filepath is not None:
        dump_json(cluster_features, filepath)
    return cluster_features


def run(args):
    save_dir = Path(args.feat_dir) / f'k-means-{args.num_clusters}'
    save_dir.mkdir(parents=True, exist_ok=True)

    img_features_npz = sorted(list(Path(args.feat_dir).glob('*.npz')))
    img_features_npz_bar = tqdm(img_features_npz)
    for i, feat_npz in enumerate(img_features_npz):
        case_id = feat_npz.stem
        npz_filepath = save_dir / f'{case_id}.npz'
        json_filepath = save_dir / f'{case_id}.json'
        if npz_filepath.exists() and not args.exist_ok:
            print(f"{npz_filepath} is exists!")
            continue

        feat_dict = np.load(str(feat_npz))
        if feat_dict['img_features'].shape[0] < args.num_clusters:
            print(f"{feat_npz.stem}'s number of features < number of clusters, can't clustering.")
            continue

        # clustering and save as .npz file and .json file
        features_cluster_indices = clustering(feat_dict['img_features'], args.num_clusters, filepath=npz_filepath)
        save_to_json(features_cluster_indices, args.num_clusters, filepath=json_filepath)

        img_features_npz_bar.set_description(
            f"{i + 1:3}/{len(img_features_npz):3}, {feat_npz.stem}'s features: {feat_dict['img_features'].shape[0]}"
        )
        img_features_npz_bar.update()
    img_features_npz_bar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feat_dir', type=str, default='', help="the directory containing features files(.npz)")
    parser.add_argument('--num_clusters', type=int, default=10, help='the numbers of k-means clusters')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
