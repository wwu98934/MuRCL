import os
import cv2
import torch
import argparse
import openslide
import numpy as np
from pathlib import Path
from xml.dom import minidom

from utils.datasets import WSIDataset
from utils.general import load_json
from models import clam


def get_datasets(args):
    train_set = WSIDataset(data_csv=args.data_csv, shuffle=False, preload=args.preload)
    return train_set, train_set.patch_dim, len(train_set)


def create_model(args, dim_patch):
    print(f"Creating model {args.arch}...")
    if args.arch == 'CLAM_SB':
        model = clam.CLAM_SB(
            gate=True,
            size_arg=args.size_arg,
            dropout=True,
            k_sample=args.k_sample,
            n_classes=args.num_classes,
            subtyping=True,
            in_dim=dim_patch
        )
    else:
        raise ValueError(f'args.arch error, {args.arch}. ')

    assert args.checkpoint is not None, f"train method is {args.train_method}, expected get checkpoint, but got None. "
    checkpoint = torch.load(args.checkpoint)
    state_dict = checkpoint['model_state_dict']

    # print("before:")
    # for k in list(state_dict.keys()):
    #     print(f"key: {k}")
    for k in list(state_dict.keys()):
        if k.startswith('module.encoder'):
            if not k.startswith('module.encoder.fc') and not k.startswith('module.encoder.classifiers'):
                state_dict[k[len("module.encoder."):]] = state_dict[k]
            del state_dict[k]
        elif k.startswith('encoder'):
            if not k.startswith('encoder.fc') and not k.startswith('encoder.classifier'):
                state_dict[k[len("encoder."):]] = state_dict[k]
            del state_dict[k]
    # print("after:")
    # for k in list(state_dict.keys()):
    #     print(f"key: {k}")

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"missing_keys: {msg.missing_keys}")

    dim_in = model.classifiers.in_features
    model.classifiers = torch.nn.Linear(dim_in, args.num_classes)

    model = torch.nn.DataParallel(model).cuda()

    assert model is not None, "creating model failed. "
    print(f"Total params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    return model


def get_three_points(x_step, y_step, size):
    top_left = (int(x_step * size), int(y_step * size))
    bottom_right = (int(top_left[0] + size), int(top_left[1] + size))
    center = (int((top_left[0] + bottom_right[0]) // 2), int((top_left[1] + bottom_right[1]) // 2))
    return top_left, bottom_right, center


def load_annotations_xml(annotations_xml):
    dom = minidom.parse(annotations_xml)
    root = dom.documentElement
    annotations = root.getElementsByTagName('Annotation')

    contours = []
    for a in annotations:
        coords = a.getElementsByTagName('Coordinates')[0].getElementsByTagName('Coordinate')
        contour = np.array([[c.getAttribute('X'), c.getAttribute('Y')] for c in coords], dtype=np.float)
        contour = np.expand_dims(contour, 1)
        contours.append(contour)
        # print(contour.shape)
    return contours


def create_heatmap(coord_filepath, attention, slide_level=-1, contours=None):
    """
    create the heatmap of WSI with coord json file and attention scores of patches.

    :param coord_filepath: coord filepath
    :param attention: attention scores of patches
    :param slide_level: the sample level of slide
    :param contours: the contours of ROI's annotation
    :return: a heatmap
    """
    # read some necessary variables from json file of coord
    coord_dict = load_json(coord_filepath)
    slide_filepath = coord_dict['slide_filepath']
    num_patches = coord_dict['num_patches']
    coords = coord_dict['coords']
    patch_size_level0 = coord_dict['patch_size_level0']
    slide = openslide.open_slide(slide_filepath)
    thumbnail = slide.get_thumbnail(slide.level_dimensions[slide_level]).convert('RGB')
    thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
    level_downsample = slide.level_downsamples[slide_level]
    assert num_patches == len(coords) == len(attention), f"{num_patches}-{len(coords)}-{len(attention)}"

    # scale the attention to [0, 1] and create the color mapping
    attention = np.uint8(255 * ((attention - np.min(attention)) / (np.max(attention) - np.min(attention))))
    colors = cv2.applyColorMap(attention, cv2.COLORMAP_JET)

    # create the blank heatmap with white background
    heatmap = np.ones(thumbnail.shape, dtype=np.uint8) * 255
    for i in range(num_patches):
        row, col = coords[i]['row'], coords[i]['col']
        points = get_three_points(col, row, patch_size_level0 / level_downsample)
        c = (int(colors[i, 0, 0]), int(colors[i, 0, 1]), int(colors[i, 0, 2]))
        # draw the rectangle filled with attention color
        cv2.rectangle(heatmap, points[0], points[1], color=c, thickness=-1)
    # mix the heatmap and slide thumbnail
    heatmap = cv2.addWeighted(heatmap, 0.5, thumbnail, 0.5, 0)

    # draw the ROI region if contours exist
    if contours is not None:
        contours = [np.asarray(c / level_downsample).astype(np.int32) for c in contours]
        heatmap = cv2.drawContours(heatmap, contours, -1, (0, 255, 255), thickness=5)

    return heatmap


def run(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    if not args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')

    # None or a list of slide indices you interested
    case_id_list = None

    dataset, dim_patch, data_size = get_datasets(args)
    model = create_model(args, dim_patch)
    model.eval()
    for patch_feature, _, case_id in dataset:
        # just creating heatmaps that you interested
        if case_id_list is not None and case_id not in case_id_list:
            continue

        heatmap_filepath = Path(args.save_dir) / f'{case_id}.png'
        # auto skip
        if heatmap_filepath.exists() and not args.exist_ok:
            continue

        # compute attention scores for every patch features
        with torch.no_grad():
            patch_feature = patch_feature.to(args.device)
            attention = model.module.bag_forward(patch_feature, attention_only=True).cpu().numpy().reshape(-1)

        # get the json file of WSI's coord
        coord_filepath = Path(args.coord_dir) / f'{case_id}.json'
        if not coord_filepath.exists():
            continue

        # get the annotation of WSI
        annotation_xml = Path(args.annotation_dir) / f'{case_id}.xml'
        if annotation_xml.exists() and args.draw_contours:
            contours = load_annotations_xml(str(annotation_xml))
        else:
            contours = None

        # creating and saving the attention heatmap
        heatmap = create_heatmap(str(coord_filepath), attention, slide_level=args.slide_level, contours=contours)
        cv2.imwrite(str(heatmap_filepath), heatmap)
        print(f'{case_id} done!')


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_csv', type=str,
                        default='/path/to/data_csv.csv',
                        help='')
    parser.add_argument('--coord_dir', type=str, default='/path/to/coord',
                        help='')
    parser.add_argument('--annotation_dir', type=str,
                        default='',
                        help='')
    parser.add_argument('--preload', action='store_true', default=False)
    # Architecture
    parser.add_argument('--arch', default='CLAM_SB', type=str, help='model name')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--checkpoint', default=None, type=str)
    # CLAM
    parser.add_argument('--size_arg', type=str, default='small', choices=['small', 'big'])
    parser.add_argument('--k_sample', type=int, default=8)
    # Save
    parser.add_argument('--save_dir', type=str, default='./heatmaps')
    parser.add_argument('--draw_contours', action='store_true', default=False,
                        help="drawing the ROI of WSI if it has annotations")
    parser.add_argument('--slide_level', type=int, default=4)
    parser.add_argument('--exist_ok', action='store_true', default=False)
    # Global
    parser.add_argument('--device', default='2',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
