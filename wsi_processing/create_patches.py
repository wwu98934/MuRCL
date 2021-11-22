import cv2
import json
import argparse
import openslide
import numpy as np

from pathlib import Path

import torch

from filters import adaptive, otsu, RGB_filter
from .utils import get_three_points, keep_patch, out_of_bound


def tiling(slide_filepath, magnification, patch_size, scale_factor=32, tissue_thresh=0.35, method='rgb',
           overview_level=-1, coord_dir=None, overview_dir=None, mask_dir=None, patch_dir=None, filename=None):
    """
    tiling a WSI into multiple patches.

    :param slide_filepath: the file path of slide
    :param magnification: the magnification
    :param patch_size: the patch size
    :param scale_factor: scale wsi to down-sampled image for judging tissue percent of each patch
    :param tissue_thresh: the ratio of tissue region of a patch
    :param method: the filtering algorithm used
    :param overview_level: the down-sampling level of overview image
    :param coord_dir: the directory to save `coord` file
    :param overview_dir: the directory to save overview image
    :param mask_dir: the directory to save mask image
    :param patch_dir: the directory to save patch image
    :param filename: the filename to save
    :return: None
    """
    # get OpenSlide object by `openslide`, and specify the magnification at level 0
    slide = openslide.open_slide(str(slide_filepath))
    if 'aperio.AppMag' in slide.properties.keys():
        level0_magnification = int(slide.properties['aperio.AppMag'])
    elif 'openslide.mpp-x' in slide.properties.keys():
        level0_magnification = 40 if int(np.floor(float(slide.properties['openslide.mpp-x']) * 10)) == 2 else 20
    else:
        level0_magnification = 40

    if level0_magnification < magnification:
        print(f"{level0_magnification}<{magnification}? magnification should <= level0_magnification.")
        return
    # compute the patch size at level 0 (maximum magnification)
    patch_size_level0 = int(patch_size * (level0_magnification / magnification))

    # get thumbnail to save overview image
    if overview_dir is not None:
        thumbnail = slide.get_thumbnail(slide.level_dimensions[overview_level]).convert('RGB')
        thumbnail = cv2.cvtColor(np.asarray(thumbnail), cv2.COLOR_RGB2BGR)
    else:
        thumbnail = None

    # create the directory to save patches
    if patch_dir is not None:
        patch_dir = patch_dir / filename
        patch_dir.mkdir(parents=True, exist_ok=True)

    # create mask image and background color by specified filtering algorithm
    mask_filepath = str(mask_dir / f'{filename}.png') if mask_dir is not None else None
    if method == 'adaptive':
        mask, color_bg = adaptive(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'otsu':
        mask, color_bg = otsu(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    elif method == 'rgb':
        mask, color_bg = RGB_filter(slide, mask_downsample=scale_factor, mask_filepath=mask_filepath)
    else:
        raise ValueError(f"filter method is wrong, {method}. ")
    # compute the number of step along x (column) and y (row)
    mask_w, mask_h = mask.size
    mask = cv2.cvtColor(np.asarray(mask), cv2.COLOR_GRAY2BGR)
    mask_patch_size = int(((patch_size_level0 // scale_factor) * 2 + 1) // 2)
    num_step_x = int(mask_w // mask_patch_size)
    num_step_y = int(mask_h // mask_patch_size)

    coord_list = []  # a list to save coord of each patch
    print(f"Processing {filename}...")
    for row in range(num_step_y):
        for col in range(num_step_x):
            # get the patch image at mask
            points_mask = get_three_points(col, row, mask_patch_size)
            row_start, row_end = points_mask[0][1], points_mask[1][1]
            col_start, col_end = points_mask[0][0], points_mask[1][0]
            patch_mask = mask[row_start:row_end, col_start:col_end]
            if keep_patch(patch_mask, tissue_thresh, color_bg):  # decide keep or drop the patch by `tissue_thresh`
                points_level0 = get_three_points(col, row, patch_size_level0)
                if out_of_bound(slide.dimensions[0], slide.dimensions[1], points_level0[1][0], points_level0[1][1]):
                    continue
                # a coord consists of the patch's index of row and column, and coordinate at level 0
                coord_list.append({'row': row, 'col': col, 'x': points_level0[0][0], 'y': points_level0[0][1]})
                # draw a rectangle at thumbnail
                if overview_dir is not None:
                    points_thumbnail = get_three_points(col, row,
                                                        patch_size_level0 / slide.level_downsamples[overview_level])
                    cv2.rectangle(thumbnail, points_thumbnail[0], points_thumbnail[1], color=(0, 0, 255), thickness=3)
                # save patch image.
                # We strongly recommend not saving patch images,
                # because this process takes a long time and takes up disk space.
                # If you want to get patch images, you can always get them using the `slide.read_region(args)` function
                # and the parameters in our saved coord file.
                if patch_dir is not None:
                    patch_level0 = slide.read_region(location=points_level0[0], level=0,
                                                     size=(patch_size_level0, patch_size_level0)).convert('RGB')
                    patch = patch_level0.resize(size=(patch_size, patch_size))
                    patch.save(str(patch_dir / f'{row}_{col}.png'))
    # A `coord` file is a Python dictionary that contains the following:
    coord_dict = {
        'slide_filepath': str(slide_filepath),
        'magnification': magnification,
        'magnification_level0': level0_magnification,
        'num_row': num_step_y,
        'num_col': num_step_x,
        'patch_size': patch_size,
        'patch_size_level0': patch_size_level0,
        'num_patches': len(coord_list),
        'coords': coord_list
    }
    with open(coord_dir / f'{filename}.json', 'w', encoding='utf-8') as fp:
        json.dump(coord_dict, fp)
    if thumbnail is not None:
        cv2.imwrite(str(overview_dir / f'{filename}.png'), thumbnail)
    print(f"{filename} | mag0: {level0_magnification} | (rows, cols): {num_step_y}, {num_step_x} | "
          f"patch_size: {patch_size} | num_patches: {len(coord_list)}")


def run(args):
    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    coord_dir = save_dir / 'coord'
    coord_dir.mkdir(parents=True, exist_ok=True)
    if args.overview:
        overview_dir = save_dir / 'overview'
        overview_dir.mkdir(parents=True, exist_ok=True)
    else:
        overview_dir = None
    if args.save_mask:
        mask_dir = save_dir / 'mask'
        mask_dir.mkdir(parents=True, exist_ok=True)
    else:
        mask_dir = None
    # strongly recommend not saving patch images
    if args.save_patch:
        patch_dir = save_dir / 'patch'
        patch_dir.mkdir(parents=True, exist_ok=True)
    else:
        patch_dir = None

    # prepare the WSI file path
    slide_filepath_list = sorted(list(Path(args.slide_dir).rglob(f'*{args.wsi_format}')))
    # if need to filter files, add code here. example:
    # slide_filepath_list = [x for x in slide_filepath_list if str(x).find('DX') != -1]
    num_slide = len(slide_filepath_list)
    print(f"Slide number: {num_slide}.\n")
    # stop
    print(f"Start tiling ...")
    for slide_idx, slide_filepath in enumerate(slide_filepath_list):
        # the filename to save, and as the index of each WSI
        if args.specify_filename:
            filename = slide_filepath.stem[args.filename_l:args.filename_r]
        else:  # If not specified, the WSI file name is used as the index of the WSI
            filename = slide_filepath.stem
        # auto skip
        if (coord_dir / f'{filename}.json').exists() and not args.exist_ok:
            print(f"{str(coord_dir / f'{filename}.json')} exists, skip!")
            continue

        print(f"{slide_idx + 1:3}/{num_slide}, Processing {filename}...")
        try:
            tiling(
                slide_filepath=slide_filepath,
                magnification=args.magnification,
                patch_size=args.patch_size,
                scale_factor=args.scale_factor,
                tissue_thresh=args.tissue_thresh,
                method=args.method,
                overview_level=args.overview_level,
                coord_dir=coord_dir,
                overview_dir=overview_dir,
                mask_dir=mask_dir,
                patch_dir=patch_dir,
                filename=filename,
            )
            print(f"{filename} Done!\n")
        except:
            print(f"{filename} Error!\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slide_dir', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--exist_ok', action='store_true', default=False)
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--magnification', type=int, default=20, choices=[40, 20, 10, 5])
    parser.add_argument('--scale_factor', type=int, default=32,
                        help="scale wsi to down-sampled image for judging tissue percent of each patch.")
    parser.add_argument('--tissue_thresh', type=float, default=0.35,
                        help="the ratio of tissue region of a patch")
    parser.add_argument('--overview', action='store_true', default=False,
                        help="save the overview image after tiling if True")
    parser.add_argument('--save_mask', action='store_true', default=False)
    parser.add_argument('--save_patch', action='store_true', default=False,
                        help="save patch images if True, but we strongly recommend not saving patch images.")
    parser.add_argument('--wsi_format', type=str, default='.svs', choices=['.svs', '.tif'])
    parser.add_argument('--specify_filename', action='store_true', default=False)
    parser.add_argument('--filename_l', type=int, default=0)
    parser.add_argument('--filename_r', type=int, default=12)
    parser.add_argument('--method', type=str, default='rgb', choices=['otsu', 'adaptive', 'rgb'],
                        help="the filtering algorithm")
    parser.add_argument('--overview_level', type=int, default=-1,
                        help="the down-sample level of overview image")
    args = parser.parse_args()
    # print(f"args:\n{args}")
    run(args)


if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
