import os
import shutil
from glob import glob
from typing import List, Tuple
import numpy as np

from config import get_dataset_config, DatasetConfig


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _copy_single_image(src: str, dst_dir: str):
    _ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy(src, dst)


def _prepare_from_landmark(src_cfg: DatasetConfig, out_base: str, num_places: int = None, images_per_place: int = None) -> Tuple[int, int]:
    place_dirs = sorted([d for d in glob(os.path.join(src_cfg.path, 'p*')) if os.path.isdir(d)],
                        key=lambda p: int(os.path.basename(p)[1:]))

    if num_places is None:
        num_places = len(place_dirs)

    copied_places = 0
    copied_images_per_place = 0

    for p_idx, place_dir in enumerate(place_dirs[:num_places]):
        i_dirs = sorted([d for d in glob(os.path.join(place_dir, 'i*')) if os.path.isdir(d)],
                        key=lambda i: int(os.path.basename(i)[1:]))
        if images_per_place is None:
            limit_i = len(i_dirs)
        else:
            limit_i = min(images_per_place, len(i_dirs))

        dst_place_dir = os.path.join(out_base, f'p{p_idx}')
        for j, i_dir in enumerate(i_dirs[:limit_i]):
            dst_i_dir = os.path.join(dst_place_dir, f'i{j}')
            # Copy exactly one image file per i*
            imgs = sorted(glob(os.path.join(i_dir, '*.jpg'))) or sorted(glob(os.path.join(i_dir, '*.png')))
            if not imgs:
                continue
            _copy_single_image(imgs[0], dst_i_dir)

        copied_places += 1
        copied_images_per_place = max(copied_images_per_place, limit_i)

    return copied_places, copied_images_per_place


def _prepare_from_sequential(src_cfg: DatasetConfig, out_base: str, num_places: int = None) -> Tuple[int, int]:
    # Supported sequential datasets in this codebase: StLuciaSmall, SFU
    from datasets.load_dataset import StLuciaDataset, SFUDataset

    if src_cfg.name == 'StLuciaSmall':
        loader = StLuciaDataset(src_cfg.path)
        db_dir = os.path.join(src_cfg.path, loader.fns_db_path)
        q_dir = os.path.join(src_cfg.path, loader.fns_q_path)
        # Load GT for precise pairing
        gt = np.load(os.path.join(src_cfg.path, 'GT.npz'))
        GThard = gt['GThard'].astype(bool)
        fns_db = sorted(glob(os.path.join(db_dir, '*.jpg')))
        fns_q = sorted(glob(os.path.join(q_dir, '*.jpg')))
        pairs: List[Tuple[int, int]] = []
        for i in range(GThard.shape[0]):
            js = np.where(GThard[i])[0]
            if js.size == 0:
                continue
            pairs.append((i, int(js[0])))
    elif src_cfg.name == 'SFU':
        loader = SFUDataset(src_cfg.path)
        db_dir = os.path.join(src_cfg.path, loader.fns_db_path)
        q_dir = os.path.join(src_cfg.path, loader.fns_q_path)
        gt = np.load(os.path.join(src_cfg.path, 'GT.npz'))
        GThard = gt['GThard'].astype(bool)
        fns_db = sorted(glob(os.path.join(db_dir, '*.jpg')))
        fns_q = sorted(glob(os.path.join(q_dir, '*.jpg')))
        pairs: List[Tuple[int, int]] = []
        for i in range(GThard.shape[0]):
            js = np.where(GThard[i])[0]
            if js.size == 0:
                continue
            pairs.append((i, int(js[0])))
    elif src_cfg.name == 'GardensPoint':
        # Fallback: pair by filename index between day_right (DB) and night_right (Q)
        db_dir = os.path.join(src_cfg.path, 'day_right')
        q_dir = os.path.join(src_cfg.path, 'night_right')
        fns_db = sorted(glob(os.path.join(db_dir, '*.jpg')))
        fns_q = sorted(glob(os.path.join(q_dir, '*.jpg')))
        limit = min(len(fns_db), len(fns_q))
        pairs = [(i, i) for i in range(limit)]
    else:
        raise NotImplementedError(f"No sequential mini-prep implemented for {src_cfg.name}")

    if num_places is not None:
        pairs = pairs[:num_places]

    # Build p*/i0 (db), p*/i1 (q)
    for p_idx, (i_db, i_q) in enumerate(pairs):
        dst_p_dir = os.path.join(out_base, f'p{p_idx}')
        _copy_single_image(fns_db[i_db], os.path.join(dst_p_dir, 'i0'))
        _copy_single_image(fns_q[i_q], os.path.join(dst_p_dir, 'i1'))

    return len(pairs), 2


def _prepare_conditions_group_and_skip(src_cfg: DatasetConfig, out_base: str,
                                       step_size: int = 8, group_size: int = 2) -> Tuple[int, int]:
    # Use explicit conditions from config if present; else infer subfolders
    if src_cfg.conditions:
        conditions = src_cfg.conditions
    else:
        conditions = [d for d in sorted(os.listdir(src_cfg.path))
                      if os.path.isdir(os.path.join(src_cfg.path, d))]
    if not conditions:
        raise ValueError(f"No condition subfolders found in {src_cfg.path}")

    # Collect and sort per-condition file lists
    cond_files: List[List[str]] = []
    for cond in conditions:
        files = sorted(glob(os.path.join(src_cfg.path, cond, '*.jpg'))) + \
                sorted(glob(os.path.join(src_cfg.path, cond, '*.jpeg'))) + \
                sorted(glob(os.path.join(src_cfg.path, cond, '*.png')))
        files = sorted(list(dict.fromkeys(files)))
        cond_files.append(files)
    # Work by index to avoid filename dependency; re-export with unified names
    min_len = min(len(files) for files in cond_files)

    # Clean output and recreate condition folders
    if os.path.exists(out_base):
        shutil.rmtree(out_base)
    for cond in conditions:
        os.makedirs(os.path.join(out_base, cond), exist_ok=True)

    place_count = 0
    idx = 0
    while idx + group_size - 1 < min_len:
        # Build a place with group_size consecutive indices across all conditions
        for g in range(group_size):
            index = idx + g
            # Export one image per condition with unified names: PlacePPPP_CondCC_GG.jpg
            for c_idx, cond in enumerate(conditions):
                src_img = cond_files[c_idx][index]
                ext = os.path.splitext(src_img)[1].lower() or '.jpg'
                dst_name = f"Place{place_count:04d}_Cond{c_idx:02d}_G{g:02d}{ext}"
                dst_path = os.path.join(out_base, cond, dst_name)
                shutil.copy(src_img, dst_path)
        place_count += 1
        idx += step_size

    # images_per_place equals number of exported images per place per condition summed across conditions
    images_per_place = group_size * len(conditions)
    return place_count, images_per_place


def prepare_mini_dataset(dataset_key: str,
                         output_name: str = None,
                         num_places: int = None,
                         images_per_place: int = None,
                         step_size: int = 8,
                         group_size: int = 2) -> Tuple[str, str, int, int]:
    """
    Create a mini dataset with landmark structure (p*/i*/image.jpg) from a source dataset.

    Returns (mini_key, output_path, num_places_out, images_per_place_out)
    """
    cfg = get_dataset_config(dataset_key)
    mini_key = (dataset_key + '_mini') if output_name is None else output_name
    # Use the key as folder name for predictability
    output_path = os.path.join('images', mini_key)

    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    if cfg.format == 'landmark':
        # Not targeted; preserve as-is unless user wants to rebuild via conditions
        n_p, n_i = _prepare_from_landmark(cfg, output_path, num_places, images_per_place)
    elif cfg.format in ('sequential', 'landmark_grouped'):
        # Unified conditions-based builder preserving folder structure
        n_p, n_i = _prepare_conditions_group_and_skip(cfg, output_path, step_size, group_size)
    else:
        raise NotImplementedError(f"Unsupported dataset format: {cfg.format}")

    return mini_key, output_path, n_p, n_i


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Prepare a mini dataset with landmark structure from a source dataset')
    parser.add_argument('--dataset', required=True, help='Source dataset key from config.DATASETS')
    parser.add_argument('--output-name', default=None, help='Mini dataset key/name (default: <dataset>_mini)')
    parser.add_argument('--num-places', type=int, default=None, help='Number of places to include (default: all)')
    parser.add_argument('--images-per-place', type=int, default=None, help='Images per place (landmark only, default: all)')
    parser.add_argument('--step-size', type=int, default=10, help='Grouping step size (grouped only)')
    parser.add_argument('--group-size', type=int, default=1, help='Images per step (grouped only)')

    args = parser.parse_args()
    mini_key, out_path, n_p, n_i = prepare_mini_dataset(
        dataset_key=args.dataset,
        output_name=args.output_name,
        num_places=args.num_places,
        images_per_place=args.images_per_place,
        step_size=args.step_size,
        group_size=args.group_size,
    )
    print(f"Prepared mini dataset '{mini_key}' at {out_path} with {n_p} places and {n_i} images/place")


