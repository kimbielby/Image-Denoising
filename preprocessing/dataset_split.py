from imports import *
from utils import *
from configs import Config
from typing import TypeAlias

# Type alias for clarity
SplitResult: TypeAlias = tuple[
    list[str], list[str],   # train (gt, noisy)
    list[str], list[str],   # val (gt, noisy)
    list[str], list[str]    # test (gt, noisy)
]

def prepare_and_split(
        config: Config
) -> SplitResult:
    """
    Prepare and split cropped images into train, validation and test sets.

    Loads all cropped ground truth and noisy image pairs, shuffles them
    randomly while maintaining pairing, then splits them into train/val/test
    sets according to the split ratios in the config.

    Args:
        config: Configuration object containing:
            - cropped_img_root: Root directory with cropped_gt and
                    cropped_noisy subdirectories
            - train_split: Training split ratio (e.g., 0.7 for first 70% of data)
            - valid_split: Cumulative validation split ratio (e.g., 0.9 = up to 90% of data)
                                This means that val set is from train_split to valid_split

    Returns:
        Tuple of six lists (gt_train, noisy_train, gt_val, noisy_val, gt_test, noisy_test):
            - gt_train_list: Ground truth training image paths
            - noisy_train_list: Noisy training image paths
            - gt_val_list: Ground truth validation image paths
            - noisy_val_list: Noisy validation image paths
            - gt_test_list: Ground truth test image paths
            - noisy_test_list: Noisy test image paths

    Note:
          - Images are shuffled together to maintain gt-noisy pairing
          - Split ratios are cumulative (e.g., train=0.7, val=0.9 means 70% train, 20% val, 10% test)
          - All lists are sorted before shuffling for reproducibility with same random seed
    """
    # Establish gt and noisy crop roots
    gt_crop_root = Path(config.paths.cropped_img_root) / "cropped_gt"
    print(f"gt_crop_root: {gt_crop_root}")
    noisy_crop_root = Path(config.paths.cropped_img_root) / "cropped_noisy"
    print(f"noisy_crop_root: {noisy_crop_root}")

    # Get Lists of image name for gt and noisy images
    gt_img_names = get_filepaths(dir_name=gt_crop_root)
    print(f"Number of GT images: {len(gt_img_names)}")
    noisy_img_names = get_filepaths(dir_name=noisy_crop_root)
    print(f"Number of Noisy images: {len(noisy_img_names)}")

    # Sort to make sure pairing remains same
    gt_img_names.sort()
    noisy_img_names.sort()

    # Shuffle lists together to keep pairing same
    gt_img_names, noisy_img_names = shuffle(gt_img_names, noisy_img_names)

    # Calculate indices that separate splits
    list_length = len(gt_img_names)
    print(f"Number of GT images: {list_length}")
    train_end = int(list_length * config.dataset.train_split)
    print(f"Train end index: {train_end}")
    val_end = int(list_length * config.dataset.valid_split)
    print(f"Val end index: {val_end}")

    # Split into Train-Val-Test
    gt_train_list = gt_img_names[:train_end]
    print(f"Number of gt train images: {len(gt_train_list)}")
    noisy_train_list = noisy_img_names[:train_end]
    print(f"Number of noisy train images: {len(noisy_train_list)}")

    gt_val_list = gt_img_names[train_end:val_end]
    print(f"Number of gt val images: {len(gt_val_list)}")
    noisy_val_list = noisy_img_names[train_end:val_end]
    print(f"Number of noisy val images: {len(noisy_val_list)}")

    gt_test_list = gt_img_names[val_end:]
    print(f"Number of gt test images: {len(gt_test_list)}")
    noisy_test_list = noisy_img_names[val_end:]
    print(f"Number of noisy test images: {len(noisy_test_list)}")

    return gt_train_list, noisy_train_list, gt_val_list, noisy_val_list, gt_test_list, noisy_test_list
