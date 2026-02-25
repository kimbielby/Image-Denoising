import numpy as np

from imports import *
from PIL import UnidentifiedImageError

def segmentation_process(
        gt_fp_list: list[str],
        noisy_fp_list: list[str],
        save_as_root: str | Path,
        patch_size: int
) -> None:
    """
    Crop ground truth and noisy images into fixed-size patches.

    Reads images from filepaths, pads them to be divisible by patch size,
    then crops them into non-overlapping patches. Saves cropped patches
    to save_as_root/cropped_gt and save_as_root/cropped_noisy
    directories. Skips completely black patches.

    Args:
        gt_fp_list: List of filepaths to ground truth images
        noisy_fp_list: List of filepaths to noisy images
        save_as_root: Root directory to save cropped images in
        patch_size: Size of square patches (e.g., 256 for 256x256 patches)

    Returns:
        None: Saves cropped images to disk

    Note:
        - Images are padded to nearest multiple of patch_size before cropping
        - Completely black patches (max pixel value = 0) are skipped
        - GT and noisy images must be in the same order to maintain pairing
        - Saves as PNG format
    """
    # Create dirs if necessary
    save_as_root = Path(save_as_root)
    noisy_root = save_as_root / "cropped_noisy"
    gt_root = save_as_root / "cropped_gt"
    noisy_root.mkdir(parents=True, exist_ok=True)
    gt_root.mkdir(parents=True, exist_ok=True)

    # Cycle through filepath lists
    for i in range(len(gt_fp_list)):
        # Read in images
        try:
            gt_img = np.array(Image.open(gt_fp_list[i]).convert("RGB"))
            noisy_img = np.array(Image.open(noisy_fp_list[i]).convert("RGB"))
        except FileNotFoundError as e:
            print(f"Image file not found for pair {i}: {e}")
            continue
        except UnidentifiedImageError as e:
            print(f"Invalid image format for pair {i}: {e}")
            continue
        except Exception as e:
            print(f"Failed to load image pair {i}: {e}")
            continue

        # Create image names
        gt_img_name = f"gt_{i}"
        noisy_img_name = f"noisy_{i}"

        gt_img_save_path = gt_root / gt_img_name
        noisy_img_save_path = noisy_root / noisy_img_name

        # Pad images
        gt_img_padded = pad_images(img=gt_img, patch_size=patch_size)
        noisy_img_padded = pad_images(img=noisy_img, patch_size=patch_size)

        # Crop and save images
        crop_and_save_images(
            img=gt_img_padded,
            patch_size=patch_size,
            img_name=gt_img_save_path
        )
        crop_and_save_images(
            img=noisy_img_padded,
            patch_size=patch_size,
            img_name=noisy_img_save_path
        )

def pad_images(
        img: np.ndarray,
        patch_size: int
) -> np.ndarray:
    """
    Pad image to be divisible by patch size.

    Adds zero-padding to the bottom and right edges to make image
    dimensions exact multiples of patch_size. Uses black pixels (0, 0, 0)
    for padding.

    Args:
        img: Input image (H, W, C)
        patch_size: Target patch size for divisibility

    Returns:
        Padded image with dimensions divisible by patch_size

    Note:
        - Only pads bottom and right edges (not top or left)
        - Uses constant black padding (value=0)
    """
    h, w = img.shape[:2]
    new_height = int(np.ceil(h / patch_size) * patch_size)
    new_width = int(np.ceil(w / patch_size) * patch_size)

    # Calculate padding
    pad_bottom = new_height - h
    pad_right = new_width - w

    # Add padding to bottom-right of image
    img_padded = np.pad(
        img,
        pad_width=((0, pad_bottom), (0, pad_right), (0, 0)),
        mode="constant",
        constant_values=0
    )

    return img_padded

def crop_and_save_images(
        img: np.ndarray,
        patch_size: int,
        img_name: str | Path,
) -> None:
    """
    Crop image into non-overlapping patches and save to disk.

    Divides image into square patches of size patch_size x patch_size,
    scanning from top-left to bottom-right. Skips completely black patches.
    Saves each patch as 'img_name_idx.png' where idx increments for
    each non-black patch.

    Args:
        img: Padded image to crop (H, W, C)
        patch_size: Size of square patches
        img_name: Base name for saved patches (without extension)

    Returns:
        None: Saves patches to disk

    Note:
        - Patches are scanned row-by-row (top to bottom, left to right)
        - Completely black patches (max pixel value = 0) are skipped
        - Each saved patch is numbered sequentially (excluding skipped patches)
        - Saves as PNG format
    """
    h, w = img.shape[:2]
    crop_name_num = 0

    # Iterate over vertical positions first (height)
    for y in range(0, h, patch_size):
        # Iterate over horizontal positions next (width)
        for x in range(0, w, patch_size):
            # Crop the image
            crop = img[y:y + patch_size, x:x + patch_size]

            # Skip completely black patches
            if np.max(crop) == 0:
                continue

            # Save patch
            crop_name = f"{img_name}_{crop_name_num}.png"
            Image.fromarray(crop).save(crop_name)
            crop_name_num += 1
