from imports import *
import random

def _load_image(path: str | Path) -> np.ndarray:
    """
    Load an image and return it as a numpy array.
    """
    return np.array(Image.open(str(path)).convert('RGB'))

def _save_image(img: np.ndarray, path: str | Path) -> None:
    """
    Save numpy array as image
    """
    Image.fromarray(img).save(str(path))

def _apply_geometric_augmentation(
        gt_img: np.ndarray,
        noisy_img: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply identical random geometric augmentation to a noisy/gt pair.

    Applies random horizontal flip, vertical flip and 90 degree rotation.
    Both images receive the same transforms to preserve their pairing.

    Args:
        gt_img: Ground truth image as numpy array (H, W, C)
        noisy_img: Noisy image as numpy array (H, W, C)

    Returns:
        Tuple of augmented (gt_img, noisy_img)
    """
    # Random horizontal flip
    if np.random.rand() > 0.5:
        gt_img = np.fliplr(gt_img)
        noisy_img = np.fliplr(noisy_img)

    # Random vertical flip
    if np.random.rand() > 0.5:
        gt_img = np.flipud(gt_img)
        noisy_img = np.flipud(noisy_img)

    # Random 90 degree rotation
    k = np.random.randint(low=0, high=4)
    if k > 0:
        gt_img = np.rot90(gt_img, k)
        noisy_img = np.rot90(noisy_img, k)

    return gt_img, noisy_img

def _get_image_brightness(path: str | Path) -> float:
    """
    Get mean brightness of an image.

    Args:
        path: Path to image file

    Returns:
        Mean pixel brightness (0-255)
    """
    img = np.array(Image.open(str(path)).convert('RGB'))
    return img.mean()

def _create_and_save_augmented_pair(
        gt_name: str,
        noisy_name: str,
        gt_root: Path,
        noisy_root: Path,
        suffix: str
) -> tuple[str, str]:
    """
    Create and save an augmented image pair.

    Loads a ground truth and noisy image pair, applies random geometric
    augmentation, saves the augmented images with a suffix and returns
    the new filenames.

    Args:
        gt_name: Original gt filename
        noisy_name: Original noisy filename
        gt_root: Directory containing gt images
        noisy_root: Directory containing noisy images
        suffix: Suffix to append to filenames (e.g., '_aug', '_bright_aug0')

    Returns:
        Tuple of (new_gt_name, new_noisy_name)
    """
    # Load images
    gt_img = _load_image(gt_root / gt_name)
    noisy_img = _load_image(noisy_root / noisy_name)

    # Apply augmentation
    gt_img, noisy_img = _apply_geometric_augmentation(gt_img, noisy_img)

    # Create new filenames
    stem_gt = Path(gt_name).stem
    stem_noisy = Path(noisy_name).stem
    ext_gt = Path(gt_name).suffix
    ext_noisy = Path(noisy_name).suffix

    new_gt_name = f"{stem_gt}{suffix}{ext_gt}"
    new_noisy_name = f"{stem_noisy}{suffix}{ext_noisy}"

    # Save augmented images
    _save_image(gt_img, gt_root / new_gt_name)
    _save_image(noisy_img, noisy_root / new_noisy_name)

    return new_gt_name, new_noisy_name

def augment_images(
        og_gt_img_names: list[str],
        og_noisy_img_names: list[str],
        crop_img_root: str | Path,
        bright_threshold: float = 200.0,
        bright_copies: int = 5,
        random_n: int = 50,
) -> tuple[list[str], list[str]]:
    """
    Augment training images with geometric transforms.

    Uses two strategies:
    1. Targeted: Augments bright images multiple times to fix class imbalance
    2. Random: Augments a random sample of remaining images for variety

    Args:
        og_gt_img_names: List of ground truth image filenames
        og_noisy_img_names: List of noisy image filenames
        crop_img_root: Root directory containing cropped_gt and cropped_noisy
        bright_threshold: Mean brightness above which images are
            considered bright. Default: 200.0
        bright_copies: Number of augmented copies to create per bright
            image. Default: 5
        random_n: Number of random non-bright images to augment. Default: 50

    Returns:
        Tuple of (gt_img_names, noisy_img_names) with augmented names appended
    """
    # Copy image name lists
    gt_img_names = og_gt_img_names.copy()
    noisy_img_names = og_noisy_img_names.copy()

    # Create crop paths
    gt_root = Path(crop_img_root) / "cropped_gt"
    noisy_root = Path(crop_img_root) / "cropped_noisy"

    print(f"gt_root: {gt_root}")
    print(f"noisy_root: {noisy_root}\n")

    """ Strategy 1: Targeted augmentation of bright images """
    print(f"Scanning for bright images...")

    bright_indices = [
        i for i, name in enumerate(gt_img_names)
        if _get_image_brightness(gt_root / name) > bright_threshold
    ]

    print(f"Found {len(bright_indices)} bright images "
          f"(>{bright_threshold} brightness)")
    print(f"Creating {bright_copies} augmented copies each...\n")

    for i in bright_indices:
        gt_name = gt_img_names[i]
        noisy_name = noisy_img_names[i]

        for copy_num in range(bright_copies):
            # Create and save augmented pair
            new_gt_name, new_noisy_name = _create_and_save_augmented_pair(
                gt_name=gt_name,
                noisy_name=noisy_name,
                gt_root=gt_root,
                noisy_root=noisy_root,
                suffix=f"_bright_aug{copy_num}"
            )

            # Add to lists
            gt_img_names.append(new_gt_name)
            noisy_img_names.append(new_noisy_name)

    print(f"Added {len(bright_indices) * bright_copies} bright augmented images")

    """ Strategy 2: Random augmentation of remaining images """

    # Randomly determine which indices get augmented
    non_bright_indices = [
        i for i in range(len(og_gt_img_names))
        if i not in bright_indices
    ]

    # Sample random subset
    random_n = min(random_n, len(non_bright_indices))
    random_indices = random.sample(non_bright_indices, random_n)

    print(f"\nAugmenting {random_n} random non-bright images...\n")

    for i in random_indices:
        gt_name = gt_img_names[i]
        noisy_name = noisy_img_names[i]

        # Create and save augmented pair
        new_gt_name, new_noisy_name = _create_and_save_augmented_pair(
            gt_name=gt_name,
            noisy_name=noisy_name,
            gt_root=gt_root,
            noisy_root=noisy_root,
            suffix="_aug"
        )

        # Add augmented names to  lists
        gt_img_names.append(new_gt_name)
        noisy_img_names.append(new_noisy_name)

    print(f"Added {random_n} random augmented images")

    """ Summary """
    og_count = len(og_gt_img_names)
    total_count = len(gt_img_names)
    added = total_count - og_count

    print(f"\n{"="*50}")
    print(f"Augmentation complete: ")
    print(f"    Original images: {og_count}")
    print(f"    Added:               {added}")
    print(f"    Total:                 {total_count}")
    print("="*50)

    return gt_img_names, noisy_img_names

