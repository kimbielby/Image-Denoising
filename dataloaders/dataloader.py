from typing import Callable

from configs import Config
from imports import *
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

class RealNoisyDataset(Dataset):
    """
    Dataset for real noisy/clean image pairs.

    Loads matching noisy and clean image pairs from separate directories.
    Images are loaded using PIL, normalised to [0,1] and returned as
    PyTorch tensors in CHW format.

    Supports optional color jitter augmentation for training.
    """
    def __init__(
            self,
            noisy_dir: str | Path,
            gt_dir: str | Path,
            gt_names: list[str],
            noisy_names: list[str],
            augment: bool = False,
            config=None
    ) -> None:
        """
        Initialise the dataset.

        Args:
            noisy_dir: Directory containing noisy images
            gt_dir: Directory containing ground truth images
            gt_names: List of ground truth image filenames
            noisy_names: List of noisy image filenames (must match gt_names order)
            augment: Whether to apply color jitter augmentation. Default: False
        """
        self.noisy_dir = Path(noisy_dir)
        self.gt_dir = Path(gt_dir)
        self.gt_names = gt_names
        self.noisy_names = noisy_names
        self.augment = augment

        # ColourJitter augmentation (currently disabled - causes training instability)
        # Geometric preprocessing augmentation (flips/rotations) proved more effective
        if augment and config:
            self.color_jitter = v2.ColorJitter(
                brightness=config.augmentation.brightness,
                contrast=config.augmentation.contrast,
                saturation=config.augmentation.saturation,
                hue=config.augmentation.hue
            )
        elif augment:
            self.color_jitter = v2.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05
            )

    def __len__(self) -> int:
        """
        Get dataset size.

        Returns:
           Number of image pairs in the dataset
        """
        return len(self.gt_names)

    def _load_image(self, path: str | Path) -> np.ndarray:
        """
        Load image and convert to normalised float array.

        Args:
            path: Path to image file

        Returns:
           Image as float32 array in HWC format, range [0,1]

        Raises:
            ValueError: If image cannot be loaded
        """
        img = Image.open(str(path)).convert('RGB')
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        return np.array(img).astype(np.float32) / 255.0

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Get a noisy/clean image pair by index.

        Args:
            idx: Index of the image pair

        Returns:
            tuple: (noisy, gt)
                - noisy: Noisy image tensor (C, H, W), range [0,1]
                - gt: Ground truth tensor (C, H, W), range [0,1]

        Raises:
            ValueError: If images have different shapes or cannot be loaded
        """
        # Get image index
        gt_path = self.gt_dir / self.gt_names[idx]
        noisy_path = self.noisy_dir / self.noisy_names[idx]

        # Load images
        gt = self._load_image(gt_path)
        noisy = self._load_image(noisy_path)

        # Verify same size
        if noisy.shape != gt.shape:
            raise ValueError(
                f"Shape mismatch: {self.noisy_names[idx]} and {self.gt_names[idx]}"
            )

        # Convert to tensors (HWC -> CHW)
        noisy = torch.from_numpy(noisy.transpose(2, 0, 1).copy())
        gt = torch.from_numpy(gt.transpose(2, 0, 1).copy())

        # Apply colour jitter augmentation (training only)
        if self.augment:
            noisy, gt = self._apply_color_jitter(noisy, gt)

        return noisy, gt

    def _apply_color_jitter(
            self,
            noisy: torch.Tensor,
            gt: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply identical color jitter to both noisy and gt images.

        Uses the same random parameters for both images so they stay
        correctly paired after augmentation.

        Args:
            noisy: Noisy image tensor (C, H, W)
            gt: Ground truth image tensor (C, H, W)

        Returns:
            Tuple of augmented (noisy, gt) tensors
        """
        # Stack into a single batch so same transform is applied to both
        stacked = torch.stack([noisy, gt])      # (2, C, H, W)
        # Apply identical transform to both
        stacked = self.color_jitter(stacked)
        # Clamp to valid range [0, 1]
        stacked = torch.clamp(stacked, 0.0, 1.0)
        return stacked[0], stacked[1]


def get_dataloader(
        config: Config,
        gt_img_names: list[str],
        noisy_img_names: list[str],
        device: torch.device,
        batch_size: int,
        collate_fn: Optional[Callable] = None,
        shuffle: bool = True,
        augment: bool = False
) -> DataLoader:
    """
    Create a dataloader for noisy/clean image pairs.

    Args:
        config: Configuration object containing paths
        gt_img_names: List of ground truth image filenames
        noisy_img_names : List of noisy image filenames (must match
            gt_names order)
        device: Device to use (cuda or cpu)
        batch_size: Number of samples per  batch
        collate_fn: Custom collate function. Default: Uses collate from utils.collate
        shuffle: If True, shuffle the dataset each epoch. Default: True
        augment: If True, apply color jitter augmentation. Default: False

    Returns:
            PyTorch DataLoader configured for efficient training
    """
    if collate_fn is None:
        from dataloaders.collate import collate as collate_fn

    gt_dir = Path(config.paths.cropped_img_root) / "cropped_gt"
    noisy_dir = Path(config.paths.cropped_img_root) / "cropped_noisy"

    dataset = RealNoisyDataset(
        noisy_dir=noisy_dir,
        gt_dir=gt_dir,
        gt_names=gt_img_names,
        noisy_names=noisy_img_names,
        augment=augment,
        config=config
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=config.train.num_workers,
        pin_memory=True if device.type == "cuda" else False,
        prefetch_factor=2 if config.train.num_workers > 0 else None
    )

    return loader

