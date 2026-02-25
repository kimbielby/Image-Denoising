import torch

def collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to stack image batches.

    Takes a list of (image, gt) tuples from the dataset and stacks
    them into batched tensors. Used as collate_fn parameter in DataLoader.

    Args:
        batch: List of (noisy, gt) pairs where:
            - noisy: Single image tensor (C, H, W)
            - gt: Single target tensor (C, H, W)

    Returns:
        tuple: (batched_noisy, batched_gt)
            - batched_images: Stacked images (B, C, H, W)
            - batched_targets: Stacked targets (B, C, H, W)

    Note:
        - All images in batch must have same dimensions
        - Used for pairing noisy and ground truth images in denoising
    """
    imgs, targs = zip(*batch)
    return torch.stack(imgs), torch.stack(targs)

