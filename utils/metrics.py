from imports import *

def get_batch_psnr(
        output: torch.Tensor,
        gt: torch.Tensor,
        psnr_metric: PeakSignalNoiseRatio
) -> float:
    """
    Calculate PSNR for a batch using pre-created metric object.

    Args:
        output: Model output predictions (B, C, H, W)
        gt: Ground truth images (B, C, H, W)
        psnr_metric: Pre-created PSNR metric instance

    Returns:
        float: Average PSNR value for the batch in dB
    """
    psnr_metric.reset()
    psnr_metric.update(preds=output, target=gt)
    return psnr_metric.compute().item()

def get_batch_ssim(
        output: torch.Tensor,
        gt: torch.Tensor,
        ssim_metric: StructuralSimilarityIndexMeasure
) -> float:
    """
    Calculate SSIM for a batch using pre-created metric object.

    Args:
        output: Model output predictions (B, C, H, W)
        gt: Ground truth images (B, C, H, W)
        ssim_metric: Pre-created SSIM metric instance

    Returns:
        Average SSIM value for the batch (range: -1 to 1, typically 0 to 1)
    """
    return ssim_metric(preds=output, target=gt).item()

def create_metrics_objects(
        device: str | torch.device
) -> tuple[PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure]:
    """
    Create PSNR and SSIM metric objects for reuse during training.

    Args:
        device: Device to create metrics objects on (cuda or cpu)

    Returns:
        tuple: (psnr_metric, ssim_metric)
            - psnr_metric: PSNR metric object
            - ssim_metric: SSIM metric object
    """
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    return psnr_metric, ssim_metric
