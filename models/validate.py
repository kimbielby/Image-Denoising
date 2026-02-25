from imports import *
from utils import *
from .losses import CombinedLoss
from typing import Callable, Optional
from configs import Config
from torch.cuda import device

def _get_loss_function(loss_fn_name: str) -> nn.Module:
    """
    Get loss function by name.

    Supports both torch.nn losses and custom losses.

    Args:
        loss_fn_name: Name of loss function

    Returns:
        Loss function instance

    Raises:
        ValueError: If loss function name is unknown
    """
    # Check torch.nn first
    if hasattr(nn, loss_fn_name):
        return getattr(nn, loss_fn_name)()

    # Check custom losses
    custom_losses = {
        "CombinedLoss": CombinedLoss
    }

    if loss_fn_name in custom_losses:
        return custom_losses[loss_fn_name]()

    raise ValueError(f"Unknown loss function: {loss_fn_name}")

def validate_function(
        val_loader: DataLoader,
        config: Config,
        device: str | device = "cuda",
        psnr_metric: Optional[PeakSignalNoiseRatio] = None,
        ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None
) -> Callable:
    """
    Create a validation function for model evaluation.

    Returns a closure function that can be called with a model to perform
    validation. The returned function evaluates the model on the validation
    dataset and computes loss, PSNR and SSIM metrics.

    Args:
        val_loader: Validation dataloader
        config: Configuration object containing loss function name
        device: Device to run validation on. Default: "cuda"
        psnr_metric: Pre-created PSNR metric object. If None, PSNR will
            not be calculated
        ssim_metric: Pre-created ssim metric object. If None, SSIM will
            not be calculated)

    Returns:
        Validation function that takes (model, device) and returns tuple of
            (avg_loss, avg_psnr, avg_ssim)

    Note:
        The returned function runs with torch.no_grad() for efficient
        inference. Model is automatically set to eval mode during validation.
    """

    @torch.no_grad()
    def _run(
            model: nn.Module,
            _device: Optional[str] = None
    ) -> tuple[float, float, float]:
        """
        Run validation on the model.

        Args:
            model: Model to be validated
            _device: Device override (not used, uses outer device)

        Returns:
            tuple of (avg_loss, avg_psnr, avg_ssim)
                - avg_loss (float): Average validation loss
                - avg_psnr (float): Average PSNR in dB (0.0 if metrics not provided)
                - avg_ssim (float): Average SSIM (0.0 if metrics not provided)
        """
        # Put model into eval mode
        model.eval()

        # Get Loss function
        loss_fn_name = config.loss.name
        loss_function = _get_loss_function(loss_fn_name)

        # Initialise values
        num_batches = 0
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        calculate_metrics = (psnr_metric is not None and ssim_metric is not None)

        for noisy, gt in tqdm(val_loader, desc='Validation'):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # Get eval results from noisy images
            output = model(noisy)

            num_batches += 1

            # Calculate Loss
            total_loss += loss_function(output, gt).item()

            #  Calculate metrics if available
            if calculate_metrics:
                total_psnr += get_batch_psnr(output, gt, psnr_metric)
                total_ssim += get_batch_ssim(output, gt, ssim_metric)

        # Return average values (var totals over num batches)
        avg_loss = total_loss / max(num_batches, 1)
        avg_psnr = total_psnr / max(num_batches, 1) if calculate_metrics else 0.0
        avg_ssim = total_ssim / max(num_batches, 1) if calculate_metrics else 0.0

        return avg_loss, avg_psnr, avg_ssim

    return _run