from imports import *

class CombinedLoss(nn.Module):
    """
    Combined MSE and SSIM loss for image denoising.
    """
    def __init__(self, alpha: float = 0.8) -> None:
        """
        Initialise combined loss.

        Args:
            alpha: Weight for MSE loss. SSIM weight = (1 - alpha).
                Higher alpha = more pixel accuracy, lower alpha = more
                perceptual quality. Default: 0.8
        """
        super().__init__()

        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.ssim = SSIM(data_range=1.0, channel=3)

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined MSE + SSIM loss.

        Args:
            predicted: Model output tensor (B, C, H, W)
            target: Ground truth tensor (B, C, H, W)

        Returns:
            Combined loss value
        """
        mse_loss = self.mse(predicted, target)
        ssim_loss = 1 - self.ssim(predicted, target)

        return self.alpha * mse_loss + (1 - self.alpha) * ssim_loss
