from utils import *
from imports import *
from typing import Any

@torch.no_grad()
def test(
        model: nn.Module,
        test_loader: DataLoader,
        device: str | torch.device,
        psnr_metric: Optional[PeakSignalNoiseRatio] = None,
        ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None,
        save_dir: Optional[str | Path] = None
) -> dict[str, Any]:
    """
    Test model on test dataset.

    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device (cuda or cpu)
        psnr_metric: PSNR metric object
        ssim_metric: SSIM metric object
        save_dir: Directory to save denoised images to. If None, images
            not saved

    Returns:
        dict: Test results containing:
            - avg_psnr: Average PSNR in dB
            - avg_ssim: Average SSIM
            - num_images: Number of test images
            - per_image: List of per-image results
    """
    # Put model in eval mode
    model.to(device).eval()

    # Initialise metrics
    total_psnr = 0.0
    total_ssim = 0.0
    num_images = 0

    # Store per-image results
    per_image_results = []

    # Calculate metrics flag
    calculate_metrics = (psnr_metric is not None and ssim_metric is not None)

    # Setup save directories if necessary
    if save_dir is not None:
        save_dir = Path(save_dir)
        denoised_dir = save_dir / "denoised"
        comparison_dir = save_dir / "comparisons"
        denoised_dir.mkdir(parents=True, exist_ok=True)
        comparison_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRunning inference on test set...")

    for batch_idx, (noisy, gt) in enumerate(tqdm(test_loader, desc="Testing")):
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        # Denoise
        denoised = model(noisy)

        # Calculate metrics for each image in batch
        batch_size = noisy.shape[0]
        for i in range(batch_size):
            # Get single images
            noisy_image = noisy[i:i+1]
            gt_image = gt[i:i+1]
            denoised_image = denoised[i:i+1]

            # Calculate PSNR and SSIM
            if calculate_metrics:
                psnr_value = get_batch_psnr(
                    output=denoised_image,
                    gt=gt_image,
                    psnr_metric=psnr_metric
                )
                ssim_value = get_batch_ssim(
                    output=denoised_image,
                    gt=gt_image,
                    ssim_metric=ssim_metric
                )
            else:
                psnr_value = 0.0
                ssim_value = 0.0

            total_psnr += psnr_value
            total_ssim += ssim_value
            num_images += 1

            # Store per-image results
            per_image_results.append({
                "image_idx": num_images - 1,
                "psnr": psnr_value,
                "ssim": ssim_value
            })

            # Save denoised image if requested
            if save_dir is not None:
                _save_test_image(
                    noisy=noisy_image,
                    denoised=denoised_image,
                    gt=gt_image,
                    denoised_dir=denoised_dir,
                    comparison_dir=comparison_dir,
                    idx=num_images - 1
                )

    # Calculate averages
    avg_psnr = total_psnr / num_images if calculate_metrics else 0.0
    avg_ssim = total_ssim / num_images if calculate_metrics else 0.0

    results = {
        "avg_psnr": avg_psnr,
        "avg_ssim": avg_ssim,
        "num_images": num_images,
        "per_image": per_image_results
    }

    return results

def _save_test_image(
        noisy: torch.Tensor,
        denoised: torch.Tensor,
        gt: torch.Tensor,
        denoised_dir: str | Path,
        comparison_dir: str | Path,
        idx: int
) -> None:
    """
    Save denoised and comparison (internal helper).

    Args:
        noisy: Noisy input (1, C, H, W)
        denoised: Denoised output (1, C, H, W)
        gt: GT image (1, C, H, W)
        denoised_dir: Directory to save denoised images to
        comparison_dir: Directory to save comparison images to
        idx: Image index
    """
    # Convert to numpy
    noisy_np = tensor_to_uint8(noisy[0])
    denoised_np = tensor_to_uint8(denoised[0])
    gt_np = tensor_to_uint8(gt[0])

    # Save individual denoised image
    denoised_path = denoised_dir / f"denoised_{idx:04d}.png"
    Image.fromarray(denoised_np).save(denoised_path)

    # Save comparison (noisy | denoised | ground truth)
    comparison = np.hstack([noisy_np, denoised_np, gt_np])
    comparison_path = comparison_dir / f"comparison_{idx:04d}.png"
    Image.fromarray(comparison).save(comparison_path)

def print_results(results: dict) -> None:
    """
    Print test results in formatted way.

    Args:
        results: Results from test() function
    """
    print(f"\n" + "="*70)
    print(f"TEST RESULTS")
    print(f"="*70)
    print(f"Number of test images: {results['num_images']}")
    print(f"\nAverage Metrics:")
    print(f"    PSNR: {results['avg_psnr']:.2f} dB")
    print(f"    SSIM: {results['avg_ssim']:.4f}")

    # Quality rating
    psnr = results['avg_psnr']
    if psnr >= 35:
        quality = "Excellent"
    elif psnr >= 30:
        quality = "Very Good"
    elif psnr >= 25:
        quality = "Good"
    else:
        quality = "Acceptable"

    print(f"\nQuality Rating: {quality}")

    # Statistics
    if results['per_image']:
        psnr_values = [img["psnr"] for img in results['per_image']]
        ssim_values = [img["ssim"] for img in results['per_image']]

        print(f"\nPSNR Statistics:")
        print(f"    Min:        {min(psnr_values):.2f} dB")
        print(f"    Max:        {max(psnr_values):.2f} dB")
        print(f"    Median:    {np.median(psnr_values):.2f} dB")
        print(f"    Std:        {np.std(psnr_values):.2f} dB")

        print(f"\nSSIM Statistics:")
        print(f"    Min:        {min(ssim_values):.4f}")
        print(f"    Max:        {max(ssim_values):.4f}")
        print(f"    Median:    {np.median(ssim_values):.4f}")
        print(f"    Std:        {np.std(ssim_values):.4f}")

    print(f"="*70)


