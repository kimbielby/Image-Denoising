from typing import Any
from imports import *

def tensor_to_uint8(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert single image tensor to uint8 numpy array.

    Args:
        tensor: Image tensor (C, H, W) in range [0, 1]

    Returns:
        RGB uint8 numpy array (H, W, C)
    """
    return np.clip(
        tensor.cpu().numpy().transpose(1, 2, 0) * 255, 0, 255
    ).astype(np.uint8)

def save_comparison(
        epoch: int,
        model: nn.Module,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        device: torch.device,
) -> None:
    """
    Save visual comparison of noisy, denoised and ground truth images.

    Creates a side-by-side [noisy | denoised | ground truth] comparison
    and saves to checkpoint_dir/comparisons/epoch_XXX.png.

    Args:
        epoch: Current epoch number
        model: Trained model
        val_loader: Validation dataloader
        checkpoint_dir: Directory to save comparison images
        device: Device to run inference on
    """
    model.eval()

    noisy, gt = next(iter(val_loader))
    noisy, gt = noisy.to(device), gt.to(device)

    with torch.no_grad():
        denoised = model(noisy)

    # Convert to uint8 numpy arrays
    comparison = np.hstack([
        tensor_to_uint8(noisy[0]),
        tensor_to_uint8(denoised[0]),
        tensor_to_uint8(gt[0])
    ])

    comparisons_dir = checkpoint_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    Image.fromarray(comparison).save(
        str(comparisons_dir / f"epoch_{epoch:03d}.png")
    )

def plot_training_metrics(
        history: dict[str, list],
        save_path: Optional[str | Path] = None
) -> None:
    """
    Plot training and validation metrics (Loss, PSNR, SSIM)

    Creates a 3-subplot figure showing loss, PSNR and SSIM over training
    epochs. Filters out None values from validation data to properly display
    sparse validation points.

    Args:
        history: Training history containing with keys epoch, train_loss,
            val_loss, train_psnr, val_psnr, train_ssim, val_ssim
        save_path: Path to save the figure to. If None, displays plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    epochs = history['epoch']

    # Filter validation data (remove None values)
    val_epochs_loss = [e for e, v in zip(epochs, history['val_loss']) if v is not None]
    val_loss = [v for v in history["val_loss"] if v is not None]

    val_epochs_psnr = [e for e, v in zip(epochs, history["val_psnr"]) if v is not None]
    val_psnr = [v for v in history["val_psnr"] if v is not None]

    val_epochs_ssim = [e for e, v in zip(epochs, history["val_ssim"]) if v is not None]
    val_ssim = [v for v in history["val_ssim"] if v is not None]

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    if val_loss:
        axes[0].plot(val_epochs_loss, val_loss, "o-", color="orange", label="Validation", linewidth=2, markersize=8)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training Loss", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # PSNR
    axes[1].plot(epochs, history["train_psnr"], "b-", label="Train", linewidth=2)
    if val_psnr:
        axes[1].plot(val_epochs_psnr, val_psnr, "o-", color="orange", label="Validation", linewidth=2, markersize=8)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("PSNR (dB)", fontsize=12)
    axes[1].set_title("PSNR Over Time", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # SSIM
    axes[2].plot(epochs, history["train_ssim"], "b-", label="Train", linewidth=2)
    if val_ssim:
        axes[2].plot(val_epochs_ssim, val_ssim, "o-", color="orange", label="Validation", linewidth=2, markersize=8)
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("SSIM (0-1)", fontsize=12)
    axes[2].set_title("SSIM over Time", fontsize=14, fontweight="bold")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved training metrics to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_test_results_distribution(
        results: dict[str, Any],
        save_path: Optional[str | Path] = None
) -> None:
    """
    Plot distribution of test results (PSNR and SSIM histograms)

    Creates histograms showing the distribution of PSNR and SSIM
    values across all test images, with mean lines indicated.

    Args:
        results: Test results from test() function containing per_image_list,
            avg_psnr, avg_ssim
        save_path: Path to save the figure to. If None, displays plot.
    """
    psnr_values = [img["psnr"] for img in results["per_image"]]
    ssim_values = [img["ssim"] for img in results["per_image"]]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # PSNR histogram
    axes[0].hist(psnr_values, bins=30, color="skyblue", edgecolor="black", alpha=0.7)
    axes[0].axvline(results["avg_psnr"], color="red", linestyle="dashed", linewidth=2, label=f"Mean: {results['avg_psnr']:.2f} dB")
    axes[0].set_xlabel("PSNR (dB)", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title("PSNR Distribution", fontsize=14, fontweight="bold")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis="y")

    # SSIM histogram
    axes[1].hist(ssim_values, bins=30, color="lightgreen", edgecolor="black", alpha=0.7)
    axes[1].axvline(results["avg_ssim"], color="red", linestyle="dashed", linewidth=2, label=f"Mean: {results['avg_ssim']:.4f}")
    axes[1].set_xlabel("SSIM", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title("SSIM Distribution", fontsize=14, fontweight="bold")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved test distribution to {save_path}")
    else:
        plt.show()

    plt.close()

def show_denoising_comparison(
        noisy_path: str | Path,
        denoised_path: str | Path,
        gt_path: Optional[str | Path] = None,
        save_path: Optional[str | Path] = None
) -> None:
    """
    Show side-by-side comparison of noisy, denoised and optionally ground
    truth images.

    Args:
        noisy_path: Path to noisy image
        denoised_path: Path to denoised image
        gt_path: Path to ground truth image. If None, only shows noisy and
            denoised.
        save_path: Path to save the figure. If None, displays plot.
    """
    noisy = np.array(Image.open(noisy_path))
    denoised = np.array(Image.open(denoised_path))

    # Determine number of subplots
    if gt_path:
        gt = np.array(Image.open(gt_path))
        images = [noisy, denoised, gt]
        titles = ["Noisy Input", "Denoised Output", "Ground Truth"]
        num_cols = 3
        figsize = (15, 5)
    else:
        images = [noisy, denoised]
        titles = ["Noisy Input", "Denoised Output"]
        num_cols = 2
        figsize = (10, 5)

    # Create figure and plot all images
    fig, axes = plt.subplots(1, num_cols, figsize=figsize)

    # Handle single subplot case (axes is not array if num_cols=1)
    if num_cols == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    plt.close()

def show_training_comparison_grid(
        checkpoint_dir: str | Path,
        max_epochs: int = 5
) -> None:
    """
    Show grid of denoising progress across training epochs.

    Displays how denoising quality improves over training by showing
    comparison images from different epochs.

    Args:
        checkpoint_dir: Path to directory containing comparison images
        max_epochs: Maximum number of epochs to show. Default: 5

    Note:
        Expects comparison images in checkpoint_dir/comparisons/ named epoch_XXX.png
    """
    comparisons_dir = Path(checkpoint_dir) / "comparisons"

    if not comparisons_dir.exists():
        print(f"No comparisons directory found at {comparisons_dir}")
        return

    # Get all comparison images
    comparison_files = sorted(comparisons_dir.glob("epoch_*.png"))

    if not comparison_files:
        print(f"No comparison images found at {comparisons_dir}")
        return

    # Select evenly spaced epochs
    num_comparisons = min(max_epochs, len(comparison_files))
    indices = np.linspace(0, len(comparison_files) - 1, num_comparisons, dtype=int)
    selected_files = [comparison_files[i] for i in indices]

    # Create grid
    fig, axes = plt.subplots(num_comparisons, 1, figsize=(15, 4 * num_comparisons))

    if num_comparisons == 1:
        axes = [axes]

    for ax, file_path in zip(axes, selected_files):
        img = Image.open(file_path)
        ax.imshow(img)

        # Extract epoch number from filename
        epoch_num = file_path.stem.split("_")[-1]
        ax.set_title(f"Epoch {int(epoch_num)}", fontsize=14, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()

def plot_metric_vs_metric(
        results: dict[str, Any],
        save_path: Optional[str | Path] = None
) -> None:
    """
    Plot PSNR vs SSIM scatter plot for test results.

    Shows relationship between PSNR and SSIM across test images.
    Helps identify if both metrics agree on image quality.

    Args:
        results: Test results from test() function
        save_path: Path to save the figure. If None, displays plot.
    """
    psnr_values = [img["psnr"] for img in results["per_image"]]
    ssim_values = [img["ssim"] for img in results["per_image"]]

    plt.figure(figsize=(8, 6))
    plt.scatter(psnr_values, ssim_values, alpha=0.6, s=50, c="blue", edgecolors="black")
    plt.xlabel("PSNR (dB)", fontsize=12)
    plt.ylabel("SSIM", fontsize=12)
    plt.title("PSNR vs. SSIM", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    # Add mean lines
    plt.axvline(results["avg_psnr"], color="red", linestyle="dashed",
                linewidth=2, label=f"Mean PSNR: {results['avg_psnr']:.2f} dB")
    plt.axhline(results["avg_ssim"], color="green", linestyle="dashed",
                linewidth=2, label=f"Mean SSIM: {results['avg_ssim']:.4f}")

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved PSNR vs SSIM scatter plot to {save_path}")
    else:
        plt.show()

    plt.close()

def show_best_worst_results(
        results: dict[str, Any],
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        save_path: Optional[str | Path] = None
) -> None:
    """
    Show best and worst denoising results from test set.

    Displays the top 3 best and 3 worst results based on PSNR.

    Args:
        results: Test results from test() function
        test_loader: Test data loader
        model: Trained model
        device: Device
        save_path: Path to save the figure. If None, displays plot.
    """
    def plot_comparison(
            ax,
            noisy: torch.Tensor,
            denoised: torch.Tensor,
            gt: torch.Tensor,
            title: str,
            psnr: float,
            ssim: float
    ) -> None:
        """Helper function for plotting comparison images."""
        # Convert tensors to numpy
        noisy_np = noisy.numpy().transpose(1, 2, 0)
        denoised_np = denoised.numpy().transpose(1, 2, 0)
        gt_np = gt.numpy().transpose(1, 2, 0)

        # Concatenate horizontally
        comparison = np.hstack([noisy_np, denoised_np, gt_np])
        comparison = np.clip(comparison, 0, 1)

        # Display
        ax.imshow(comparison)
        ax.set_title(f"{title}\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}",
                     fontsize=12, fontweight="bold")
        ax.axis("off")

    #  Sort by PSNR
    sorted_results = sorted(results["per_image"], key=lambda x: x["psnr"], reverse=True)

    # Get best 3 and worst 3 indices
    best_indices = {sorted_results[i]["image_idx"]: i for i in range(3)}
    worst_indices = {sorted_results[-(3-i)]["image_idx"]: len(sorted_results)-(3-i) for i in range(3)}

    needed_indices = set(best_indices.keys()) | set(worst_indices.keys())

    # Only load the images we need
    selected_images = {}
    current_idx = 0

    print(f"Loading only {len(needed_indices)} images (out of {results["num_images"]})...")

    for noisy, gt in test_loader:
        batch_size = noisy.shape[0]
        for i in range(batch_size):
            if current_idx in needed_indices:
                noisy_single = noisy[i:i+1].to(device)
                gt_single = gt[i:i+1]

                # Denoise single image
                with torch.no_grad():
                    denoised_single = model(noisy_single).cpu()

                selected_images[current_idx] = {
                    "noisy": noisy[i],
                    "denoised": denoised_single[0],
                    "gt": gt_single[0]
                }

                # Early exit once all images acquired
                if len(selected_images) == len(needed_indices):
                    break

            current_idx += 1

        if len(selected_images) == len(needed_indices):
            break

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot best 3
    for i, (img_idx, result_idx) in enumerate(best_indices.items()):
        if img_idx in selected_images:
            img_data = selected_images[img_idx]
            psnr = sorted_results[result_idx]["psnr"]
            ssim = sorted_results[result_idx]["ssim"]
            plot_comparison(
                ax=axes[0, i],
                noisy=img_data["noisy"],
                denoised=img_data["denoised"],
                gt=img_data["gt"],
                title=f"Best #{i+1}",
                psnr=psnr,
                ssim=ssim
            )

    # Plot worst 3
    for i, (img_idx, result_idx) in enumerate(worst_indices.items()):
        if img_idx in selected_images:
            img_data = selected_images[img_idx]
            psnr = sorted_results[result_idx]["psnr"]
            ssim = sorted_results[result_idx]["ssim"]
            plot_comparison(
                ax=axes[1, i],
                noisy=img_data["noisy"],
                denoised=img_data["denoised"],
                gt=img_data["gt"],
                title=f"Worst #{i+1}",
                psnr=psnr,
                ssim=ssim
            )

    fig.suptitle("Best and Worst Denoising Results\n[Noisy | Denoised "
                 "| Ground Truth]", fontsize=16, fontweight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved best/worst results to {save_path}")
    else:
        plt.show()

    plt.close()
