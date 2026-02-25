from utils.visuals import *
from imports import *
from typing import Any

def save_all_visualisations(
        history: dict[str, list],
        results: dict[str, Any],
        test_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        checkpoint_dir: str | Path
) -> None:
    """
    Save all training and test visualisations to checkpoint_dir/figures/.

    Creates organised subdirectory structure and saves all plots in one call.

    Args:
        history: Training history from train()
        results: Test results from test()
        test_loader : Test data loader
        model: Trained model
        device: Device to use
        checkpoint_dir: Run directory (e.g., runs/20260201-091045)

    Returns:
        None: Saves all plots in checkpoint_dir/figures/
    """
    # Create figures directory
    figures_dir = Path(checkpoint_dir) / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save all plots
    print(f"Saving visualisations...")

    plot_training_metrics(
        history=history,
        save_path=figures_dir / "training_metrics.png"
    )

    plot_test_results_distribution(
        results=results,
        save_path=figures_dir / "test_results_distribution.png"
    )

    plot_metric_vs_metric(
        results=results,
        save_path=figures_dir / "psnr_vs_ssim.png"
    )

    show_best_worst_results(
        results=results,
        test_loader=test_loader,
        model=model,
        device=device,
        save_path=figures_dir / "best_worst_results.png"
    )

    print(f"All visualisations saved in {figures_dir}")

def save_example_comparisons(
        checkpoint_dir: str | Path,
        inference_dir: str | Path = "data/images/inference",
        denoised_dir: Optional[str | Path] = None,
        num_examples: int = 3
) -> None:
    """
    Save example denoising comparisons to checkpoint_dir/figures/examples/.

    Args:
        checkpoint_dir: Run directory
        inference_dir: Directory with inf_gt_*.png and inf_noisy_*.png.
            Default: "data/images/inference"
        denoised_dir: Directory with denoised images. If None, uses
            checkpoint_dir/inference_results/. Default: None
        num_examples: Number of examples to save. Default: 3

    Returns:
        None: Saves comparison images

    Note:
        - Expects inf_noisy_{i}.png and inf_gt_{i}.png in inference_dir
        - Expects denoised_{i}.png in denoised_dir (or checkpoint_dir/inference_results/)
    """

    checkpoint_dir = Path(checkpoint_dir)
    inference_dir = Path(inference_dir)

    # Default denoised directory is in the checkpoint folder
    if denoised_dir is None:
        denoised_dir = checkpoint_dir / "inference_results"
    else:
        denoised_dir = Path(denoised_dir)

    # Create examples directory
    examples_dir = checkpoint_dir / "figures" / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving {num_examples} example denoising comparisons...")

    for i in range(num_examples):
        noisy_path = inference_dir / f"inf_noisy_{i}.png"
        gt_path = inference_dir / f"inf_gt_{i}.png"
        denoised_path = denoised_dir / f"denoised_{i}.png"

        # Check if files exist
        if not noisy_path.exists():
            print(f"Warning: {noisy_path} not found, skipping example {i}")
            continue
        if not gt_path.exists():
            print(f"Warning: {gt_path} not found, skipping example {i}")
            continue
        if not denoised_path.exists():
            print(f"Warning: {denoised_path} not found, skipping example {i}")
            continue

        show_denoising_comparison(
            noisy_path=noisy_path,
            denoised_path=denoised_path,
            gt_path=gt_path,
            save_path=examples_dir / f"example_{i}.png"
        )

    print(f"Saved {num_examples} examples to {examples_dir}")

