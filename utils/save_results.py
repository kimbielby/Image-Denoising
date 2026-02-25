from typing import Any
from imports import *

def save_training_history(
        history: dict[str, Any],
        save_path: str | Path
) -> None:
    """
    Save training history to JSON file.

    Args:
        history: Training history dictionary containing epoch, train_loss, val_loss,
            train_psnr, val_psnr, train_ssim, val_ssim
        save_path: Path to save file (e.g., "checkpoint_dir/history.json")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert any None values to null for JSON compatibility
    clean_history = {}
    for key, value in history.items():
        if isinstance(value, list):
            clean_history[key] = [v if v is not None else None for v in value]
        else:
            clean_history[key] = value

    with open(save_path, 'w') as f:
        json.dump(clean_history, f, indent=2)

    print(f"Saved training history to {save_path}")

def load_training_history(load_path: str | Path) -> dict[str, Any]:
    """
    Load training history from JSON file.

    Args:
        load_path: Path to history file

    Returns:
        Training history dictionary
    """
    with open(load_path, 'r') as f:
        history = json.load(f)

    print(f"Loaded training history from {load_path}")

    return history

def save_test_results(
        results: dict[str, Any],
        save_path: str | Path
) -> None:
    """
    Save test results to JSON file.

    Args:
        results: Dictionary of test results from test() function containing
            avg_psnr, avg_ssim, num_images, per_image
        save_path: Path to save file (e.g., "checkpoint_dir/test_results.json")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved test results to {save_path}")

def load_test_results(load_path: str | Path) -> dict[str, Any]:
    """
    Load test results from JSON file.

    Args:
        load_path: Path to test results file

    Returns:
        Test results dictionary
    """
    with open(load_path, 'r') as f:
        results = json.load(f)

    print(f"Loaded test results from {load_path}")

    return results

def save_summary(
        history: dict[str, Any],
        results: dict[str, Any],
        save_path: str | Path
) -> None:
    """
    Save combined training and test summary to text file.

    Creates a human-readable summary of the training and testing results.

    Args:
        history: Training history
        results: Test results
        save_path: Path to save summary file (e.g., "checkpoint_dir/summary.txt")
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, 'w') as f:
        f.write("="*70 + "\n ")
        f.write("TRAINING AND TEST SUMMARY\n")
        f.write("="*70 + "\n\n")

        # Training info
        f.write("TRAINING RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Total epochs: {len(history["epoch"])}\n")

        if history["train_loss"]:
            final_train_loss = [x for x in history["train_loss"] if x is not None][-1]
            f.write(f"Final train loss: {final_train_loss:.4f}\n")

        if history["val_loss"]:
            final_val_loss = [x for x in history["val_loss"] if x is not None][-1]
            best_val_loss = min([x for x in history["val_loss"] if x is not None])
            f.write(f"Final validation loss: {final_val_loss:.4f}\n")
            f.write(f"Best validation loss: {best_val_loss:.4f}\n")

        if history["val_psnr"]:
            final_val_psnr = [x for x in history["val_psnr"] if x is not None][-1]
            best_val_psnr = max([x for x in history["val_psnr"] if x is not None])
            f.write(f"Final validation PSNR: {final_val_psnr:.2f} dB\n")
            f.write(f"Best validation PSNR: {best_val_psnr:.2f} dB\n")

        if history["val_ssim"]:
            final_val_ssim = [x for x in history["val_ssim"] if x is not None][-1]
            best_val_ssim = max([x for x in history["val_ssim"] if x is not None])
            f.write(f"Final validation SSIM: {final_val_ssim:.4f}\n")
            f.write(f"Best validation SSIM: {best_val_ssim:.4f}\n")

        f.write("\n")

        # Test info
        f.write("TEST RESULTS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of test images: {results['num_images']}\n")
        f.write(f"Average PSNR: {results['avg_psnr']:.2f} dB\n")
        f.write(f"Average SSIM: {results['avg_ssim']:.4f}\n")

        # PSNR statistics
        psnr_values = [img["psnr"] for img in results["per_image"]]
        f.write(f"\nPSNR Statistics:\n")
        f.write(f"  Min:        {min(psnr_values):.2f} dB\n")
        f.write(f"  Max:        {max(psnr_values):.2f} dB\n")
        f.write(f"  Median:   {np.median(psnr_values):.2f} dB\n")
        f.write(f"  Std:        {np.std(psnr_values):.2f} dB\n")

        # SSIM statistics
        ssim_values = [img["ssim"] for img in results["per_image"]]
        f.write(f"\nSSIM Statistics:\n")
        f.write(f"  Min:        {min(ssim_values):.4f}\n")
        f.write(f"  Max:        {max(ssim_values):.4f}\n")
        f.write(f"  Median:   {np.median(ssim_values):.4f}\n")
        f.write(f"  Std:        {np.std(ssim_values):.4f}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"Saved summary to {save_path}")





