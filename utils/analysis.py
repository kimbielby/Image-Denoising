from typing import Dict, Optional, Tuple

def get_best_epoch(history: Dict) -> Optional[Tuple[int, float, float]]:
    """
    Find the epoch with the best validation PSNR.

    Filters out None values from validation history and returns the epoch
    with the highest PSNR, along with the corresponding PSNR and SSIM
    values.

    Args:
        history: Training history dictionary with keys 'val_psnr', 'val_ssim'
            and 'epoch'

    Returns:
         Tuple of (epoch, psnr, ssim) for the best epoch, or None if there is
            no validation data
    """
    val_psnr = [x for x in history["val_psnr"] if x is not None]

    if not val_psnr:
        return None

    best_psnr = max(val_psnr)
    best_epoch = history["val_psnr"].index(best_psnr)
    best_ssim = history["val_ssim"][best_epoch]

    return best_epoch, best_psnr, best_ssim

def print_best_results(history: Dict, verbose: bool = True) -> None:
    """
    Print formatted summary of best validation results.

    Args:
        history: Training history dictionary
        verbose: If True, shows all validation points. If False, shows only
            the best
    """
    result = get_best_epoch(history)

    if result is None:
        print("No validation data available")
        return

    best_epoch, best_psnr, best_ssim = result
    total_epochs = len(history["epoch"])

    print("=" * 60)
    print(f"BEST VALIDATION RESULTS")
    print("=" * 60)
    print(f"Best epoch: {best_epoch} / {total_epochs}")
    print(f"Best val PSNR: {best_psnr:.2f} dB")
    print(f"Best val SSIM: {best_ssim:.4f}")
    print("=" * 60)

    if verbose:
        print("\nValidation history:")
        for epoch, psnr, ssim in zip(
            history['epoch'],
            history["val_psnr"],
            history["val_ssim"]
        ):
            if psnr is not None:
                print(f"    Epoch {epoch}: PSNR {psnr:.2f} dB, SSIM {ssim:.4f}")

def get_final_metrics(history: Dict) -> Dict[str, float]:
    """
    Get final training and validation metrics.

    Returns the metrics from the last epoch, filtering out None values.

    Args:
        history: Training history dictionary

    Returns:
        Dictionary with final metrics: train_loss, train_psnr, train_ssim,
            val_loss, val_psnr, val_ssim (values are None if not available)
    """
    metrics = {
        "train_loss": history["train_loss"][-1] if history["train_loss"] else None,
        "train_psnr": history["train_psnr"][-1] if history["train_psnr"] else None,
        "train_ssim": history["train_ssim"][-1] if history["train_ssim"] else None,
        "val_loss": None,
        "val_psnr": None,
        "val_ssim": None
    }

    # Get last non-None validation values
    val_psnr = [x for x in history["val_psnr"] if x is not None]
    val_ssim = [x for x in history["val_ssim"] if x is not None]
    val_loss = [x for x in history["val_loss"] if x is not None]

    if val_psnr:
        metrics["val_psnr"] = val_psnr[-1]
    if val_ssim:
        metrics["val_ssim"] = val_ssim[-1]
    if val_loss:
        metrics["val_loss"] = val_loss[-1]

    return metrics

def print_training_summary(history: Dict) -> None:
    """
    Print comprehensive training summary.

    Shows  both best validation results and final metrics for easy comparison.

    Args:
        history: Training history dictionary
    """
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)

    # Best results
    result = get_best_epoch(history)
    if result is not None:
        best_epoch, best_psnr, best_ssim = result
        print(f"\n Best Validation (Epoch {best_epoch}):)")
        print(f"    PSNR: {best_psnr:.2f} dB")
        print(f"    SSIM: {best_ssim:.4f}")

    # Final metrics
    final = get_final_metrics(history)
    print(f"\n Final Training:")
    if final["train_psnr"] is not None:
        print(f"    PSNR: {final['train_psnr']:.2f} dB")
        print(f"    SSIM: {final['train_ssim']:.4f}")
        print(f"    Loss: {final['train_loss']:.4f}")

    # Train/val gap if available
    if result is not None and final["train_psnr"] is not None:
        gap = final["train_psnr"] - best_psnr
        print(f"\n  Train/Val Gap: {gap:.2f} dB")
        print(f"    PSNR difference: {gap:.2f} dB")
        if gap > 2.0:
            print(f"    Possible overfitting (gap > 2 dB)")
        else:
            print(f"    Good generalisation (gap ≤ 2 dB)")

    print("=" * 60)

def compare_models(
        history1: Dict,
        history2: Dict,
        name1: str = "Model 1",
        name2: str = "Model 2"
) -> None:
    """
    Compare two training runs side by side.

    Args:
        history1: First training history
        history2: Second training history
        name1: Name of the first model
        name2: Name of the second model
    """
    result1 = get_best_epoch(history1)
    result2 = get_best_epoch(history2)

    if result1 is None or result2 is None:
        print(" Cannot compare - missing validation data")
        return

    _, psnr1, ssim1 = result1
    _, psnr2, ssim2 = result2

    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric': <20} {name1:<15} {'Difference': <15}")
    print("-" * 60)
    print(f"{'PSNR (dB)':<20} {psnr1:<15.2f} {psnr2:<15.2f} {psnr2-psnr1:+.2f}")
    print(f"{'SSIM':<20} {ssim1:<15.4f} {ssim2:<15.4f} {ssim2-ssim1:+.4f}")
    print("=" * 60)

    if psnr2 > psnr1:
        print(f"    {name2} is better by {psnr2-psnr1:.2f} dB PSNR")
    elif psnr1 > psnr2:
        print(f"    {name1} is better by {psnr1-psnr2:.2f} dB PSNR")
    else:
        print("Models perform equally")

