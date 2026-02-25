from imports import *
from typing import Dict, Optional

def print_test_statistics(results: Dict) -> None:
    """
    Print comprehensive statistical analysis of test results.

    Shows mean, median, std, min, max for both PSNR and SSIM metrics.

    Args:
        results: Test results dictionary from test() function
    """
    psnr_values = [img["psnr"] for img in results["per_image"]]
    ssim_values = [img["ssim"] for img in results["per_image"]]

    print("=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    print(f"\nPSNR Statistics:")
    print(f"    Mean: {np.mean(psnr_values):.2f} dB")
    print(f"    Std: {np.std(psnr_values):.2f} dB")
    print(f"    Min: {np.min(psnr_values):.2f} dB")
    print(f"    Max: {np.max(psnr_values):.2f} dB")
    print(f"    Median: {np.median(psnr_values):.2f} dB")

    print(f"\nSSIM Statistics:")
    print(f"    Mean: {np.mean(ssim_values):.4f}")
    print(f"    Std: {np.std(ssim_values):.4f}")
    print(f"    Min: {np.min(ssim_values):.4f}")
    print(f"    Max: {np.max(ssim_values):.4f}")
    print(f"    Median: {np.median(ssim_values):.4f}")

    print("=" * 60)

def compare_with_validation(test_results: Dict, history: Dict) -> None:
    """
    Compare test results with validation results from training.

    Shows whether the model generalises well by comparing best validation
    metrics with test metrics.

    Args:
        test_results: Test results dictionary
        history: Training history dictionary
    """
    # Get best validation metrics
    val_psnr = [x for x in history["val_psnr"] if x is not None]
    val_ssim = [x for x in history["val_ssim"] if x is not None]

    if not val_psnr:
        print(f"No validation data available for comparison")
        return

    best_val_psnr = max(val_psnr)
    best_val_ssim = max(val_ssim)

    # Get test metrics
    test_psnr = test_results["avg_psnr"]
    test_ssim = test_results["avg_ssim"]

    # Calculate gaps
    psnr_gap = best_val_psnr - test_psnr
    ssim_gap = best_val_ssim - test_ssim

    print("=" * 60)
    print("GENERALISATION ANALYSIS")
    print("=" * 60)

    print(f"\nPSNR:")
    print(f"    Best val: {best_val_psnr:.2f} dB")
    print(f"    Test: {test_psnr:.2f} dB")
    print(f"    Gap: {psnr_gap:+.2f} dB")

    print(f"\nSSIM:")
    print(f"    Best val: {best_val_ssim:.4f}")
    print(f"    Test: {test_ssim:.4f}")
    print(f"    Gap: {ssim_gap:+.4f}")

    print(f"\nAssessment:")
    if abs(psnr_gap) < 0.5:
        print(f"    Excellent generalisation (gap < 0.5 dB)")
    elif abs(psnr_gap) < 1.0:
        print(f"    Good generalisation (gap < 1.0 dB)")
    elif abs(psnr_gap) < 2.0:
        print(f"    Fair generalisation (gap < 2.0 dB)")
    else:
        print(f"    Poor generalisation (gap ≥ 2.0 dB)")
        if psnr_gap > 0:
            print(f"    -> Model may be overfitting")
        else:
            print(f"    ->Test set may be harder than validation")

    print("=" * 60)

def print_evaluation_summary(
        test_results: Dict,
        history: Optional[Dict] = None
) -> None:
    """
    Print comprehensive evaluation summary.

    Combines test statistics with optional comparison to training/validation.

    Args:
        test_results: Test results dictionary
        history: Optional training history for comparison
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Test results
    print(f"\n Test Results:")
    print(f"    Images tested: {test_results['num_images']}")
    print(f"    Avg PSNR: {test_results['avg_psnr']:.2f} dB")
    print(f"    Avg SSIM: {test_results['avg_ssim']:.4f}")

    # Statistics
    psnr_values = [img["psnr"] for img in test_results["per_image"]]
    print(f"\n PSNR Distribution:")
    print(f"    Range: [{np.min(psnr_values):.2f}, {np.max(psnr_values):.2f}] dB")
    print(f"    Std: {np.std(psnr_values):.2f} dB")

    # Comparison with validation if available
    if history is not None:
        val_psnr = [x for x in history["val_psnr"] if x is not None]
        if val_psnr:
            best_val_psnr = max(val_psnr)
            gap = best_val_psnr - test_results["avg_psnr"]

            print(f"\n Val/Test Gap:")
            print(f"    Best val PSNR: {best_val_psnr:.2f} dB")
            print(f"    Gap: {gap:+.2f} dB")

            if abs(gap) < 1.0:
                print(f"    Good generalisation")
            else:
                print(f"    Check generalisation")

    print("=" * 60)

def analyse_failure_cases(
        test_results: Dict,
        threshold_psnr: float = 25.0
) -> None:
    """
    Analyse images where the model performed poorly.

    Identifies failure cases below a PSNR threshold and provides statistics.

    Args:
        test_results: Test results dictionary
        threshold_psnr: PSNR threshold below which images are considered failures
    """
    per_image = test_results["per_image"]
    failures = [
        (idx, img) for idx, img in enumerate(per_image)
        if img["psnr"] < threshold_psnr
    ]

    print("=" * 60)
    print(f"FAILURE CASE ANALYSIS (PSNR < {threshold_psnr:.1f} dB)")
    print("=" * 60)

    if not failures:
        print(f"\n No failures! All images above {threshold_psnr:.1f} dB PSNR")
        print("=" * 60)
        return

    failure_rate = len(failures) / len(per_image) * 100
    failure_psnr = [img["psnr"] for _, img in failures]

    print(f"\n Failed images: {len(failures)} / {len(per_image)} ({failure_rate:.1f}%)")
    print(f"\nFailure statistics:")
    print(f"    Mean PSNR: {np.mean(failure_psnr):.2f} dB")
    print(f"    Worst PSNR: {np.min(failure_psnr):.2f} dB")
    print(f"    Std: {np.std(failure_psnr):.2f} dB")

    # Show worst cases
    worst_cases = sorted(failures, key=lambda x: x[1]["psnr"])[:5]
    print(f"\nWorst {min(5, len(worst_cases))} cases:")
    for i, (idx, case) in enumerate(worst_cases, 1):
        print(f"    {i}. Image {idx:3d}: PSNR={case['psnr']:.2f} dB, SSIM={case['ssim']:.4f}")

    print("=" * 60)

def print_percentile_analysis(test_results: Dict) -> None:
    """
    Print percentile analysis of test results.

    Shows 25th, 50th (median), 75th, 90th and 95th percentiles for PSNR.

    Args:
        test_results: Test results dictionary
    """
    psnr_values = [img["psnr"] for img in test_results["per_image"]]

    percentiles = [25, 50, 75, 90, 95]
    psnr_percentiles = np.percentile(psnr_values, percentiles)

    print("=" * 60)
    print("PERCENTILE ANALYSIS")
    print("=" * 60)

    for p, val in zip(percentiles, psnr_percentiles):
        suffix = "  (median)" if p == 50 else ""
        print(f"{p}th percentile: {val:.2f} dB{suffix}")

    # Interpretation
    iqr = psnr_percentiles[2] - psnr_percentiles[0]     # 75th - 25th
    print(f"\nInterquartile range (IQR): {iqr:.2f} dB")

    if iqr < 2.0:
        print("Very consistent performance across images")
    elif iqr < 3.0:
        print("Consistent performance")
    else:
        print("High variability in performance across images")

    print("=" * 60)

def create_evaluation_report(
        test_results: Dict,
        history: Optional[Dict] = None,
        checkpoint_dir: Optional[str] = None
) -> str:
    """
    Create comprehensive text evaluation report.

    Generates a formatted text report with all evaluation metrics,
    statistics and analysis. Can be saved to file.

    Args:
        test_results: Test results dictionary
        history: Optional training history
        checkpoint_dir: Optional checkpoint directory for reference

    Returns:
        Formatted evaluation report as string
    """
    lines = ["=" * 70, "EVALUATION REPORT", "=" * 70]

    # Basic info
    if checkpoint_dir:
        lines.append(f"\nCheckpoint: {checkpoint_dir}")

    lines.append(f"\nTest Set:")
    lines.append(f"     Images: {test_results["num_images"]}")

    # Overall metrics
    lines.append(f"\nOverall Performance:")
    lines.append(f"     Average PSNR: {test_results['avg_psnr']:.2f} dB")
    lines.append(f"     Average SSIM: {test_results['avg_ssim']:.4f}")

    # Statistics
    psnr_values = [img["psnr"] for img in test_results["per_image"]]
    ssim_values = [img["ssim"] for img in test_results["per_image"]]

    lines.append(f"\nPSNR Statistics:")
    lines.append(f"  Mean:   {np.mean(psnr_values):.2f} dB")
    lines.append(f"  Median: {np.median(psnr_values):.2f} dB")
    lines.append(f"  Std:    {np.std(psnr_values):.2f} dB")
    lines.append(f"  Range:  [{np.min(psnr_values):.2f}, {np.max(psnr_values):.2f}] dB")

    lines.append(f"\nSSIM Statistics:")
    lines.append(f"  Mean:   {np.mean(ssim_values):.4f}")
    lines.append(f"  Median: {np.median(ssim_values):.4f}")
    lines.append(f"  Std:    {np.std(ssim_values):.4f}")
    lines.append(f"  Range:  [{np.min(ssim_values):.4f}, {np.max(ssim_values):.4f}]")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    psnr_percentiles = np.percentile(psnr_values, percentiles)

    lines.append(f"\nPercentiles (PSNR):")
    for p, val in zip(percentiles, psnr_percentiles):
        lines.append(f"  {p}th: {val:.2f} dB")

    # Comparison with validation
    if history is not None:
        val_psnr = [x for x in history['val_psnr'] if x is not None]
        if val_psnr:
            best_val = max(val_psnr)
            gap = best_val - test_results['avg_psnr']

            lines.append(f"\nGeneralization:")
            lines.append(f"  Best val PSNR: {best_val:.2f} dB")
            lines.append(f"  Test PSNR:     {test_results['avg_psnr']:.2f} dB")
            lines.append(f"  Gap:           {gap:+.2f} dB")

            if abs(gap) < 1.0:
                lines.append(f"  Assessment: Good generalization ✓")
            else:
                lines.append(f"  Assessment: Check generalization")

    # Best and worst
    per_image = test_results["per_image"]
    best_idx = max(range(len(per_image)), key=lambda i: per_image[i]["psnr"])
    worst_idx = min(range(len(per_image)), key=lambda i: per_image[i]["psnr"])

    best = per_image[best_idx]
    worst = per_image[worst_idx]

    lines.append(f"\nBest result:")
    lines.append(f"  Image {best_idx}: PSNR={best['psnr']:.2f} dB, SSIM={best['ssim']:.4f}")

    lines.append(f"\nWorst result:")
    lines.append(f"  Image {worst_idx}: PSNR={worst['psnr']:.2f} dB, SSIM={worst['ssim']:.4f}")

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)

