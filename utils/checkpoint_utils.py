from typing import Any
from imports import *
import shutil

def load_checkpoint_inference(
        checkpoint_path: str | Path,
        model: nn.Module,
        device: str = "cuda"
) -> nn.Module:
    """
    Load checkpoint for inference only (no optimiser or scheduler).

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model on. Default: "cuda".

    Returns:
        Model with loaded weights in eval mode
    """

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f" Loaded model from {checkpoint_path}")
    print(f"    Epoch: {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"    Val Loss: {checkpoint['best_val_loss']:.4f}")
    if checkpoint.get("is_best"):
        print(f"    This was the best model")
    if checkpoint.get("is_final"):
        print(f"    This was final model")

    return model

def load_checkpoint_training(
        checkpoint_path: str | Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: str = "cuda"
) -> tuple[int, float]:
    """
    Load checkpoint to resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        device: Device to load model on. Default: "cuda".

    Returns:
        tuple: (start_epoch, best_val_loss)
            - start_epoch: Epoch number to resume from
            - best_val_loss: Best validation loss from previous training
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Resuming training from {checkpoint_path}")
    print(f"    Will start at epoch: {start_epoch}")
    print(f"    Best val loss so far: {best_val_loss:.4f}")

    return start_epoch, best_val_loss

def get_checkpoint_info(checkpoint_path: str | Path) -> dict[str, Any]:
    """
    Get checkpoint information without loading the model.

    Args:
    checkpoint_path: Path to checkpoint file

    Returns:
        Dictionary of checkpoint metadata containing:
            - epoch: Training epoch
            - best_val_loss: Best validation loss
            - is_best: Whether this was the best model or not
            - is_final: Whether this is the final model or not
            - metrics: Training metrics
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    info = {
        "epoch": checkpoint.get("epoch"),
        "best_val_loss": checkpoint.get("best_val_loss"),
        "is_best": checkpoint.get("is_best", False),
        "is_final": checkpoint.get("is_final", False),
        "metrics": checkpoint.get("metrics", {})
    }
    return info

def list_checkpoints(checkpoint_dir: str | Path) -> None:
    """
    List all checkpoints found in directory with their info.

    Prints checkpoint information including epoch and validation loss
    for best_model.pth, final_model.pth and regular checkpoints.

    Args:
        checkpoint_dir: Path to checkpoint directory
    """
    checkpoint_dir = Path(checkpoint_dir)

    if not checkpoint_dir.exists():
        print(f"Directory not found: {checkpoint_dir}")
        return

    print(f"\nCheckpoints in {checkpoint_dir}:")
    print("="*60)

    # Check for best model
    best_path = checkpoint_dir / "best_model.pth"
    if best_path.exists():
        info = get_checkpoint_info(best_path)
        print(f"best_model.pth")
        print(f"    Epoch: {info['epoch']}, Val Loss: {info['best_val_loss']:.4f}")

    # Check for final model
    final_path = checkpoint_dir / "final_model.pth"
    if final_path.exists():
        info = get_checkpoint_info(final_path)
        print(f"final_model.pth")
        print(f"    Epoch: {info['epoch']}, Val Loss: {info['best_val_loss']:.4f}")

    # List regular checkpoints
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
    if checkpoints:
        print(f"\nRegular checkpoints:")
        for ckpt_path in checkpoints:
            info = get_checkpoint_info(ckpt_path)
            print(f"    {ckpt_path.name}")
            print(f"    Epoch: {info['epoch']}, Val Loss: {info['best_val_loss']:.4f}")

    print("="*60)

def cleanup_old_runs(
        runs_dir: str | Path,
        keep_last_k: int = 5
) -> None:
    """
    Clean up old training runs, keeping only the most recent k runs.

    Deletes old timestamped run directories, preserving only the most
    recent runs based on directory name (timestamp). Use with caution
    as deletion is permanent.

    Args:
        runs_dir: Path to directory containing training runs
        keep_last_k: Number of most recent runs to keep. Default: 5

    Warning:
        This permanently deletes directories. Cannot be undone.
    """

    runs_dir = Path(runs_dir)

    if not runs_dir.exists():
        print(f"Directory not found: {runs_dir}")
        return

    # Get all timestamped run directories
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    run_dirs.sort(key=lambda x: x.name)     # Sort by timestamp

    if len(run_dirs) <= keep_last_k:
        print(f"Only {len(run_dirs)} runs found, nothing to clean up")
        return

    # Delete old runs
    runs_to_delete = run_dirs[:-keep_last_k]

    print(f"Cleaning up old runs (keeping last {keep_last_k})")
    for run_dir in runs_to_delete:
        try:
            shutil.rmtree(run_dir)
            print(f"Deleted {run_dir.name}")
        except Exception as e:
            print(f"Failed to delete {run_dir.name}: {e}")

    print(f"Cleanup complete! Kept {keep_last_k} most recent runs")



