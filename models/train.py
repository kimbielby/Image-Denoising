from typing import Callable, TypeAlias
from utils import *
from imports import *
from configs import Config
from collections import deque
from dataclasses import asdict
from torch import amp
from torch.nn.utils import clip_grad_norm_
import time

TrainingHistory: TypeAlias = dict[str, list]
ValidationFunction: TypeAlias = Callable[
    [torch.nn.Module, str],
    tuple[float, float, float]  # (loss, psnr, ssim)
]

class EarlyStopping:
    """
    Early stopping to halt training when validation loss stops improving.
    """
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        """
        Initialise early stopping object.

        Args:
            patience: Epochs to wait for improvement before stopping
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """
        Update early stopping state with latest validation loss.

        Args:
            val_loss: Current epoch validation loss

        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found - reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            print(f"    Early stopping: {self.counter}/{self.patience} "
                  f"(no improvement > {self.min_delta})")

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"\n Early stopping triggered after {self.patience} "
                      f"epochs without improvement")

        return self.should_stop

def _now() -> str:
    """
    Generate timestamp string for checkpoint directories.
    """
    return time.strftime("%Y%m%d-%H%M%S")

def save_comparison(
        epoch: int,
        model: torch.nn.Module,
        val_loader: DataLoader,
        checkpoint_dir: Path,
        device: str | torch.device,
) -> None:
    """
    Save visual comparison of noisy, denoised and ground truth images.

    Creates a side-by-side comparison showing
    [noisy | denoised | ground truth] and saves to
    checkpoint_dir/comparisons/epoch_XXX.png.

    Args:
        epoch: Current Epoch number
        model: Trained model
        val_loader: Validation dataloader
        checkpoint_dir: Directory to save comparison images
        device: Device (cuda or cpu)

    Returns:
        None: Saves image to disk
    """
    model.eval()

    # Get a batch from validation
    noisy, gt = next(iter(val_loader))
    noisy = noisy.to(device)
    gt = gt.to(device)

    with torch.no_grad():
        denoised = model(noisy)

    # Convert to numpy
    noisy_np = noisy[0].cpu().numpy().transpose(1, 2, 0)
    gt_np = gt[0].cpu().numpy().transpose(1, 2, 0)
    denoised_np = denoised[0].cpu().numpy().transpose(1, 2, 0)

    # Clip and convert to uint8
    noisy_np = np.clip(noisy_np * 255, 0, 255).astype(np.uint8)
    gt_np = np.clip(gt_np * 255, 0, 255).astype(np.uint8)
    denoised_np = np.clip(denoised_np * 255, 0, 255).astype(np.uint8)

    # Concatenate horizontally [noisy | denoised | gt]
    comparison = np.hstack([noisy_np, denoised_np, gt_np])

    # Create comparisons subdirectory
    comparisons_dir = checkpoint_dir / "comparisons"
    comparisons_dir.mkdir(exist_ok=True)

    save_path = comparisons_dir / f"epoch_{epoch:03d}.png"
    Image.fromarray(comparison).save(str(save_path))

def _load_checkpoint(
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
        device: str = "cpu"
) -> tuple[int, float]:
    """
    Load model checkpoint to resume training.

    Note:
        This function is used internally by train(). For loading checkpoints
        in other contexts, use load_checkpoint_training() from
        checkpoint_utils.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance
        optimizer: Optimizer instance
        scheduler: Scheduler instance
        device: Device to load on. Default: "cpu"

    Returns:
        Tuple of (epoch, best_val_loss)
            - epoch: Epoch number to resume from
            - best_val_loss: Best validation loss
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    print(f"Resumed from epoch {epoch}")

    return epoch, best_val_loss

def _get_loss_function(
        loss_fn_name: str,
        alpha: float = 0.8
) -> nn.Module:
    """
    Get loss function by name (handles both torch.nn and custom losses).
    """
    if hasattr(nn, loss_fn_name):
        return getattr(nn, loss_fn_name)()

    from .losses import CombinedLoss
    custom_losses = {
        "CombinedLoss": lambda: CombinedLoss(alpha=alpha)
    }

    if loss_fn_name in custom_losses:
        return custom_losses[loss_fn_name]()

    raise ValueError(f"Unknown loss function {loss_fn_name}")

def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        config: Config,
        device: str = "cuda",
        val_fn: Optional[ValidationFunction] = None,
        val_loader: Optional[DataLoader] = None,
        resume_path: Optional[str | Path] = None,
        psnr_metric: Optional[PeakSignalNoiseRatio] = None,
        ssim_metric: Optional[StructuralSimilarityIndexMeasure] = None,
        save_comparisons: bool = True
) -> tuple[TrainingHistory, Path]:
    """
    Train model with checkpoint management and optional visual comparisons

    Args:
        model: Model to train
        train_loader: Training data loader
        config: Configuration object
        device: Device to train on. Default: "cuda"
        val_fn: Validation function returning (loss, psnr, ssim)
        val_loader: Validation loader for comparison images
        resume_path: Path to checkpoint file to resume training from
        psnr_metric: Pre-created PSNR metric
        ssim_metric: Pre-created SSIM metric
        save_comparisons: Save visual comparison images. Default: True

    Returns:
        Tuple of (training_history, checkpoint_dir):
            - training_history: Dict with keys epoch, train_loss, val_loss etc.
            - checkpoint_dir: Path to directory containing checkpoints
    """
    print(f"\nStarting training on {device}")
    print(f"Batch size: {config.train.batch_size}")
    print(f"Epochs: {config.train.epochs}")

    start_time = time.time()

    # Create timestamped directory
    checkpoint_root = Path(getattr(config.paths, "checkpoint_dir", "../runs"))
    checkpoint_dir = checkpoint_root / _now()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to {checkpoint_dir}")

    # Set up for keeping last k checkpoints
    keep_last_k = getattr(config.train, "keep_last_k", 3)
    checkpoint_paths = deque(maxlen=keep_last_k)
    print(f"Will keep last {keep_last_k} checkpoints")

    model.to(device)

    # Setup loss and optimizer
    loss_fn_name = config.loss.name
    loss_function = _get_loss_function(loss_fn_name, alpha=config.loss.alpha)
    lr = getattr(config.train, "learning_rate", 1e-4)
    print(f"Learning rate: {lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Mixed precision
    use_amp = (torch.device(device).type == "cuda")
    scaler = amp.GradScaler("cuda") if use_amp else None
    ac = amp.autocast(device_type="cuda") if use_amp else None

    # Learning rate scheduler
    if config.scheduler.type == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.scheduler.mode,
            factor=config.scheduler.factor,
            patience=config.scheduler.patience,
            min_lr=config.scheduler.min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler type {config.scheduler.type}")

    history = {
        "epoch": [], "train_loss": [], "train_psnr": [], "train_ssim": [],
        "val_loss": [], "val_psnr": [], "val_ssim": []
    }

    # Initialise best validation loss
    start_epoch = 0
    best_val_loss = float("inf")

    if resume_path:
        start_epoch, best_val_loss = _load_checkpoint(
            checkpoint_path=resume_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        print(f"Resumed training with best_val_loss: {best_val_loss:.4f}")

    calculate_train_metrics = (
            psnr_metric is not None and ssim_metric is not None
    )

    # Setup early stopping
    patience = getattr(config.train, "patience", 10)
    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)
    print(f"Early stopping patience: {patience} epochs\n")

    # Start the epoch loop
    for epoch in range(start_epoch, config.train.epochs):
        epoch_start = time.time()
        model.train()

        # Initialise values
        num_batches = 0
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        # Iterate through train_loader
        for i, (noisy, gt) in enumerate(pbar, start=1):
            noisy = noisy.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision training
            if scaler is not None:
                with ac:
                    output = model(noisy)
                    loss = loss_function(output, gt)
                scaler.scale(loss).backward()

                # Gradient clipping for mixed precision
                scaler.unscale_(optimizer=optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(noisy)
                loss = loss_function(output, gt)
                loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            # Update metrics
            batch_loss = loss.item()
            num_batches += 1
            total_loss += batch_loss

            if calculate_train_metrics:
                total_psnr += get_batch_psnr(output, gt, psnr_metric)
                total_ssim += get_batch_ssim(output, gt, ssim_metric)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'batch': f'{i}/{len(train_loader)}',
            })
            # Exit the enumerator loop

        # Calculate Loss, PSNR and SSIM averages
        train_loss = total_loss / max(num_batches, 1)
        train_psnr = total_psnr / max(num_batches, 1)
        train_ssim = total_ssim / max(num_batches, 1)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch} training completed in {epoch_time / 60:.2f} "
              f"minutes")
        print(f"    Train:  Loss={train_loss:.4f}  PSNR={train_psnr:.2f}   "
              f"SSIM={train_ssim:.4f}")

        """ VALIDATION and BEST MODEL TRACKING """
        val_loss = None
        val_psnr = None
        val_ssim = None
        is_best = False

        if val_fn and (epoch % config.validation.val_every == 0):
            val_loss, val_psnr, val_ssim = val_fn(model, device)
            print(f"{epoch=}: \n    "
                  f"Val:    Loss: {val_loss:.4f}     PSNR={val_psnr:.2f}    "
                  f"SSIM={val_ssim:.4f}  \n")

            # Save visual comparison
            if save_comparisons and config.validation.save_comparisons and val_loader is not None:
                try:
                    save_comparison(
                        epoch=epoch,
                        model=model,
                        val_loader=val_loader,
                        checkpoint_dir=checkpoint_dir,
                        device=device
                    )
                    print(f"    Saved comparison image")
                except Exception as e:
                    print(f"    Could not save comparison image: {e}")

            # Update learning rate scheduler
            scheduler.step(val_loss)

            # Check if this is the best model so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                is_best = True
                print(f"New best validation loss: {best_val_loss:.4f}")

            if early_stopping.step(val_loss):
                # Save history and checkpoint before stopping
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["train_psnr"].append(train_psnr)
                history["train_ssim"].append(train_ssim)
                history["val_loss"].append(val_loss)
                history["val_psnr"].append(val_psnr)
                history["val_ssim"].append(val_ssim)
                break

        # Append to Lists in history set
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_psnr"].append(train_psnr)
        history["train_ssim"].append(train_ssim)
        history["val_loss"].append(val_loss)
        history["val_psnr"].append(val_psnr)
        history["val_ssim"].append(val_ssim)

        # Prepare metrics for checkpoint
        metrics = {
            "train_loss": train_loss,
            "train_psnr": train_psnr,
            "train_ssim": train_ssim,
            "val_loss": val_loss,
            "val_psnr": val_psnr,
            "val_ssim": val_ssim
        }

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": asdict(config),
            "metrics": metrics
        }

        # Save regular checkpoints
        save_every = getattr(config.train, "save_every", 5)
        should_save_regularly = (epoch + 1) % save_every == 0

        if should_save_regularly:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")

            # Manage keep_last_k
            checkpoint_paths.append(checkpoint_path)
            all_checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pth"))
            checkpoints_to_keep = set(checkpoint_paths)

            for checkpoint_path in all_checkpoints:
                if checkpoint_path not in checkpoints_to_keep:
                    try:
                        checkpoint_path.unlink()
                        print(f"Deleted old checkpoint at {checkpoint_path.name}")
                    except Exception as e:
                        print(f"Could not delete old checkpoint {checkpoint_path.name}: {e}")

        # Save best model
        if is_best:
            best_path = checkpoint_dir / "best_model.pth"
            checkpoint["is_best"] = True
            torch.save(checkpoint, best_path)
            print(f"Saved best model as {best_path.name}    (val_loss: {best_val_loss:.4f})\n")
            checkpoint["is_best"] = False

        # Save final model at last epoch
        if epoch == config.train.epochs - 1:
            final_path = checkpoint_dir / "final_model.pth"
            final_checkpoint = checkpoint.copy()
            final_checkpoint["is_final"] = True
            torch.save(final_checkpoint, final_path)
            print(f"Saved final model as {final_path.name}")
            # Note if final is also best
            if val_loss is not None and val_loss == best_val_loss:
                print(f"    (Final model is also the best model)")

    # Training complete
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"\nTraining completed in {elapsed / 3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final model saved at {checkpoint_dir}")
    print(f"{'='*60}")

    return history, checkpoint_dir


