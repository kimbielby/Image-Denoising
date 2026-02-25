from imports import *

def load_model(
        checkpoint_path: str | Path,
        device: str = "cuda"
) -> tuple[nn.Module, torch.device]:
    """
    Load trained denoising model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        device: Device to use (cuda or cpu). Default: "cuda"

    Returns:
        tuple: (model, device)
            - model: Loaded UNet model in eval mode
            - device: Device the model is on
    """
    from models.model import UNet

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config if available
    config = checkpoint.get('config', {})
    init_features = config.get('init_features', 32)

    model = UNet(in_channels=3, out_channels=3, init_features=init_features)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"Best val loss: {checkpoint['best_val_loss']:.4f}")

    return model, device

def _load_and_normalise_image(
        image_path: str | Path
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Load image and normalise to [0,1].

    Args:
        image_path: Path to image file

    Returns:
        Tuple of (normalised_image, og_size)
        - normalised_image: RGB float32 array in [0, 1]
        - og_size: (height, width) tuple

    Raises:
        ValueError: If image cannot be loaded
    """
    try:
        img = Image.open(str(image_path)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to load image {image_path}. Error: {e}")

    img_rgb = np.array(img)
    og_size = img_rgb.shape[:2]     # (height, width)

    # Normalise to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0

    return img_float, og_size

def _save_image(
        image_array: np.ndarray,
        output_path: str | Path
) -> None:
    """
    Save numpy array as image file.

    Args:
        image_array: RGB uint8 array (H, W, 3)
        output_path: Path to save image
    """
    img = Image.fromarray(image_array)
    img.save(str(output_path))

def _ensure_model_loaded(
        checkpoint_path: Optional[str | Path],
        model: Optional[nn.Module],
        device: Optional[str | torch.device]
) -> tuple[nn.Module, torch.device]:
    """
    Ensure model is loaded, either from checkpoint or use provided model.

    Args:
        checkpoint_path: Optional path to checkpoint
        model: Optional pre-loaded model
        device: Optional device to use

    Returns:
        Tuple of (model, device)

    Raises:
        ValueError: If neither checkpoint_path nor model is provided
    """
    if model is None:
        if checkpoint_path is None:
            raise ValueError("Must provide either checkpoint_path or model")
        model, device = load_model(checkpoint_path, device or "cuda")

    if device is None:
        device = next(model.parameters()).device

    return model, device

def create_blend_weight(patch_size: int, overlap: int) -> np.ndarray:
    """
    Create a weight  map for smooth blending of overlapping patches.

    Uses cosine blending in overlap regions for seamless transitions.

    Args:
        patch_size: Size of square patch
        overlap: Overlap size in pixels

    Returns:
        Weight map with smooth falloff at edges
    """
    weight = np.ones((patch_size, patch_size))

    if overlap == 0:
        return weight

    # Create cosine falloff in overlap regions
    for i in range(overlap):
        # Cosine interpolation from 0 to 1
        alpha = 0.5 * (1 - np.cos(np.pi * (i + 1) / overlap))

        # Apply to all four edges
        weight[i, :] *= alpha       # Top edge
        weight[-i-1, :] *= alpha    # Bottom edge
        weight[:, i] *= alpha       # Left edge
        weight[:, -i-1] *= alpha    # Right edge

    return weight

def denoise_image(
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Denoise a single image (processes entire image at once).

    Use this for small images. For large images, use denoise_with_patches().

    Args:
        image_path: Path to noisy input image
        output_path: Path to save denoised image. If None, image is not saved.
            Default: None
        checkpoint_path: Path to checkpoint. Required if model is not
            provided. Default: None
        model: Pre-loaded model. If None, loads from checkpoint_path.
        device: Device to use. If None, uses CUDA if available.

    Returns:
        Denoised image as RGB uint8 array (H, W, 3)

    Raises:
        ValueError: If image cannot be loaded from image_path
        ValueError: If neither checkpoint_path nor model is provided
    """
    # Ensure model is loaded
    model, device = _ensure_model_loaded(checkpoint_path, model, device)

    # Load and normalise image
    img_float, (og_h, og_w) = _load_and_normalise_image(image_path)

    # Pad to nearest multiple of 16
    pad_h = (16 - og_h % 16) % 16
    pad_w = (16 - og_w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        img_float = np.pad(img_float, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')

    # Convert to tensor (HWC -> CHW)
    img_tensor = torch.from_numpy(img_float.transpose(2, 0, 1)).unsqueeze(0)
    img_tensor = img_tensor.to(device)

    # Denoise
    model.eval()
    with torch.no_grad():
        denoised_tensor = model(img_tensor)

    # Convert back to numpy
    denoised_np = denoised_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    denoised_np = np.clip(denoised_np * 255, 0, 255).astype(np.uint8)

    # Remove padding to return to og size
    if pad_h > 0 or pad_w > 0:
        denoised_np = denoised_np[:og_h, :og_w, :]

    # Save if output path provided
    if output_path:
        _save_image(denoised_np, output_path)

    return denoised_np

def denoise_with_patches(
        image_path: str | Path,
        output_path: Optional[str | Path] = None,
        patch_size: int = 512,
        overlap: int = 64,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None
) -> np.ndarray:
    """
    Denoise large image using overlapping patches with weighted blending.

    Better for very large images that don't fit in GPU memory. Uses
    cosine-weighted blending in overlap regions for seamless results.

    Args:
        image_path: Path to noisy input image
        output_path: Path to save denoised image. Default: None
        patch_size: Size of square patches. Default: 512
        overlap: Overlap between adjacent patches for blending. Default: 64
        checkpoint_path: Path to model checkpoint. Required if model is
            not provided
        model: Pre-loaded model
        device: Device to use

    Returns:
        Denoised image as RGB uint8 array (H, W, 3)
    """
    # Ensure model is loaded
    model, device = _ensure_model_loaded(checkpoint_path, model, device)

    # Load and normalise image
    img_float, (h, w) = _load_and_normalise_image(image_path)

    print(f"Processing {w}x{h} image in {patch_size}x{patch_size} tiles (overlap={overlap})...)")

    # Output array and weight map for blending
    output = np.zeros_like(img_float)
    weight_map = np.zeros((h, w), dtype=np.float32)

    # Create blend weight for smooth transitions
    blend_weight = create_blend_weight(patch_size, overlap)

    stride = patch_size - overlap

    # Calculate number of patches
    n_h = ((h - overlap + stride - 1) // stride)
    n_w = ((w - overlap + stride - 1) // stride)
    total_patches = n_h * n_w

    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=total_patches, desc="Processing tiles")

        for i in range(0, h, stride):
            for j in range(0, w, stride):
                # Extract patch with bounds checking
                end_i = min(i + patch_size, h)
                end_j = min(j + patch_size, w)
                start_i = max(0, end_i - patch_size)
                start_j = max(0, end_j - patch_size)

                patch = img_float[start_i:end_i, start_j:end_j, :]

                # Actual patch dimensions (may be smaller at edges)
                actual_h, actual_w = patch.shape[:2]

                # Pad if necessary (edges of image)
                pad_h = patch_size - actual_h
                pad_w = patch_size - actual_w
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)),
                                   mode='reflect')

                # Convert to tensor and process
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0)
                patch_tensor = patch_tensor.to(device)

                denoised_patch = model(patch_tensor)
                denoised_patch = denoised_patch.squeeze(0).cpu().numpy().transpose(1, 2, 0)

                # Remove padding from denoised patch
                if pad_h > 0 or pad_w > 0:
                    denoised_patch = denoised_patch[:actual_h, :actual_w, :]

                # Get corresponding weight for this patch
                patch_weight = blend_weight[:actual_h, :actual_w].copy()

                # Accumulate with weights
                output[start_i:end_i, start_j:end_j, :] += denoised_patch * patch_weight[:, :, np.newaxis]
                weight_map[start_i:end_i, start_j:end_j] += patch_weight

                pbar.update(1)

        pbar.close()

    # Normalise by weight map (average weighted overlaps)
    output = output / (weight_map[:, :, np.newaxis] + 1e-8)   # Add small epsilon to avoid divide by zero
    output = np.clip(output * 255, 0, 255).astype(np.uint8)

    print(f"Processed {total_patches} tiles successfully")

    # Save if output path provided
    if output_path:
        _save_image(output, output_path)

    return output

def denoise_directory(
        input_dir: str | Path,
        output_dir: str | Path,
        checkpoint_path: Optional[str | Path] = None,
        model: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        use_patches: bool = False,
        patch_size: int = 512,
        overlap: int = 64
) -> None:
    """
    Denoise all images in a directory.

    Args:
        input_dir: Path to directory with images to denoise
        output_dir: Path to directory to save denoised images
        checkpoint_path: Path to checkpoint. Required if model is not provided
        model: Pre-loaded model
        device: Device to use
        use_patches: Whether to use patch-based processing. Default: False
        patch_size: Size of patches if use_patches=True. Default: 512
        overlap: Overlap between adjacent patches if use_patches=True.
            Default: 64
    """
    # Ensure model is loaded
    model, device = _ensure_model_loaded(checkpoint_path, model, device)

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']:
        image_files.extend(list(input_path.glob(ext)))

    print(f"Found {len(image_files)} images")

    for img_path in tqdm(image_files, desc="Denoising images"):
        output_file = output_path / img_path.name

        if use_patches:
            denoise_with_patches(
                image_path=img_path,
                output_path=output_file,
                patch_size=patch_size,
                overlap=overlap,
                model=model,
                device=device
            )
        else:
            denoise_image(
                image_path=img_path,
                output_path=output_file,
                model=model,
                device=device
            )

    print(f"Saved denoised images to {output_dir}")


