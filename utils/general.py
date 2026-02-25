from pathlib import Path
from configs import Config
from imports import *

def save_inference(
        inf_gt_list: list[str],
        inf_noisy_list: list[str],
        save_dir: str | Path
) -> None:
    """
    Save inference image pairs for later testing.

    Copies ground truth and noisy images to a separate inference
    directory for final model evaluation. Images are renamed with
    'inf_' prefix and sequential numbering.

    Args:
        inf_gt_list: File paths to ground truth images
        inf_noisy_list: File paths to noisy images
        save_dir: Directory to save inference images in

    Note:
        - Creates save_dir if it doesn't exist
        - Images are saved with sequential numbering: inf_gt_0.png, inf_noisy_0.png etc.
        - Maintains pairing between gt and noisy images
        - Does not modify original images
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    for i in range(len(inf_gt_list)):
        gt_save_name = f"inf_gt_{i}.png"
        noisy_save_name = f"inf_noisy_{i}.png"

        # Read in image of each list
        gt_img = Image.open(inf_gt_list[i])
        noisy_img = Image.open(inf_noisy_list[i])

        # Save each image
        gt_img.save(save_path / gt_save_name)
        noisy_img.save(save_path / noisy_save_name)

def load_data_split(config: Config) -> tuple[list[str], list[str]]:
    """
    Load saved test split created during run_preprocessing().

    Args:
        config: Configuration object with paths

    Returns:
        Tuple of (gt_test_list, noisy_test_list)

    Raises:
        FileNotFoundError: If data_split.json doesn't exist
    """
    cropped_dir = Path(config.paths.cropped_img_root)
    split_file = cropped_dir / "data_split.json"

    if not split_file.exists():
        raise FileNotFoundError(
            f"Data split file not found: {split_file}\n "
            f"Run training pipeline first to create the split"
        )

    with open(split_file, "r") as f:
        split_data = json.load(f)

    return (split_data["gt_test_list"],
            split_data["noisy_test_list"])

