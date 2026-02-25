from pathlib import Path

def read_in_og_images(
        top_dir: str | Path
) -> tuple[list[str], list[str], list[str]]:
    """
    Read and categorise original image file paths from directory structure.

    Recursively scans top_dir and categorises images into ground truth
    (GT), noisy and non-deterministic based on filename patterns.
    Expects images to have 'GT' or 'NOISY' in their filenames.

    Args:
        top_dir: Root directory containing subdirectories with images

    Returns:
        Tuple of three lists of file paths:
            - non_det_list: Paths to images without GT or NOISY in name
            - gt_img_list: Paths to ground truth images (contain 'GT')
            - noisy_img_list: Paths to noisy images (contain 'NOISY')

    Note:
        - Scans one level of subdirectories (not fully recursive)
        - Image classification is based on 'GT' and 'NOISY' substrings in filepath
        - Files without these patterns go to non_det_list
        - Returns sorted filepaths
    """
    non_det_list = []
    gt_img_list = []
    noisy_img_list = []

    top_dir = Path(top_dir)

    # Get subdirectories
    subdirs = sorted([d.name for d in top_dir.iterdir() if d.is_dir()])

    # Scan each subdirectory
    for subdir_name in subdirs:
        subdir_path = top_dir / subdir_name

        # Get all files in subdirectory
        for file_path in subdir_path.iterdir():
            if file_path.is_file():
                fp_str = str(file_path)

                # Categorise based on filename
                if "GT" in fp_str:
                    gt_img_list.append(fp_str)
                elif "NOISY" in fp_str:
                    noisy_img_list.append(fp_str)
                else:
                    non_det_list.append(fp_str)

    return non_det_list, gt_img_list, noisy_img_list

def get_filepaths(dir_name: str | Path) -> list[str]:
    """
    Get list of all files and directories in a directory.

    Returns names (not full paths) of all items in the specified directory.
    Does not filter by file type or recurse into subdirectories.

    Args:
        dir_name: Directory that holds images

    Returns:
        Names of files and directories  in dir_name

    Note:
        - Returns only names, not full paths
        - Includes both files and subdirectories
        - Does not filter hidden files
        - Order is not guaranteed
    """
    dir_path = Path(dir_name)
    return [item.name for item in dir_path.iterdir()]

