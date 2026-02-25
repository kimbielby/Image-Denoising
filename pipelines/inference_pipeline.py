from utils import *
from imports import *
from inference import *
from configs import *
from typing import Optional

class InferencePipeline:
    """
    Inference pipeline for denoising new images.

    Loads trained model and runs inference on individual images or batches.
    Automatically uses tiled processing for large images.
    """
    def __init__(
            self,
            config: Config,
            checkpoint_dir: Optional[str | Path] = None
    ) -> None:
        """
        Initialise inference pipeline.

        Args:
            config: Configuration object
            checkpoint_dir: Directory with trained model (optional)
        """
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Initialise attributes
        self.model = None
        self.device = None
        self.inference_results_dir = None

    def load_model(
            self,
            checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        """
        Load trained model for inference.

        Args:
            checkpoint_path: Path to checkpoint file. If None, uses
                checkpoint_dir/best_model.pth
        """
        print("\n" + "="*70)
        print("LOADING MODEL FOR INFERENCE")
        print("="*70)

        # Determine checkpoint path
        if checkpoint_path is None:
            if self.checkpoint_dir is None:
                raise ValueError("Must provide either checkpoint_path or checkpoint_dir")
            checkpoint_path = self.checkpoint_dir / 'best_model.pth'
        else:
            checkpoint_path = Path(checkpoint_path)
            if self.checkpoint_dir is None:
                self.checkpoint_dir = checkpoint_path.parent

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load model
        self.model, self.device = load_model(checkpoint_path=checkpoint_path)

        print(f"    Model loaded from: {checkpoint_path}")
        print(f"    Using device: {self.device}")

    def denoise_single_image(
            self,
            image_path: str | Path,
            output_path: str | Path,
    ) -> np.ndarray:
        """
        Denoise a single image.

        Automatically chooses between full-image and tiled processing
        based on image size for optimal memory usage.

        Args:
            image_path: Path to noisy input image
            output_path: Path to save denoised output

        Returns:
            Denoised image as numpy array
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Check image size to determine processing method
        img = Image.open(image_path)
        w, h = img.size
        megapixels = (w * h) / 1e6

        # Automatically use tiling for large images (> 1 megapixel)
        if megapixels > 1.0:
            if self.config.inference.tiled:
                print(f"Large image ({w}x{h}, {megapixels:.1f}MP) - using tiled processing")
                denoised = denoise_with_patches(
                    image_path=image_path,
                    output_path=output_path,
                    patch_size=self.config.inference.tile_size,     # Matches training patch size
                    overlap=self.config.inference.overlap,             # 12.5% overlap for seamless blending
                    model=self.model,
                    device=self.device
                )
        else:
            print(f"Small image {w}x{h}, {megapixels:.1f}MP - processing full image")
            denoised = denoise_image(
                image_path=image_path,
                output_path=output_path,
                model=self.model,
                device=self.device
            )

        return denoised

    def run_inference(
            self,
            num_images: Optional[int] = None
    ) -> None:
        """
        Run inference on saved inference images.

        Denoised all inference images from config.paths.inference_dir.

        Args:
            num_images: Number of images to denoise. If None, uses config.dataset.num_inference_imgs
        """
        print("\n" + "="*70)
        print("RUNNING INFERENCE")
        print("="*70)

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Create inference results directory
        if self.checkpoint_dir:
            self.inference_results_dir = self.checkpoint_dir / "inference_results"
        else:
            self.inference_results_dir = Path("inference_results")

        self.inference_results_dir.mkdir(exist_ok=True)

        # Determine number of images
        num_images = num_images or self.config.dataset.num_inference_imgs

        # Denoise all inference images
        print(f"Denoising {num_images} images...")
        inference_dir = Path(self.config.paths.inference_dir)

        for i in range(num_images):
            noisy_path = inference_dir / f"inf_noisy_{i}.png"
            denoised_path = self.inference_results_dir / f"denoised_{i}.png"

            if not noisy_path.exists():
                print(f"Warning: {noisy_path} not found, skipping...")
                continue

            self.denoise_single_image(
                image_path=noisy_path,
                output_path=denoised_path
            )

        print(f"\nDenoised {num_images} images")
        print(f"Saved to {self.inference_results_dir}")

    def save_examples(
            self,
            num_examples: int = 3
    ) -> None:
        """
        Save example comparison images.

        Creates side-by-side [noisy | denoised | gt] comparisons.

        Args:
            num_examples: Number of examples to save
        """
        print("\n" + "="*70)
        print(f"SAVING {num_examples} EXAMPLE COMPARISONS")
        print("="*70)

        if self.checkpoint_dir is None:
            raise ValueError("checkpoint_dir not set. Cannot save examples.")

        save_example_comparisons(
            checkpoint_dir=self.checkpoint_dir,
            inference_dir=self.config.paths.inference_dir,
            denoised_dir=self.inference_results_dir,
            num_examples=num_examples
        )

    def run_inference_workflow(
            self,
            num_images: Optional[int] = None,
            num_examples: int = 3
    ) -> None:
        """
        Run complete inference workflow: denoise -> save examples.

        Args:
            num_images: Number of images to denoise (optional)
            num_examples: Number of comparison examples (default: 3)
        """
        self.run_inference(num_images=num_images)
        self.save_examples(num_examples=num_examples)

    def denoise_directory(
            self,
            input_dir: str | Path,
            output_dir: str | Path
    ) -> None:
        """
        Denoise all images in a directory.

        Args:
            input_dir: Directory with noisy images
            output_dir: Directory to save denoised images
        """
        print("\n" + "="*70)
        print("BATCH INFERENCE")
        print("="*70)

        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get all image files
        image_files = list(input_dir.glob("*.png"))

        print(f"Found {len(image_files)} images in {input_dir}")
        print(f"Denoising...")

        for img_path in image_files:
            output_path = output_dir / f"denoised_{img_path.name}"
            self.denoise_single_image(
                image_path=img_path,
                output_path=output_path
            )

        print(f"\n Denoised {len(image_files)} images ")
        print(f"Saved to {output_dir}")

    def get_summary(self) -> None:
        """
        Print summary of inference results.
        """
        print("\n" + "="*70)
        print("INFERENCE SUMMARY")
        print("="*70)

        print(f"Model loaded: {self.model is not None}")
        print(f"Device: {self.device}")
        
        if self.inference_results_dir and self.inference_results_dir.exists():
            num_denoised = len(list(self.inference_results_dir.glob("*.png")))
            print(f"\nDenoised images: {num_denoised}")
            print(f"Saved to {self.inference_results_dir}")

        if self.checkpoint_dir:
            examples_dir = self.checkpoint_dir / "figures" / "examples"
            if examples_dir.exists():
                num_examples = len(list(examples_dir.glob("*.png")))
                print(f"\nComparison examples: {num_examples}")
                print(f"Saved to {examples_dir}")

        print("="*70)

def run_inference_pipeline(
        config: Config,
        checkpoint_path: str | Path,
        num_images: Optional[int] = None,
        num_examples: int = 3
) -> InferencePipeline:
    """
    Run complete inference pipeline in one function call.

    Args:
        config: Configuration object
        checkpoint_path: Path to checkpoint file
        num_images: Number of images to denoise (optional)
        num_examples: Number of comparison examples (default: 3)

    Returns:
        InferencePipeline object with inference results
    """
    pipeline = InferencePipeline(config=config)
    pipeline.load_model(checkpoint_path=checkpoint_path)
    pipeline.run_inference_workflow(num_images=num_images, num_examples=num_examples)
    pipeline.get_summary()

    return pipeline
