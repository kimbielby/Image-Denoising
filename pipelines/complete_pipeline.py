from pipelines import TrainingPipeline, InferencePipeline, TestingPipeline
from utils import *
from imports import *
from configs import Config
from typing import Optional

class CompletePipeline:
    def __init__(self, config: Config) -> None:
        """
        Initialise complete pipeline.

        Args:
            config: Configuration object.
        """
        self.config = config

        # Initialise sub-pipelines
        self.training_pipeline = None
        self.testing_pipeline = None
        self.inference_pipeline = None

        # Shared attributes
        self.checkpoint_dir = None

    def run_training(
            self,
            train_size: int = 1000,
            val_size: int = 150,
            test_size: int = 150,
            resume_path: Optional[str | Path] = None
    ) -> None:
        """
        Run training workflow using TrainingPipeline.

        Args:
            train_size: Number of training pairs
            val_size: Number of validation pairs
            test_size: Number of test pairs
            resume_path: Optional checkpoint to resume training from
        """
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)

        self.training_pipeline = TrainingPipeline(self.config)
        self.training_pipeline.run_preprocessing()
        self.training_pipeline.create_dataloaders(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size
        )
        self.training_pipeline.setup_model()
        self.training_pipeline.run_training(resume_path=resume_path)

        # Get checkpoint directory from training
        self.checkpoint_dir = self.training_pipeline.checkpoint_dir

        print(f"\n  Training complete. Checkpoint: {self.checkpoint_dir}")

    def run_testing(self) -> None:
        """
        Run testing workflow using TestingPipeline.

        Uses the test_loader and checkpoint from training pipeline.
        """
        print("\n" + "="*70)
        print("TESTING")
        print("="*70)

        if self.training_pipeline is None:
            raise ValueError("Must run training first or provide checkpoint dir.")

        # Create testing pipeline using results from training
        self.testing_pipeline = TestingPipeline(
            config=self.config,
            test_loader=self.training_pipeline.test_loader,
            checkpoint_dir=self.checkpoint_dir
        )

        # Reuse model and metrics from training
        self.testing_pipeline.model = self.training_pipeline.model
        self.testing_pipeline.psnr_metric = self.training_pipeline.psnr_metric
        self.testing_pipeline.ssim_metric = self.training_pipeline.ssim_metric

        # Load from best checkpoint
        self.testing_pipeline.model = load_checkpoint_inference(
            checkpoint_path=self.checkpoint_dir / "best_model.pth",
            model=self.testing_pipeline.model,
            device=self.testing_pipeline.device
        )

        # Run testing workflow
        self.testing_pipeline.run_testing_workflow()

        print(f"\n  Testing complete. ")

    def run_inference(
            self,
            num_images: Optional[int] = None,
            num_examples: int = 3
    ) -> None:
        """
        Run inference workflow using InferencePipeline.

        Args:
            num_images: Number of images to denoise (optional)
            num_examples: Number of comparison examples
        """
        print("\n" + "="*70)
        print("INFERENCE")
        print("="*70)

        if self.checkpoint_dir is None:
            raise ValueError("Must run training first or provide checkpoint dir.")

        self.inference_pipeline = InferencePipeline(
            config=self.config,
            checkpoint_dir=self.checkpoint_dir
        )

        self.inference_pipeline.load_model()
        self.inference_pipeline.run_inference_workflow(
            num_images=num_images,
            num_examples=num_examples
        )

        print(f"\n  Inference complete. ")

    def run_complete_workflow(
            self,
            train_size: int = 1000,
            val_size: int = 150,
            test_size: int = 150,
            num_examples: int = 3,
            resume_path: Optional[str | Path] = None
    ) -> None:
        """
        Run complete workflow: training -> testing -> inference

        Args:
            train_size: Number of training pairs
            val_size: Number of validation pairs
            test_size: Number of test pairs
            num_examples: Number of inference comparison examples
            resume_path: Optional checkpoint to resume training from
        """
        # 1. Training
        self.run_training(
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            resume_path=resume_path
        )

        # 2. Testing
        self.run_testing()

        # 3. Inference
        self.run_inference(
            num_examples=num_examples
        )

        # Final summary
        self.get_summary()

    def get_summary(self) -> None:
        """
        Print summary of complete pipeline
        """
        print("\n" + "="*70)
        print("COMPLETE PIPELINE SUMMARY")
        print("="*70)

        # Training summary
        if self.training_pipeline:
            print("\n   TRAINING:")
            if self.training_pipeline.history:
                best_psnr = max([x for x in self.training_pipeline.history['val_psnr'] if x is not None])
                best_ssim = max([x for x in self.training_pipeline.history['val_ssim'] if x is not None])
                print(f"    Epochs: {len(self.training_pipeline.history['epoch'])}")
                print(f"    Best val PSNR: {best_psnr:.2f} dB")
                print(f"    Best val SSIM: {best_ssim:.4f}")

        # Testing summary
        if self.testing_pipeline and self.testing_pipeline.test_results:
            print("\n   TESTING:")
            print(f"    Test PSNR: {self.testing_pipeline.test_results['avg_psnr']:.2f} dB")
            print(f"    Test SSIM: {self.testing_pipeline.test_results['avg_ssim']:.4f}")
            print(f"    Images tested: {self.testing_pipeline.test_results['num_images']}")

        # Inference summary
        if self.inference_pipeline and self.inference_pipeline.inference_results_dir:
            if self.inference_pipeline.inference_results_dir.exists():
                num_denoised = len(list(self.inference_pipeline.inference_results_dir.glob("*.png")))
                print("\n   INFERENCE:")
                print(f"    Number of images denoised: {num_denoised}")

        # Output location
        if self.checkpoint_dir:
            print(f"\n ALL RESULTS SAVED TO:")
            print(f"  {self.checkpoint_dir}")
            print(f"\n  Contents:")
            print(f"    ├── best_model.pth")
            print(f"    ├── final_model.pth")
            print(f"    ├── training_history.json")
            print(f"    ├── test_results.json")
            print(f"    ├── summary.txt")
            print(f"    ├── figures/")
            print(f"    │   ├── training_metrics.png")
            print(f"    │   ├── test_results_distribution.png")
            print(f"    │   ├── psnr_vs_ssim.png")
            print(f"    │   ├── best_worst_results.png")
            print(f"    │   └── examples/")
            print(f"    ├── test_results/")
            print(f"    └── inference_results/")

        print("\n" + "="*70)
        print(" COMPLETE PIPELINE FINISHED")
        print("="*70)

def run_complete_pipeline(
        config: Config,
        train_size: int = 1000,
        val_size: int = 150,
        test_size: int = 150,
        num_examples: int = 3,
) -> CompletePipeline:
    """
    Run EVERYTHING in one function call: train -> test -> inference.

    Args:
        config: Configuration object
        train_size: Number of training pairs
        val_size: Number of validation pairs
        test_size: Number of test pairs
        num_examples: Number of inference comparison examples

    Returns:
        CompletePipeline with all results
    """
    pipeline = CompletePipeline(config=config)
    pipeline.run_complete_workflow(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        num_examples=num_examples
    )

    return pipeline

